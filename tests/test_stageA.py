import csv
import json

import numpy as np
import pytest
import torch

from build_dataset import stratified_split
from model import (
    MultiTaskCNN,
    checkpoint_payload,
    load_model_checkpoint,
    quality_from_snr,
    scale_snr_target,
)
from simulator import ANOMALY_TYPES
from train import (
    _anomaly_losses,
    _make_loader,
    _run_epoch,
    load_anomaly_training_splits,
    load_training_splits,
    main,
    rotate_phase,
)


def _write_small_training_dataset(path, frame_count=240):
    rng = np.random.default_rng(2026)
    X = rng.standard_normal((frame_count, 2, 128), dtype=np.float32)
    mod = (np.arange(frame_count) % 11).astype(np.int16)
    snr = np.resize(np.array([-20, -10, 0, 10, 18], dtype=np.int16), frame_count)
    train_end = int(frame_count * 0.70)
    val_end = int(frame_count * 0.85)
    split = np.concatenate((
        np.zeros(train_end, dtype=np.int8),
        np.ones(val_end - train_end, dtype=np.int8),
        np.full(frame_count - val_end, 2, dtype=np.int8),
    ))
    np.savez_compressed(
        path, X=X, mod=mod, snr=snr, split=split,
        anomaly=np.zeros(frame_count, dtype=np.int8),
    )
    return path


def _small_run_args(dataset_path, out_dir, run_name, epochs, *extra):
    return [
        "--dataset", "simulated",
        "--data", str(dataset_path),
        "--out-dir", str(out_dir),
        "--run-name", run_name,
        "--epochs", str(epochs),
        "--batch-size", "64",
        "--device", "cpu",
        "--seed", "2026",
        *extra,
    ]


def _write_small_anomaly_dataset(path):
    rng = np.random.default_rng(2026)
    split_parts = []
    for split_id, normal_count, anomaly_count in (
        (0, 120, 40), (1, 30, 10), (2, 30, 10),
    ):
        split_parts.extend([
            (split_id, 0, normal_count),
            (split_id, 1, anomaly_count),
        ])
    X, mod, snr, split, anomaly, anomaly_type = [], [], [], [], [], []
    for split_id, is_anomaly, count in split_parts:
        for index in range(count):
            X.append(rng.standard_normal((2, 128), dtype=np.float32))
            split.append(split_id)
            anomaly.append(is_anomaly)
            if is_anomaly:
                type_id = index % 5 + 1
                anomaly_type.append(type_id)
                mod.append(index % 11 if type_id <= 3 else -1)
            else:
                anomaly_type.append(0)
                mod.append(index % 11)
            snr.append(int(rng.choice(np.arange(-20, 20, 2))))
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        mod=np.asarray(mod, dtype=np.int16),
        snr=np.asarray(snr, dtype=np.int16),
        split=np.asarray(split, dtype=np.int8),
        anomaly=np.asarray(anomaly, dtype=np.int8),
        anomaly_type=np.asarray(anomaly_type, dtype=np.int8),
    )
    return path


def test_modulation_output_shape():
    model = MultiTaskCNN()
    assert model(torch.randn(4, 2, 128)).shape == (4, 11)


def test_snr_enabled_output_shapes_and_db_prediction():
    model = MultiTaskCNN(use_snr=True).eval()
    inputs = torch.randn(4, 2, 128)
    mod_logits, snr_pred = model(inputs)

    assert mod_logits.shape == (4, 11)
    assert snr_pred.shape == (4, 1)
    assert torch.equal(model.predict_snr_db(inputs), snr_pred * 20)
    assert scale_snr_target(torch.tensor([-20.0, 0.0, 20.0])).tolist() == [-1, 0, 1]


def test_anomaly_model_output_keys_and_shapes():
    model = MultiTaskCNN(
        use_snr=True, use_anomaly=True, use_anomaly_type=True
    ).eval()
    outputs = model(torch.randn(4, 2, 128))

    assert set(outputs) == {
        "mod_logits", "snr_pred", "anomaly_logit", "type_logits",
    }
    assert outputs["mod_logits"].shape == (4, 11)
    assert outputs["snr_pred"].shape == (4, 1)
    assert outputs["anomaly_logit"].shape == (4, 1)
    assert outputs["type_logits"].shape == (4, 5)


def test_default_snr_loss_beta_matches_smooth_l1_default():
    torch.manual_seed(2026)
    model = MultiTaskCNN(use_snr=True).eval()
    X = np.random.default_rng(2026).standard_normal((4, 2, 128)).astype(np.float32)
    mod = np.array([0, 1, 2, 3], dtype=np.int64)
    snr_db = np.array([-20, -2, 10, 18], dtype=np.int16)
    loader = _make_loader((X, mod, snr_db), 4, shuffle=False, use_snr=True)
    criterion = torch.nn.CrossEntropyLoss()
    inputs, mod_target, snr_target = next(iter(loader))

    with torch.inference_mode():
        mod_logits, snr_pred = model(inputs)
        expected_loss = criterion(mod_logits, mod_target) + torch.nn.functional.smooth_l1_loss(
            snr_pred.squeeze(1), snr_target
        )
    actual_loss, _, _ = _run_epoch(
        model, loader, criterion, torch.device("cpu"), use_snr=True
    )

    assert actual_loss == pytest.approx(expected_loss.item())


def test_parameter_count_is_in_expected_range():
    model = MultiTaskCNN()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 200_000 <= parameter_count <= 260_000


def test_parameter_counts_for_both_configurations():
    modulation_only = MultiTaskCNN()
    with_snr = MultiTaskCNN(use_snr=True)

    assert sum(parameter.numel() for parameter in modulation_only.parameters()) == 220_427
    assert sum(parameter.numel() for parameter in with_snr.parameters()) == 228_748


def test_legacy_stage_a_checkpoint_loads_without_config(tmp_path):
    torch.manual_seed(2026)
    original = MultiTaskCNN().eval()
    path = tmp_path / "legacy_stageA.pt"
    torch.save({"model_state_dict": original.state_dict(), "epoch": 1}, path)

    loaded, checkpoint = load_model_checkpoint(path)
    loaded.eval()
    inputs = torch.randn(4, 2, 128)

    assert "config" not in checkpoint
    assert torch.equal(original(inputs), loaded(inputs))


def test_new_checkpoint_contains_config_and_loads(tmp_path):
    original = MultiTaskCNN(use_snr=True)
    path = tmp_path / "configured.pt"
    torch.save(checkpoint_payload(original, epoch=1), path)

    loaded, checkpoint = load_model_checkpoint(path)

    assert checkpoint["config"]["use_snr"] is True
    assert loaded.get_config() == original.get_config()


def test_anomaly_checkpoint_round_trip(tmp_path):
    original = MultiTaskCNN(
        use_snr=True, use_anomaly=True, use_anomaly_type=True
    ).eval()
    path = tmp_path / "anomaly_configured.pt"
    torch.save(checkpoint_payload(original, epoch=1), path)

    loaded, checkpoint = load_model_checkpoint(path)
    loaded.eval()
    inputs = torch.randn(4, 2, 128)

    assert checkpoint["config"]["use_anomaly"] is True
    assert checkpoint["config"]["use_anomaly_type"] is True
    for key, value in original(inputs).items():
        assert torch.equal(value, loaded(inputs)[key])


def test_quality_from_snr_thresholds():
    snr_db = np.array([-2, 0, 8, 9, 10, 18])

    assert quality_from_snr(snr_db).tolist() == [0, 1, 1, 1, 2, 2]


def test_anomaly_losses_mask_mod_snr_and_type_targets():
    torch.manual_seed(2026)
    outputs = {
        "mod_logits": torch.randn(4, 11),
        "snr_pred": torch.randn(4, 1),
        "anomaly_logit": torch.randn(4, 1),
        "type_logits": torch.randn(4, 5),
    }
    mod_target = torch.tensor([0, 1, -1, -1])
    snr_target = torch.tensor([-1.0, 0.5, 0.2, -0.3])
    anomaly_target = torch.tensor([0.0, 0.0, 1.0, 1.0])
    type_target = torch.tensor([0, 0, 1, 5])
    pos_weight = torch.tensor(1.0)
    initial = _anomaly_losses(
        outputs, mod_target, snr_target, anomaly_target, type_target,
        pos_weight, use_anomaly_type=True,
    )

    changed_targets = _anomaly_losses(
        outputs,
        torch.tensor([0, 1, 999, 999]),
        torch.tensor([-1.0, 0.5, 999.0, -999.0]),
        anomaly_target,
        torch.tensor([999, -999, 1, 5]),
        pos_weight,
        use_anomaly_type=True,
    )

    assert torch.equal(initial["mod"], changed_targets["mod"])
    assert torch.equal(initial["snr"], changed_targets["snr"])
    assert torch.equal(initial["type"], changed_targets["type"])


def test_holdout_anomaly_type_is_excluded_from_train_and_val(tmp_path):
    dataset_path = _write_small_anomaly_dataset(tmp_path / "anomaly.npz")

    splits = load_anomaly_training_splits(dataset_path, "unknown_ofdm")

    heldout_type_id = ANOMALY_TYPES.index("unknown_ofdm")
    assert not np.any(splits["train"][4] == heldout_type_id)
    assert not np.any(splits["val"][4] == heldout_type_id)
    assert np.any(splits["train"][3] == 1)
    assert np.any(splits["val"][3] == 1)


def test_all_heads_anomaly_training_smoke(tmp_path):
    dataset_path = _write_small_anomaly_dataset(tmp_path / "anomaly.npz")
    args = [
        "--dataset", "simulated",
        "--data", str(dataset_path),
        "--out-dir", str(tmp_path),
        "--run-name", "anomaly-smoke",
        "--epochs", "1",
        "--batch-size", "64",
        "--device", "cpu",
        "--seed", "2026",
        "--use-anomaly",
        "--use-anomaly-type",
    ]

    output_dir = main(args)

    assert (tmp_path / "anomaly-smoke" / "best.pt").is_file()
    assert (tmp_path / "anomaly-smoke" / "last.pt").is_file()
    checkpoint = torch.load(
        tmp_path / "anomaly-smoke" / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["config"]["use_snr"] is True
    assert checkpoint["config"]["use_anomaly"] is True
    assert checkpoint["config"]["use_anomaly_type"] is True
    assert np.isfinite(checkpoint["val_metrics"]["total_loss"])
    assert output_dir == str(tmp_path / "anomaly-smoke")
    with open(tmp_path / "anomaly-smoke" / "history.csv", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert all(np.isfinite(float(rows[0][name])) for name in (
        "train_loss", "val_loss", "val_mod_acc", "val_snr_mae_db",
        "val_anomaly_f1", "val_type_acc",
    ))


def test_snr_training_smoke_writes_checkpoints_and_finite_losses(tmp_path):
    dataset_path = _write_small_training_dataset(tmp_path / "small.npz")
    train_data = load_training_splits(dataset_path)["train"]
    batch = next(iter(_make_loader(
        train_data, 32, shuffle=False, device="cpu", use_snr=True
    )))
    assert batch[2].dtype == torch.float32
    assert batch[2].shape == (32,)

    output_dir = main(_small_run_args(
        dataset_path, tmp_path, "snr-smoke", 1, "--use-snr"
    ))
    assert (tmp_path / "snr-smoke" / "best.pt").is_file()
    assert (tmp_path / "snr-smoke" / "last.pt").is_file()
    with open(tmp_path / "snr-smoke" / "history.csv", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert all(np.isfinite(float(rows[0][key])) for key in (
        "train_loss", "val_loss", "val_mod_acc", "val_snr_mae_db",
    ))
    checkpoint = torch.load(
        tmp_path / "snr-smoke" / "last.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["epoch"] == 1
    assert checkpoint["config"]["use_snr"] is True
    assert checkpoint["config"]["snr_loss_beta"] == 1.0
    assert {
        "optimizer_state_dict", "scheduler_state_dict", "rng_states", "best_score",
    }.issubset(checkpoint)
    assert (tmp_path / "snr-smoke" / "config.json").is_file()
    with open(tmp_path / "snr-smoke" / "config.json", encoding="utf-8") as stream:
        assert json.load(stream)["snr_loss_beta"] == 1.0
    assert output_dir == str(tmp_path / "snr-smoke")


def test_resume_continues_epoch_and_history(tmp_path):
    dataset_path = _write_small_training_dataset(tmp_path / "small.npz")
    main(_small_run_args(
        dataset_path, tmp_path, "resume-smoke", 2, "--use-snr"
    ))
    checkpoint_path = tmp_path / "resume-smoke" / "last.pt"
    before_resume = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )["model_state_dict"]["modulation_head.weight"].clone()
    main(_small_run_args(
        dataset_path, tmp_path, "resume-smoke", 3, "--use-snr", "--resume"
    ))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    with open(tmp_path / "resume-smoke" / "history.csv", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert checkpoint["epoch"] == 3
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3]
    assert not torch.equal(
        before_resume, checkpoint["model_state_dict"]["modulation_head.weight"]
    )


def test_default_training_saves_modulation_only_checkpoint(tmp_path):
    dataset_path = _write_small_training_dataset(tmp_path / "small.npz")
    main(_small_run_args(dataset_path, tmp_path, "modulation-only", 1))

    checkpoint = torch.load(
        tmp_path / "modulation-only" / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model, _ = load_model_checkpoint(tmp_path / "modulation-only" / "best.pt")
    assert checkpoint["config"]["use_snr"] is False
    assert model(torch.zeros(4, 2, 128)).shape == (4, 11)
    assert (tmp_path / "modulation-only" / "best_model.pt").is_file()


def test_phase_rotation_preserves_unit_power():
    generator = torch.Generator().manual_seed(2026)
    signals = torch.randn(8, 2, 128, generator=generator)
    power = torch.mean(torch.sum(signals ** 2, dim=1), dim=1)
    signals = signals / torch.sqrt(power)[:, None, None]
    angles = torch.rand(8, generator=generator) * (2 * torch.pi)

    rotated = rotate_phase(signals, angles)
    rotated_power = torch.mean(torch.sum(rotated ** 2, dim=1), dim=1)
    assert torch.allclose(rotated_power, torch.ones_like(rotated_power), atol=1e-6)


def test_stratified_split_assigns_each_frame_to_one_split():
    strata = np.repeat(np.arange(4), 20)
    split = stratified_split(strata, np.random.default_rng(2026))
    indices = [set(np.flatnonzero(split == split_id)) for split_id in (0, 1, 2)]

    assert indices[0].isdisjoint(indices[1])
    assert indices[0].isdisjoint(indices[2])
    assert indices[1].isdisjoint(indices[2])
    assert set.union(*indices) == set(range(len(strata)))
    for stratum in np.unique(strata):
        assignments = split[strata == stratum]
        assert set(assignments) == {0, 1, 2}
