import numpy as np
import torch

from build_dataset import stratified_split
from model import (
    MultiTaskCNN,
    checkpoint_payload,
    load_model_checkpoint,
    quality_from_snr,
    scale_snr_target,
)
from train import rotate_phase


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


def test_quality_from_snr_thresholds():
    snr_db = np.array([-2, 0, 8, 9, 10, 18])

    assert quality_from_snr(snr_db).tolist() == [0, 1, 1, 1, 2, 2]


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
