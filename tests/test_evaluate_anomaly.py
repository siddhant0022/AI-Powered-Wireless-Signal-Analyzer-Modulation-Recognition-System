import json

import numpy as np
import torch

import evaluate_anomaly
from train import main as train_main


def _write_anomaly_dataset(path):
    rng = np.random.default_rng(2026)
    X, mod, snr, split, channel, anomaly, anomaly_type = [], [], [], [], [], [], []
    for split_id, normal_count, anomaly_count in (
        (0, 120, 40), (1, 30, 10), (2, 30, 10),
    ):
        for is_anomaly, count in ((0, normal_count), (1, anomaly_count)):
            for index in range(count):
                X.append(rng.standard_normal((2, 128), dtype=np.float32))
                split.append(split_id)
                channel.append(index % 2)
                anomaly.append(is_anomaly)
                if is_anomaly:
                    type_id = index % 5 + 1
                    anomaly_type.append(type_id)
                    mod.append(index % 11 if type_id <= 3 else -1)
                else:
                    anomaly_type.append(0)
                    mod.append(index % 11)
                snr.append(int(rng.choice(np.arange(-20, 20, 2))))
    snr = np.asarray(snr, dtype=np.int16)
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        mod=np.asarray(mod, dtype=np.int16),
        snr=snr,
        split=np.asarray(split, dtype=np.int8),
        channel=np.asarray(channel, dtype=np.int8),
        anomaly=np.asarray(anomaly, dtype=np.int8),
        anomaly_type=np.asarray(anomaly_type, dtype=np.int8),
        quality=np.where(snr < 0, 0, np.where(snr < 10, 1, 2)).astype(np.int8),
    )
    return path


def test_anomaly_evaluator_runs_on_holdout_smoke_checkpoint(tmp_path, monkeypatch):
    dataset_path = _write_anomaly_dataset(tmp_path / "signals.npz")
    results_dir = tmp_path / "results"
    train_main([
        "--dataset", "simulated",
        "--data", str(dataset_path),
        "--out-dir", str(results_dir),
        "--run-name", "holdout_unknown_ofdm",
        "--epochs", "1",
        "--batch-size", "64",
        "--device", "cpu",
        "--seed", "2026",
        "--use-anomaly",
        "--use-anomaly-type",
        "--holdout-anomaly-type", "unknown_ofdm",
    ])
    monkeypatch.setattr(evaluate_anomaly, "ROOT", str(tmp_path))

    metrics = evaluate_anomaly.evaluate(
        "holdout_unknown_ofdm", torch.device("cpu")
    )
    threshold = metrics["validation_thresholds"]["max_f1"]["threshold"]
    assert metrics["holdout_anomaly_type"] == "unknown_ofdm"
    assert metrics["per_anomaly_type"]["unknown_ofdm"]["exposure"] == "held-out (unseen)"
    assert metrics["table_threshold"] == "max_f1"
    assert metrics["type_classification"]["confusion_matrix"]
    eval_dir = results_dir / "holdout_unknown_ofdm" / "eval_anomaly"
    assert (eval_dir / "eval_anomaly.md").is_file()
    assert (eval_dir / "eval_anomaly_metrics.json").is_file()
    assert (eval_dir / "roc_curves.png").is_file()

    with np.load(dataset_path) as archive:
        changed = {key: archive[key] for key in archive.files}
    changed["X"][changed["split"] == 2] = 0
    np.savez_compressed(dataset_path, **changed)
    changed_test_metrics = evaluate_anomaly.evaluate(
        "holdout_unknown_ofdm", torch.device("cpu")
    )
    assert changed_test_metrics["validation_thresholds"]["max_f1"]["threshold"] == threshold
    with open(eval_dir / "eval_anomaly_metrics.json", encoding="utf-8") as stream:
        assert json.load(stream)["holdout_anomaly_type"] == "unknown_ofdm"
