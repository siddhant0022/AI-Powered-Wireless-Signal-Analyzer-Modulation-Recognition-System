import numpy as np
import torch

import evaluate_snr
from model import MultiTaskCNN, checkpoint_payload


def test_snr_evaluator_accepts_all_model_output_formats(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    run_dir = results_root / "format-test"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(evaluate_snr, "ROOT", str(tmp_path))

    frames = np.random.default_rng(2026).standard_normal((12, 2, 128)).astype(np.float32)
    mods = np.arange(12, dtype=np.int16) % 11
    snrs = np.resize(np.array([-20, -10, 0, 10, 18], dtype=np.int16), 12)
    channels = np.zeros(12, dtype=np.int8)
    monkeypatch.setattr(
        evaluate_snr,
        "load_dataset",
        lambda *args, **kwargs: {
            "X": frames,
            "mod": mods,
            "snr": snrs,
            "channel": channels,
        },
    )
    monkeypatch.setattr(
        evaluate_snr,
        "load_stage_a_baseline",
        lambda dataset: {
            "snr_ge_10_db": 0.5,
            "snr_0_to_9_db": 0.5,
            "snr_lt_0_db": 0.5,
            "source": "test",
        },
    )
    model_configs = (
        ("modulation-only", {"use_snr": False}),
        ("snr-only", {"use_snr": True}),
        ("anomaly", {"use_snr": True, "use_anomaly": True}),
    )
    for checkpoint_name, config in model_configs:
        model = MultiTaskCNN(**config).eval()
        if checkpoint_name == "modulation-only":
            torch.save(
                {"model_state_dict": model.state_dict()},
                run_dir / "best_model.pt",
            )
            requested_checkpoint = "best.pt"
        else:
            torch.save(checkpoint_payload(model), run_dir / f"{checkpoint_name}.pt")
            requested_checkpoint = f"{checkpoint_name}.pt"
        metrics, _, _, _, snr_predictions = evaluate_snr._evaluate_checkpoint(
            "format-test",
            "simulated",
            torch.device("cpu"),
            requested_checkpoint,
        )
        if config["use_snr"]:
            assert snr_predictions.shape == (len(frames),)
            assert metrics["snr_estimation"] is not None
        else:
            assert snr_predictions is None
            assert metrics["snr_estimation"] is None
