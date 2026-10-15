"""Analyze saved Stage A prediction distributions and AWGN confusions.

Usage:
    python analyze_extra2.py
"""
import os

import numpy as np
import torch

from dataset_loader import MODULATIONS
from model import MultiTaskCNN


RUNS = {
    "RadioML": ("radioml_stageA", os.path.join("data", "radioml_2016a.npz")),
    "Simulated": ("simulated_stageA", os.path.join("data", "signal_dataset.npz")),
}
LOW_SNRS = (-12, -10)
AWGN = 0


def _load_test_data(path, need_channel=False):
    """Load known-modulation, normal test examples and their labels."""
    with np.load(path) as archive:
        required = {"X", "mod", "snr", "split"}
        if need_channel:
            required.add("channel")
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")

        mod = archive["mod"]
        keep = (archive["split"] == 2) & (mod >= 0)
        if "anomaly" in archive.files:
            keep &= archive["anomaly"] == 0
        X = np.ascontiguousarray(archive["X"][keep], dtype=np.float32)
        targets = np.ascontiguousarray(mod[keep], dtype=np.int64)
        snrs = np.ascontiguousarray(archive["snr"][keep], dtype=np.int16)
        channels = (
            np.ascontiguousarray(archive["channel"][keep], dtype=np.int8)
            if need_channel else None
        )

    if not len(targets):
        raise ValueError(f"{path} has no known, normal test frames")
    if need_channel and not np.isin(channels, (0, 1)).all():
        raise ValueError(f"{path} contains test frames without a known channel")
    return X, targets, snrs, channels


def _predict(tag, X):
    """Predict with the saved best checkpoint; this does not train or update it."""
    checkpoint_path = os.path.join("results", tag, "best_model.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Saved checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"{checkpoint_path} has no model_state_dict")

    model = MultiTaskCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(X), 256):
            batch = torch.from_numpy(X[start:start + 256])
            outputs.append(model(batch).argmax(dim=1).numpy())
    return np.concatenate(outputs)


def _confusion_counts(targets, predictions):
    confusion = np.zeros((len(MODULATIONS), len(MODULATIONS)), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    return confusion


def _format_precision(correct, predicted):
    if predicted == 0:
        return "N/A (0 predicted)"
    return f"{100 * correct / predicted:.1f}% ({correct}/{predicted})"


def _append_distribution(lines, dataset, snr, targets, predictions, snrs):
    mask = snrs == snr
    selected_targets = targets[mask]
    selected_predictions = predictions[mask]
    confusion = _confusion_counts(selected_targets, selected_predictions)
    predicted_counts = confusion.sum(axis=0)
    true_counts = confusion.sum(axis=1)
    lines.extend(
        [
            "",
            f"## {dataset}: SNR {snr} dB ({int(mask.sum())} test frames)",
            "",
            "Precision is correct predictions for the class divided by all frames "
            "predicted as that class. Counts are restricted to this SNR.",
            "",
            "| Class | Predicted count | True count | Precision (correct / predicted) |",
            "|---|---:|---:|---:|",
        ]
    )
    for class_id, name in enumerate(MODULATIONS):
        predicted = int(predicted_counts[class_id])
        true = int(true_counts[class_id])
        correct = int(confusion[class_id, class_id])
        lines.append(
            f"| {name} | {predicted} | {true} "
            f"| {_format_precision(correct, predicted)} |"
        )


def _top_confused_pairs(targets, predictions):
    confusion = _confusion_counts(targets, predictions)
    pairs = []
    for class_a in range(len(MODULATIONS)):
        for class_b in range(class_a + 1, len(MODULATIONS)):
            count = int(confusion[class_a, class_b] + confusion[class_b, class_a])
            pairs.append((count, class_a, class_b))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    return pairs[:5]


def main():
    evaluated = {}
    for dataset, (tag, data_path) in RUNS.items():
        X, targets, snrs, channels = _load_test_data(
            data_path, need_channel=(dataset == "Simulated")
        )
        predictions = _predict(tag, X)
        evaluated[dataset] = (targets, predictions, snrs, channels)
        print(f"{dataset}: generated predictions for {len(targets)} test frames")

    lines = [
        "# Stage A prediction distribution",
        "",
        "Predictions were generated from the saved best checkpoints and the original "
        "test splits. No training was performed.",
    ]
    for dataset, (targets, predictions, snrs, _) in evaluated.items():
        for snr in LOW_SNRS:
            _append_distribution(lines, dataset, snr, targets, predictions, snrs)

    targets, predictions, snrs, channels = evaluated["Simulated"]
    awgn_high_snr = (channels == AWGN) & (snrs >= 10)
    frame_count = int(awgn_high_snr.sum())
    pairs = _top_confused_pairs(targets[awgn_high_snr], predictions[awgn_high_snr])
    lines.extend(
        [
            "",
            "## Simulated AWGN confusions at SNR >= 10 dB",
            "",
            f"Test frames used: **{frame_count}** (normal, known-modulation test "
            "frames on the AWGN channel with SNR >= 10 dB).",
            "",
            "Counts combine both directions for each unordered pair.",
            "",
            "| Rank | Class pair | Two-way confusion count |",
            "|---:|---|---:|",
        ]
    )
    for rank, (count, class_a, class_b) in enumerate(pairs, start=1):
        lines.append(
            f"| {rank} | {MODULATIONS[class_a]} ↔ {MODULATIONS[class_b]} | {count} |"
        )

    output_path = os.path.join("results", "stageA_prediction_distribution.md")
    with open(output_path, "w", encoding="utf-8") as report:
        report.write("\n".join(lines) + "\n")
    print(f"Simulated AWGN, SNR >= 10 dB: {frame_count} test frames")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
