"""Analyze saved Stage A checkpoints by class and simulated channel.

Usage:
    python analyze_extra.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_loader import MODULATIONS
from model import load_model_checkpoint, unpack_model_output


RUNS = {
    "RadioML": ("radioml_stageA", os.path.join("data", "radioml_2016a.npz")),
    "Simulated": ("simulated_stageA", os.path.join("data", "signal_dataset.npz")),
}
LOW_SNRS = (-12, -10, -8)
CHANNEL_NAMES = {0: "AWGN", 1: "Fading"}


def _load_test_data(path, need_channel=False):
    """Load the same known, normal test examples used by Stage A evaluation."""
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
        mod = np.ascontiguousarray(mod[keep], dtype=np.int64)
        snr = np.ascontiguousarray(archive["snr"][keep], dtype=np.int16)
        channel = (
            np.ascontiguousarray(archive["channel"][keep], dtype=np.int8)
            if need_channel else None
        )

    if not len(mod):
        raise ValueError(f"{path} has no known, normal test examples")
    if need_channel and not np.isin(channel, list(CHANNEL_NAMES)).all():
        raise ValueError(f"{path} contains test frames without a known channel label")
    return X, mod, snr, channel


def _predict(tag, X):
    """Return predictions from a run's saved best checkpoint without retraining."""
    checkpoint_path = os.path.join("results", tag, "best_model.pt")
    if not os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.join("results", tag, "best.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Saved checkpoint not found: {checkpoint_path}")
    model, _ = load_model_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(X), 256):
            batch = torch.from_numpy(X[start:start + 256])
            logits, _, _, _ = unpack_model_output(model(batch))
            predictions.append(logits.argmax(dim=1).numpy())
    return np.concatenate(predictions)


def _cell_text(correct, count):
    if count == 0:
        return "N/A (0/0)"
    return f"{100 * correct / count:.1f}% ({correct}/{count})"


def _class_accuracy(targets, predictions, snrs, modulation, snr):
    mask = (targets == modulation) & (snrs == snr)
    count = int(mask.sum())
    correct = int((predictions[mask] == targets[mask]).sum())
    return correct, count


def _write_low_snr_report(results):
    lines = [
        "# Stage A per-class accuracy at low SNR",
        "",
        "Values show accuracy followed by correct predictions / test-frame count.",
        "Each cell uses only that modulation's normal, known-modulation test frames "
        "at the listed SNR.",
    ]
    for snr in LOW_SNRS:
        lines += [
            "",
            f"## SNR {snr} dB",
            "",
            "| Modulation | RadioML accuracy (correct / n) | Simulated accuracy (correct / n) |",
            "|---|---:|---:|",
        ]
        for modulation, name in enumerate(MODULATIONS):
            cells = []
            for dataset in ("RadioML", "Simulated"):
                result = results[dataset]
                correct, count = _class_accuracy(
                    result["targets"], result["predictions"],
                    result["snrs"], modulation, snr,
                )
                cells.append(_cell_text(correct, count))
            lines.append(f"| {name} | {cells[0]} | {cells[1]} |")

    output_path = os.path.join("results", "stageA_per_class_low_snr.md")
    with open(output_path, "w", encoding="utf-8") as report:
        report.write("\n".join(lines) + "\n")
    return output_path


def _write_channel_report(result):
    targets, predictions = result["targets"], result["predictions"]
    snrs, channels = result["snrs"], result["channels"]
    snr_levels = sorted(int(value) for value in np.unique(snrs))
    rows = []
    for snr in snr_levels:
        row = {"snr": snr}
        for channel, name in CHANNEL_NAMES.items():
            mask = (snrs == snr) & (channels == channel)
            count = int(mask.sum())
            correct = int((predictions[mask] == targets[mask]).sum())
            row[name] = (correct, count)
        rows.append(row)

    lines = [
        "# Simulated Stage A accuracy by channel",
        "",
        "Values show accuracy followed by correct predictions / test-frame count.",
        "Only normal, known-modulation test frames with a labeled AWGN or fading "
        "channel are included.",
        "",
        "| SNR (dB) | AWGN accuracy (correct / n) | Fading accuracy (correct / n) |",
        "|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['snr']} | {_cell_text(*row['AWGN'])} "
            f"| {_cell_text(*row['Fading'])} |"
        )

    output_path = os.path.join("results", "stageA_simulated_by_channel.md")
    with open(output_path, "w", encoding="utf-8") as report:
        report.write("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in CHANNEL_NAMES.values():
        accuracies = [
            correct / count if count else np.nan
            for correct, count in (row[name] for row in rows)
        ]
        ax.plot(snr_levels, accuracies, marker="o", label=name)
    ax.set(
        xlabel="SNR (dB)",
        ylabel="Test accuracy",
        title="Simulated Stage A accuracy by channel",
        ylim=(0, 1),
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join("results", "stageA_simulated_by_channel.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return output_path, plot_path, rows


def main():
    results = {}
    for dataset, (tag, data_path) in RUNS.items():
        X, targets, snrs, channels = _load_test_data(
            data_path, need_channel=(dataset == "Simulated")
        )
        results[dataset] = {
            "targets": targets,
            "snrs": snrs,
            "channels": channels,
            "predictions": _predict(tag, X),
        }
        print(f"{dataset}: {len(targets)} test frames evaluated from {data_path}")

    low_snr_path = _write_low_snr_report(results)
    channel_report_path, channel_plot_path, channel_rows = _write_channel_report(
        results["Simulated"]
    )

    print(f"Saved {low_snr_path}")
    for snr in LOW_SNRS:
        for modulation, name in enumerate(MODULATIONS):
            counts = []
            for dataset in ("RadioML", "Simulated"):
                result = results[dataset]
                _, count = _class_accuracy(
                    result["targets"], result["predictions"],
                    result["snrs"], modulation, snr,
                )
                counts.append(count)
            print(
                f"SNR {snr:>3} dB, {name:<7}: "
                f"RadioML n={counts[0]}, Simulated n={counts[1]}"
            )
    print(f"Saved {channel_report_path}")
    print("Simulated per-channel cell counts:")
    for row in channel_rows:
        print(
            f"SNR {row['snr']:>3} dB: "
            f"AWGN n={row['AWGN'][1]}, Fading n={row['Fading'][1]}"
        )
    print(f"Saved {channel_plot_path}")


if __name__ == "__main__":
    main()
