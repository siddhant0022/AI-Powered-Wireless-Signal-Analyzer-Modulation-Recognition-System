"""Summarize Stage A runs and compare their test accuracy versus SNR.

Usage:
    python compare_results.py
"""
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_loader import MODULATIONS
from model import MultiTaskCNN


RUNS = {
    "RadioML": ("radioml_stageA", os.path.join("data", "radioml_2016a.npz")),
    "Simulated": ("simulated_stageA", os.path.join("data", "signal_dataset.npz")),
}


def _load_metrics(tag):
    folder = os.path.join("results", tag)
    with open(os.path.join(folder, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    with open(os.path.join(folder, "history.json"), encoding="utf-8") as f:
        history = json.load(f)
    with open(os.path.join(folder, "train_log.txt"), "rb") as f:
        log_bytes = f.read()
    log_encoding = "utf-16" if log_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    log = log_bytes.decode(log_encoding)

    device_match = re.search(r"Device:\s*([^;]+)", log)
    if not device_match:
        raise ValueError(f"Cannot find device in {folder}\\train_log.txt")
    epoch_durations = history["seconds_per_epoch"]
    return {
        "folder": folder,
        "summary": summary,
        "history": history,
        "device": device_match.group(1).strip(),
        "parameters": sum(parameter.numel() for parameter in MultiTaskCNN().parameters()),
        "epochs": len(history["epoch"]),
        "best_epoch": history["best_epoch"],
        "training_seconds": float(sum(epoch_durations)),
    }


def _test_arrays(path):
    with np.load(path) as archive:
        X, mod, snr, split = archive["X"], archive["mod"], archive["snr"], archive["split"]
        normal = archive["anomaly"] == 0 if "anomaly" in archive.files else np.ones(len(mod), bool)
        keep = (split == 2) & normal & (mod >= 0)
        return (
            np.ascontiguousarray(X[keep], dtype=np.float32),
            np.ascontiguousarray(mod[keep], dtype=np.int64),
            np.ascontiguousarray(snr[keep], dtype=np.int16),
        )


def _confused_pairs(tag, data_path, device):
    """Use the best checkpoint's test predictions to rank class confusions."""
    folder = os.path.join("results", tag)
    X, targets, snr = _test_arrays(data_path)
    model = MultiTaskCNN().to(device)
    checkpoint = torch.load(
        os.path.join(folder, "best_model.pt"), map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(X), 256):
            batch = torch.from_numpy(X[start:start + 256]).to(device)
            predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    predictions = np.concatenate(predictions)

    high_snr = snr >= 10
    confusion = np.zeros((11, 11), dtype=np.int64)
    np.add.at(confusion, (targets[high_snr], predictions[high_snr]), 1)
    ranked_pairs = []
    for true_class in range(11):
        for predicted_class in range(true_class + 1, 11):
            count = int(confusion[true_class, predicted_class] +
                        confusion[predicted_class, true_class])
            ranked_pairs.append((count, true_class, predicted_class))
    ranked_pairs.sort(reverse=True)
    return ranked_pairs[:3]


def _read_curve(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(int(row["snr_db"]), float(row["accuracy"])) for row in csv.DictReader(f)]


def main():
    results = {}
    for name, (tag, data_path) in RUNS.items():
        results[name] = {
            **_load_metrics(tag),
            "tag": tag,
            "data_path": data_path,
            "curve": _read_curve(os.path.join("results", tag, "accuracy_vs_snr.csv")),
            "confused_pairs": _confused_pairs(tag, data_path, "cpu"),
        }

    output_dir = "results"
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, result in results.items():
        curve = result["curve"]
        ax.plot([item[0] for item in curve], [item[1] for item in curve],
                marker="o", label=name)
    ax.axhline(1 / 11, linestyle=":", color="gray", label="Chance (1/11)")
    ax.set(
        xlabel="SNR (dB)",
        ylabel="Test accuracy",
        title="Stage A: test accuracy vs SNR",
        ylim=(0, 1),
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "stageA_both_accuracy_vs_snr.png"), dpi=150)
    plt.close(fig)

    lines = [
        "# Stage A results",
        "",
        "## Test accuracy",
        "",
        "| Dataset | Overall test accuracy | SNR >= 10 dB | SNR < 0 dB | Test frames |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        summary = result["summary"]
        lines.append(
            f"| {name} | {summary['test_accuracy']:.4f} "
            f"| {summary['test_accuracy_snr_ge_10_db']:.4f} "
            f"| {summary['test_accuracy_snr_lt_0_db']:.4f} "
            f"| {summary['test_frames']} |"
        )

    snrs = sorted(set(snr for result in results.values() for snr, _ in result["curve"]))
    curve_maps = {
        name: dict(result["curve"]) for name, result in results.items()
    }
    lines += [
        "",
        "## Accuracy by SNR",
        "",
        "| SNR (dB) | RadioML | Simulated |",
        "|---:|---:|---:|",
    ]
    for snr in snrs:
        radio_acc = curve_maps["RadioML"].get(snr)
        simulated_acc = curve_maps["Simulated"].get(snr)
        lines.append(
            f"| {snr} | {radio_acc:.4f}" if radio_acc is not None else f"| {snr} | N/A"
        )
        lines[-1] += f" | {simulated_acc:.4f} |" if simulated_acc is not None else " | N/A |"

    lines += [
        "",
        "## Training details",
        "",
        "| Dataset | Epochs trained | Best epoch | Parameters | Device | Training time (s) | Test frames |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['epochs']} | {result['best_epoch']} "
            f"| {result['parameters']} | {result['device']} "
            f"| {result['training_seconds']:.1f} | {result['summary']['test_frames']} |"
        )

    lines += ["", "## Most confused class pairs at SNR >= 10 dB", ""]
    for name, result in results.items():
        lines += [f"### {name}", "", "| Class pair | Two-way confusions |", "|---|---:|"]
        for count, class_a, class_b in result["confused_pairs"]:
            lines.append(
                f"| {MODULATIONS[class_a]} ↔ {MODULATIONS[class_b]} | {count} |"
            )
        lines.append("")

    summary_path = os.path.join(output_dir, "stageA_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    print(f"Saved results/stageA_both_accuracy_vs_snr.png")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
