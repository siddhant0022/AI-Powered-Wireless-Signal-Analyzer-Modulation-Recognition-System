"""Compare simulated Stage A results with the narrower-CFO run.

Usage:
    python compare_cfo_results.py
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
    "Original simulated": ("simulated_stageA", "signal_dataset.npz"),
    "CFO max 0.001": ("simulated_cfo0001", "signal_dataset_cfo0001.npz"),
}
CONFUSION_PAIRS = (("QPSK", "8PSK"), ("QAM16", "QAM64"))


def _load_test_data(path):
    with np.load(path) as archive:
        required = {"X", "mod", "snr", "split", "anomaly"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        mod = archive["mod"]
        keep = (archive["split"] == 2) & (archive["anomaly"] == 0) & (mod >= 0)
        return (
            np.ascontiguousarray(archive["X"][keep], dtype=np.float32),
            np.ascontiguousarray(mod[keep], dtype=np.int64),
            np.ascontiguousarray(archive["snr"][keep], dtype=np.int16),
        )


def _predict(tag, X):
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


def _evaluate(tag, data_name):
    data_path = os.path.join("data", data_name)
    X, targets, snrs = _load_test_data(data_path)
    predictions = _predict(tag, X)
    by_snr = {}
    for snr in sorted(int(value) for value in np.unique(snrs)):
        mask = snrs == snr
        by_snr[snr] = (
            int((predictions[mask] == targets[mask]).sum()),
            int(mask.sum()),
        )

    high_snr = snrs >= 10
    class_ids = {name: MODULATIONS.index(name) for pair in CONFUSION_PAIRS for name in pair}
    pair_counts = {}
    for class_a, class_b in CONFUSION_PAIRS:
        a, b = class_ids[class_a], class_ids[class_b]
        count = int(
            (((targets == a) & (predictions == b)) |
             ((targets == b) & (predictions == a)))[high_snr].sum()
        )
        pair_counts[(class_a, class_b)] = count
    return by_snr, pair_counts, int(high_snr.sum())


def main():
    results = {
        label: _evaluate(tag, dataset)
        for label, (tag, dataset) in RUNS.items()
    }
    snrs = sorted(set().union(*(result[0] for result in results.values())))

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, (by_snr, _, _) in results.items():
        accuracies = [
            correct / count if count else np.nan
            for correct, count in (by_snr.get(snr, (0, 0)) for snr in snrs)
        ]
        ax.plot(snrs, accuracies, marker="o", label=label)
    ax.axhline(1 / len(MODULATIONS), linestyle=":", color="gray",
               label="Chance (1/11)")
    ax.set(
        xlabel="SNR (dB)",
        ylabel="Test accuracy",
        title="Simulated Stage A: original vs narrower CFO",
        ylim=(0, 1),
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join("results", "simulated_cfo0001_accuracy_vs_snr.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    lines = [
        "# Simulated Stage A: original vs narrower CFO",
        "",
        "Both rows are computed from the saved best checkpoints and their own "
        "normal, known-modulation test split.",
        "",
        "## Test accuracy by SNR",
        "",
        "| SNR (dB) | Original simulated (accuracy; correct / n) "
        "| CFO max 0.001 (accuracy; correct / n) |",
        "|---:|---:|---:|",
    ]
    for snr in snrs:
        cells = []
        for label in results:
            correct, count = results[label][0].get(snr, (0, 0))
            cells.append(
                f"{100 * correct / count:.1f}% ({correct}/{count})"
                if count else "N/A"
            )
        lines.append(f"| {snr} | {cells[0]} | {cells[1]} |")

    lines.extend(
        [
            "",
            "## Selected confusions at SNR >= 10 dB",
            "",
            "Pair counts combine both confusion directions. Each run uses all "
            "normal, known-modulation test frames at SNR >= 10 dB (both channels).",
            "",
            "| Class pair | Original simulated | CFO max 0.001 |",
            "|---|---:|---:|",
        ]
    )
    for pair in CONFUSION_PAIRS:
        lines.append(
            f"| {pair[0]} ↔ {pair[1]} | "
            f"{results['Original simulated'][1][pair]} | "
            f"{results['CFO max 0.001'][1][pair]} |"
        )
    lines.extend(
        [
            "",
            "| Test frames at SNR >= 10 dB | Original simulated | CFO max 0.001 |",
            "|---|---:|---:|",
            f"| Count | {results['Original simulated'][2]} "
            f"| {results['CFO max 0.001'][2]} |",
            "",
            f"Accuracy plot: `{os.path.basename(plot_path)}`.",
        ]
    )

    report_path = os.path.join("results", "simulated_cfo0001_comparison.md")
    with open(report_path, "w", encoding="utf-8") as report:
        report.write("\n".join(lines) + "\n")
    print(f"Saved {report_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
