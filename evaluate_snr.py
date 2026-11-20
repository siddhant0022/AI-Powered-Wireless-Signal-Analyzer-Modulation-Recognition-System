"""Evaluate a trained modulation/SNR model on the held-out test split."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_loader import CHANNEL_NAMES, MODULATIONS, QUALITY_NAMES, load_dataset
from model import load_model_checkpoint, quality_from_snr, unpack_model_output
from snr_metrics import (
    modulation_accuracy_by_snr_range,
    per_snr_regression_metrics,
    quality_confusion_metrics,
    regression_metrics,
)


ROOT = os.path.dirname(os.path.abspath(__file__))
SNR_LEVELS = tuple(range(-20, 20, 2))
SNR_RANGES = (
    ("snr_ge_10_db", "SNR >= 10 dB"),
    ("snr_0_to_9_db", "SNR 0-9 dB"),
    ("snr_lt_0_db", "SNR < 0 dB"),
)


def _compute_metrics(mod_true, mod_pred, snr_true, snr_pred, modulation_names):
    range_accuracy = modulation_accuracy_by_snr_range(
        snr_true, mod_true, mod_pred
    )
    per_snr_accuracy = {}
    for level in SNR_LEVELS:
        mask = snr_true == level
        per_snr_accuracy[str(level)] = {
            "accuracy": (
                float(np.mean(mod_true[mask] == mod_pred[mask]))
                if np.any(mask) else None
            ),
            "count": int(mask.sum()),
        }

    per_modulation = {}
    for mod_id, name in enumerate(modulation_names):
        mask = mod_true == mod_id
        per_modulation[name] = {
            "accuracy": (
                float(np.mean(mod_true[mask] == mod_pred[mask]))
                if np.any(mask) else None
            ),
            "count": int(mask.sum()),
        }

    if snr_pred is None:
        snr_estimation = None
        quality_metrics = None
    else:
        true_quality = np.asarray(quality_from_snr(snr_true), dtype=np.int64)
        estimated_quality = np.asarray(quality_from_snr(snr_pred), dtype=np.int64)
        snr_estimation = {
            "overall": regression_metrics(snr_true, snr_pred),
            "per_true_snr": per_snr_regression_metrics(
                snr_true, snr_true, snr_pred, levels=SNR_LEVELS
            ),
        }
        quality_metrics = quality_confusion_metrics(true_quality, estimated_quality)
    return {
        "test_frames": int(len(mod_true)),
        "modulation": {
            "overall_accuracy": float(np.mean(mod_true == mod_pred)),
            "accuracy_by_snr_range": range_accuracy,
            "accuracy_by_snr": per_snr_accuracy,
            "per_modulation": per_modulation,
        },
        "snr_estimation": snr_estimation,
        "quality": quality_metrics,
    }


def _baseline_from_json(path):
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if "test_accuracy_snr_ge_10_db" not in data or "test_accuracy_snr_lt_0_db" not in data:
        return None
    per_snr = data.get("per_snr", {})
    middle_levels = [per_snr.get(str(level), {}) for level in (0, 2, 4, 6, 8)]
    if not all("accuracy" in level for level in middle_levels):
        return None
    counts = [int(level.get("count", 1)) for level in middle_levels]
    middle_accuracy = float(np.average(
        [level["accuracy"] for level in middle_levels], weights=counts
    ))
    return {
        "snr_ge_10_db": float(data["test_accuracy_snr_ge_10_db"]),
        "snr_0_to_9_db": middle_accuracy,
        "snr_lt_0_db": float(data["test_accuracy_snr_lt_0_db"]),
        "source": os.path.relpath(path, ROOT).replace(os.sep, "/"),
    }


def _baseline_from_markdown(path, dataset):
    with open(path, encoding="utf-8") as stream:
        lines = stream.readlines()
    display_name = "RadioML" if dataset == "radioml" else "Simulated"
    accuracy_row = None
    section = None
    per_snr = {}
    for line in lines:
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if section == "Test accuracy" and len(parts) >= 5 and parts[0] == display_name:
            try:
                accuracy_row = [float(value) for value in parts[1:4]]
            except ValueError:
                pass
        if section == "Accuracy by SNR" and len(parts) >= 3:
            try:
                snr_level = int(parts[0])
                value = float(parts[1] if dataset == "radioml" else parts[2])
            except ValueError:
                continue
            per_snr[snr_level] = value
    if accuracy_row is None or not all(level in per_snr for level in (0, 2, 4, 6, 8)):
        raise ValueError(f"Could not parse Stage A baseline for {dataset} from {path}")
    return {
        "snr_ge_10_db": accuracy_row[1],
        "snr_0_to_9_db": float(np.mean([per_snr[level] for level in (0, 2, 4, 6, 8)])),
        "snr_lt_0_db": accuracy_row[2],
        "source": os.path.relpath(path, ROOT).replace(os.sep, "/"),
    }


def load_stage_a_baseline(dataset):
    """Load Stage A range accuracies from its JSON summary or the report table."""
    json_path = os.path.join(ROOT, "results", f"{dataset}_stageA", "summary.json")
    if os.path.isfile(json_path):
        baseline = _baseline_from_json(json_path)
        if baseline is not None:
            return baseline
    markdown_path = os.path.join(ROOT, "results", "stageA_summary.md")
    if not os.path.isfile(markdown_path):
        raise FileNotFoundError(
            f"No Stage A baseline found at {json_path} or {markdown_path}"
        )
    return _baseline_from_markdown(markdown_path, dataset)


def _resolve_checkpoint_path(run_name, checkpoint_name):
    if os.path.isabs(checkpoint_name):
        return checkpoint_name
    run_dir = os.path.join(ROOT, "results", run_name)
    checkpoint_path = os.path.join(run_dir, checkpoint_name)
    if not os.path.isfile(checkpoint_path) and checkpoint_name == "best.pt":
        legacy_path = os.path.join(run_dir, "best_model.pt")
        if os.path.isfile(legacy_path):
            return legacy_path
    return checkpoint_path


def _evaluate_checkpoint(run_name, dataset, device, checkpoint_name):
    checkpoint_path = _resolve_checkpoint_path(run_name, checkpoint_name)
    model, checkpoint = load_model_checkpoint(checkpoint_path, map_location="cpu")
    model = model.to(device).eval()

    data = load_dataset("test", normal_only=True, dataset=dataset)
    known_modulation = data["mod"] >= 0
    X = torch.as_tensor(data["X"][known_modulation], dtype=torch.float32, device=device)
    mod_true = data["mod"][known_modulation].astype(np.int64, copy=False)
    snr_true = data["snr"][known_modulation].astype(np.float64, copy=False)
    channel = data["channel"][known_modulation]
    if not len(mod_true):
        raise ValueError("The test split contains no normal frames with known modulation")

    mod_predictions = []
    snr_predictions = [] if model.use_snr else None
    with torch.inference_mode():
        for start in range(0, len(X), 512):
            batch = X[start:start + 512]
            mod_logits, _, _, _ = unpack_model_output(model(batch))
            mod_predictions.append(mod_logits.argmax(dim=1).cpu().numpy())
            if model.use_snr:
                snr_predictions.append(
                    model.predict_snr_db(batch).squeeze(1).cpu().numpy()
                )
    mod_pred = np.concatenate(mod_predictions)
    snr_pred = (
        np.concatenate(snr_predictions).astype(np.float64, copy=False)
        if model.use_snr else None
    )

    metrics = _compute_metrics(mod_true, mod_pred, snr_true, snr_pred, MODULATIONS)
    baseline = load_stage_a_baseline(dataset)
    current_ranges = metrics["modulation"]["accuracy_by_snr_range"]
    metrics["stage_a_comparison"] = {
        "baseline_source": baseline["source"],
        "baseline_accuracy": {
            key: baseline[key] for key, _ in SNR_RANGES
        },
        "current_accuracy": {
            key: current_ranges[key]["accuracy"] for key, _ in SNR_RANGES
        },
        "delta_accuracy": {
            key: (
                current_ranges[key]["accuracy"] - baseline[key]
                if current_ranges[key]["accuracy"] is not None else None
            )
            for key, _ in SNR_RANGES
        },
    }
    if dataset == "simulated":
        metrics["by_channel"] = {}
        for channel_id, channel_name in enumerate(CHANNEL_NAMES):
            mask = channel == channel_id
            if np.any(mask):
                metrics["by_channel"][channel_name] = _compute_metrics(
                    mod_true[mask], mod_pred[mask], snr_true[mask],
                    snr_pred[mask] if snr_pred is not None else None,
                    MODULATIONS,
                )
    return metrics, mod_true, mod_pred, snr_true, snr_pred


def _save_plots(output_dir, metrics, snr_true, snr_pred):
    accuracy_by_snr = metrics["modulation"]["accuracy_by_snr"]
    levels = [int(level) for level in accuracy_by_snr if accuracy_by_snr[level]["count"]]
    accuracies = [accuracy_by_snr[str(level)]["accuracy"] for level in levels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(levels, accuracies, marker="o")
    ax.set(xlabel="True SNR (dB)", ylabel="Modulation accuracy",
           title="Modulation accuracy vs SNR", ylim=(0, 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_vs_snr.png"), dpi=300)
    plt.close(fig)

    if snr_pred is None:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(snr_true, snr_pred, s=8, alpha=0.2)
    limits = [
        min(float(snr_true.min()), float(snr_pred.min())),
        max(float(snr_true.max()), float(snr_pred.max())),
    ]
    ax.plot(limits, limits, linestyle="--", color="black", label="y = x")
    ax.set(xlabel="True SNR (dB)", ylabel="Estimated SNR (dB)",
           title="Estimated vs true SNR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "snr_scatter.png"), dpi=300)
    plt.close(fig)

    snr_metrics = metrics["snr_estimation"]["per_true_snr"]
    mae_levels = [
        int(level) for level, values in snr_metrics.items() if values["count"]
    ]
    maes = [snr_metrics[str(level)]["mae_db"] for level in mae_levels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mae_levels, maes, marker="o")
    ax.set(xlabel="True SNR (dB)", ylabel="Mean absolute error (dB)",
           title="SNR estimation MAE vs true SNR")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "snr_mae_vs_true_snr.png"), dpi=300)
    plt.close(fig)

    confusion = np.asarray(metrics["quality"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, label="Frames")
    ax.set(
        xlabel="Estimated quality", ylabel="True quality",
        title="Quality classification confusion matrix",
        xticks=range(3), yticks=range(3),
    )
    ax.set_xticklabels(QUALITY_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(QUALITY_NAMES)
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(confusion[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "quality_confusion_matrix.png"), dpi=300)
    plt.close(fig)


def _format_value(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _metrics_tables(title, metrics, include_comparison=None):
    lines = [f"### {title}", "", "#### Modulation accuracy", "",
             "| SNR range | Test accuracy | Frames |", "|---|---:|---:|"]
    lines.append(
        f"| Overall | {_format_value(metrics['modulation']['overall_accuracy'])} "
        f"| {metrics['test_frames']} |"
    )
    for key, label in SNR_RANGES:
        values = metrics["modulation"]["accuracy_by_snr_range"][key]
        lines.append(
            f"| {label} | {_format_value(values['accuracy'])} | {values['count']} |"
        )
    lines.extend(["", "| Modulation | Accuracy | Frames |", "|---|---:|---:|"])
    for name, values in metrics["modulation"]["per_modulation"].items():
        lines.append(
            f"| {name} | {_format_value(values['accuracy'])} | {values['count']} |"
        )
    if metrics["snr_estimation"] is None:
        lines.extend([
            "",
            "SNR estimates and quality metrics are unavailable for this checkpoint.",
        ])
    else:
        lines.extend([
            "",
            "#### SNR estimation by true SNR",
            "",
            "| True SNR (dB) | Frames | MAE (dB) | RMSE (dB) | Bias (dB) |",
            "|---:|---:|---:|---:|---:|",
        ])
        for level, values in metrics["snr_estimation"]["per_true_snr"].items():
            lines.append(
                f"| {level} | {values['count']} | {_format_value(values['mae_db'])} "
                f"| {_format_value(values['rmse_db'])} | {_format_value(values['bias_db'])} |"
            )
        overall = metrics["snr_estimation"]["overall"]
        quality = metrics["quality"]
        lines.extend([
            "",
            "#### Overall SNR and quality metrics",
            "",
            f"- SNR MAE: {_format_value(overall['mae_db'])} dB",
            f"- SNR RMSE: {_format_value(overall['rmse_db'])} dB",
            f"- SNR bias (estimated - true): {_format_value(overall['bias_db'])} dB",
            f"- Quality accuracy: {_format_value(quality['accuracy'])}",
            f"- Fraction of quality errors off by one class: "
            f"{_format_value(quality['fraction_of_errors_off_by_one'])}",
            "",
            "| True / estimated | Poor | Fair | Good |",
            "|---|---:|---:|---:|",
        ])
        for row, name in enumerate(QUALITY_NAMES):
            counts = quality["confusion_matrix"][row]
            lines.append(f"| {name} | {counts[0]} | {counts[1]} | {counts[2]} |")
    if include_comparison is not None:
        lines.extend([
            "",
            "#### Change vs Stage A",
            "",
            "| SNR range | Stage A accuracy | Current accuracy | Delta (percentage points) |",
            "|---|---:|---:|---:|",
        ])
        for key, label in SNR_RANGES:
            baseline = include_comparison["baseline_accuracy"][key]
            current = include_comparison["current_accuracy"][key]
            delta = include_comparison["delta_accuracy"][key]
            lines.append(
                f"| {label} | {_format_value(baseline)} | {_format_value(current)} "
                f"| {_format_value(None if delta is None else 100 * delta, 2)} |"
            )
    return lines


def _write_summary(path, run_name, dataset, checkpoint_path, metrics):
    lines = [
        f"# SNR evaluation: {run_name}",
        "",
        f"- Dataset: {dataset}",
        f"- Checkpoint: `{os.path.relpath(checkpoint_path, ROOT)}`",
        f"- Normal, known-modulation test frames: {metrics['test_frames']}",
        f"- Stage A baseline source: `{metrics['stage_a_comparison']['baseline_source']}`",
        "",
    ]
    lines.extend(_metrics_tables(
        "Overall test results", metrics, metrics["stage_a_comparison"]
    ))
    for channel_name, channel_metrics in metrics.get("by_channel", {}).items():
        lines.extend(["", *_metrics_tables(f"Channel: {channel_name}", channel_metrics)])
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def evaluate(run_name, dataset, device, checkpoint_name="best.pt"):
    metrics, mod_true, mod_pred, snr_true, snr_pred = _evaluate_checkpoint(
        run_name, dataset, device, checkpoint_name
    )
    output_dir = os.path.join(ROOT, "results", run_name, "eval")
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = _resolve_checkpoint_path(run_name, checkpoint_name)
    metrics["run_name"] = run_name
    metrics["dataset"] = dataset
    metrics["checkpoint"] = os.path.relpath(checkpoint_path, ROOT)
    with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    _write_summary(
        os.path.join(output_dir, "eval_summary.md"),
        run_name,
        dataset,
        checkpoint_path,
        metrics,
    )
    _save_plots(output_dir, metrics, snr_true, snr_pred)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", choices=("simulated", "radioml"), default="radioml")
    parser.add_argument("--device")
    parser.add_argument("--checkpoint", default="best.pt")
    args = parser.parse_args(argv)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    metrics = evaluate(args.run_name, args.dataset, device, args.checkpoint)
    print(
        f"Evaluated {metrics['test_frames']} test frames; "
        f"modulation accuracy={metrics['modulation']['overall_accuracy']:.4f}; "
        f"SNR MAE={metrics['snr_estimation']['overall']['mae_db']:.3f} dB"
    )


if __name__ == "__main__":
    main()
