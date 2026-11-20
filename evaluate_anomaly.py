"""Evaluate anomaly detection on the held-out simulated test split."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from anomaly_metrics import (
    binary_metrics,
    pr_auc_numpy,
    precision_recall_curve_numpy,
    roc_auc_numpy,
    roc_curve_numpy,
    select_threshold_for_recall,
    select_threshold_max_f1,
)
from dataset_loader import ANOMALY_TYPES, load_dataset
from model import load_model_checkpoint


ROOT = os.path.dirname(os.path.abspath(__file__))
TYPE_NAMES = ANOMALY_TYPES[1:]
SNR_RANGES = (
    ("snr_lt_0_db", "SNR < 0 dB", lambda snr: snr < 0),
    ("snr_0_to_9_db", "SNR 0-9 dB", lambda snr: (snr >= 0) & (snr < 10)),
    ("snr_ge_10_db", "SNR >= 10 dB", lambda snr: snr >= 10),
)


def _read_run_config(path, checkpoint):
    config_path = os.path.join(path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as stream:
            config = json.load(stream)
    else:
        config = {}
    training_config = checkpoint.get("training_config", {})
    holdout = config.get(
        "holdout_anomaly_type",
        training_config.get("holdout_anomaly_type"),
    )
    data_path = config.get("data_path", training_config.get("data_path"))
    if data_path is None:
        data_path = os.path.join(ROOT, "data", "signal_dataset.npz")
    return config, holdout, data_path


def _predict_anomaly_scores(model, data, device):
    X = torch.as_tensor(data["X"], dtype=torch.float32, device=device)
    logits = []
    with torch.inference_mode():
        for start in range(0, len(X), 512):
            output = model(X[start:start + 512])
            if not isinstance(output, dict) or "anomaly_logit" not in output:
                raise ValueError("Anomaly checkpoint must return anomaly_logit")
            logits.append(output["anomaly_logit"].squeeze(1))
    if not logits:
        raise ValueError("Evaluation split contains no frames")
    return torch.sigmoid(torch.cat(logits)).cpu().numpy().astype(np.float64, copy=False)


def _threshold_metrics_on_validation(labels, scores):
    best_f1 = select_threshold_max_f1(labels, scores)
    recall_95 = select_threshold_for_recall(labels, scores, target_recall=0.95)
    return {"max_f1": best_f1, "recall_95": recall_95}


def _compute_test_metrics(data, scores, threshold, type_head):
    labels = data["anomaly"].astype(np.int8, copy=False)
    snr = data["snr"].astype(np.float64, copy=False)
    anomaly_type = data["anomaly_type"].astype(np.int64, copy=False)
    overall = binary_metrics(labels, scores, threshold)
    overall["roc_auc"] = roc_auc_numpy(labels, scores)
    overall["pr_auc"] = pr_auc_numpy(labels, scores)
    by_type = {}
    normal_mask = labels == 0
    for type_id, type_name in enumerate(TYPE_NAMES, start=1):
        type_mask = anomaly_type == type_id
        detection_rate = (
            float(np.mean(scores[type_mask] >= threshold))
            if np.any(type_mask) else None
        )
        comparison_mask = normal_mask | type_mask
        type_labels = type_mask[comparison_mask].astype(np.int8)
        by_type[type_name] = {
            "anomaly_count": int(type_mask.sum()),
            "detection_rate": detection_rate,
            "roc_auc": (
                roc_auc_numpy(type_labels, scores[comparison_mask])
                if np.any(type_mask) and np.any(normal_mask) else None
            ),
        }

    snr_by_range = {}
    for key, _, predicate in SNR_RANGES:
        range_mask = predicate(snr)
        normal_in_range = range_mask & normal_mask
        snr_by_range[key] = {
            "false_positive_rate_normal": (
                float(np.mean(scores[normal_in_range] >= threshold))
                if np.any(normal_in_range) else None
            ),
            "normal_count": int(normal_in_range.sum()),
            "detection_rate_by_type": {},
        }
        for type_id, type_name in enumerate(TYPE_NAMES, start=1):
            type_in_range = range_mask & (anomaly_type == type_id)
            snr_by_range[key]["detection_rate_by_type"][type_name] = {
                "detection_rate": (
                    float(np.mean(scores[type_in_range] >= threshold))
                    if np.any(type_in_range) else None
                ),
                "count": int(type_in_range.sum()),
            }

    type_metrics = None
    if type_head:
        anomaly_mask = labels == 1
        confusion = np.zeros((5, 5), dtype=np.int64)
        target_type = anomaly_type[anomaly_mask] - 1
        predicted_type = data["type_prediction"][anomaly_mask]
        np.add.at(confusion, (target_type, predicted_type), 1)
        type_metrics = {
            "confusion_matrix": confusion.tolist(),
            "accuracy_by_type": {
                name: (
                    float(np.mean(predicted_type[target_type == class_id] == class_id))
                    if np.any(target_type == class_id) else None
                )
                for class_id, name in enumerate(TYPE_NAMES)
            },
            "anomaly_frames": int(anomaly_mask.sum()),
        }
    return {
        "test_frames": int(len(labels)),
        "normal_frames": int(normal_mask.sum()),
        "anomaly_frames": int((labels == 1).sum()),
        "overall": overall,
        "per_anomaly_type": by_type,
        "by_snr_range": snr_by_range,
        "type_classification": type_metrics,
    }


def _infer_type_predictions(model, data, device):
    if not model.use_anomaly_type:
        return None
    X = torch.as_tensor(data["X"], dtype=torch.float32, device=device)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(X), 512):
            output = model(X[start:start + 512])
            if "type_logits" not in output:
                raise ValueError("Checkpoint config enables type head but type_logits are missing")
            predictions.append(output["type_logits"].argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def _load_scores(model, dataset_name, data_path, split, device, holdout=None):
    data = load_dataset(
        split, path=data_path, dataset=dataset_name, normal_only=False
    )
    if holdout is not None and split == "val":
        heldout_id = ANOMALY_TYPES.index(holdout)
        keep = ~((data["anomaly"] == 1) & (data["anomaly_type"] == heldout_id))
        data = {key: value[keep] for key, value in data.items()}
    if not len(data["X"]):
        raise ValueError(f"{split} split has no frames after applying holdout filters")
    scores = _predict_anomaly_scores(model, data, device)
    if model.use_anomaly_type:
        data["type_prediction"] = _infer_type_predictions(model, data, device)
    return data, scores


def _save_plots(output_dir, test_data, scores, metrics):
    labels = test_data["anomaly"]
    fpr, tpr, _ = roc_curve_numpy(labels, scores)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"Overall (AUC={metrics['overall']['roc_auc']:.3f})")
    normal_mask = labels == 0
    anomaly_type = test_data["anomaly_type"]
    for type_id, type_name in enumerate(TYPE_NAMES, start=1):
        mask = normal_mask | (anomaly_type == type_id)
        type_labels = (anomaly_type[mask] == type_id).astype(np.int8)
        if np.any(type_labels == 1) and np.any(type_labels == 0):
            type_fpr, type_tpr, _ = roc_curve_numpy(type_labels, scores[mask])
            auc = metrics["per_anomaly_type"][type_name]["roc_auc"]
            ax.plot(type_fpr, type_tpr, label=f"{type_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate",
           title="Anomaly ROC curves", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve_numpy(labels, scores)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"PR-AUC={metrics['overall']['pr_auc']:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Anomaly precision-recall curve",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=300)
    plt.close(fig)

    range_keys = [key for key, _, _ in SNR_RANGES]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(range_keys))
    width = 0.15
    for class_index, type_name in enumerate(TYPE_NAMES):
        values = [
            metrics["by_snr_range"][key]["detection_rate_by_type"][type_name][
                "detection_rate"
            ]
            for key in range_keys
        ]
        ax.bar(
            x + (class_index - 2) * width,
            [0.0 if value is None else value for value in values],
            width,
            label=type_name,
        )
    ax.set(
        xlabel="True pre-fault SNR range",
        ylabel="Detection rate (recall)",
        title="Anomaly detection rate by SNR range",
        xticks=x,
        xticklabels=[label for _, label, _ in SNR_RANGES],
        ylim=(0, 1),
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "detection_by_snr_range.png"), dpi=300)
    plt.close(fig)

    type_metrics = metrics["type_classification"]
    if type_metrics is not None:
        confusion = np.asarray(type_metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
        fig.colorbar(image, ax=ax, label="Anomaly frames")
        ax.set(
            xlabel="Predicted anomaly type", ylabel="True anomaly type",
            title="Anomaly type confusion matrix",
            xticks=range(5), yticks=range(5),
        )
        ax.set_xticklabels(TYPE_NAMES, rotation=35, ha="right")
        ax.set_yticklabels(TYPE_NAMES)
        for row in range(5):
            for column in range(5):
                ax.text(column, row, str(confusion[row, column]),
                        ha="center", va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "type_confusion_matrix.png"), dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[labels == 0], bins=30, alpha=0.65, density=True, label="Normal")
    ax.hist(scores[labels == 1], bins=30, alpha=0.65, density=True, label="Anomaly")
    ax.set(xlabel="Anomaly score (sigmoid probability)", ylabel="Density",
           title="Anomaly score distributions")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_histograms.png"), dpi=300)
    plt.close(fig)


def _fmt(value, precision=4):
    return "n/a" if value is None else f"{value:.{precision}f}"


def _write_markdown(path, run_name, holdout, thresholds, metrics):
    lines = [
        f"# Anomaly evaluation: {run_name}",
        "",
        f"- Test frames: {metrics['test_frames']} "
        f"({metrics['normal_frames']} normal, {metrics['anomaly_frames']} anomaly)",
        f"- Held-out anomaly type: {holdout or 'none'}",
        "- Tables use the validation-selected maximum-F1 threshold "
        f"`{thresholds['max_f1']['threshold']:.8g}`.",
        f"- Validation 95%-recall threshold: `{thresholds['recall_95']['threshold']:.8g}` "
        "(saved but not used in the tables).",
        "",
        "## Overall test performance",
        "",
        "| Threshold selection | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Normal FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in (
        ("Validation max F1 (used)", metrics["overall"]),
        ("Validation 95% recall", metrics["overall_at_95_recall"]),
    ):
        lines.append(
            f"| {name} | {result['threshold']:.8g} | {_fmt(result['precision'])} "
            f"| {_fmt(result['recall'])} | {_fmt(result['f1'])} "
            f"| {_fmt(result['roc_auc'])} | {_fmt(result['pr_auc'])} "
            f"| {_fmt(result['false_positive_rate'])} |"
        )
    lines.extend([
        "",
        "## Per-type detection (maximum-F1 threshold)",
        "",
        "| Type | Exposure | Detection rate | ROC-AUC vs normal | Anomaly frames |",
        "|---|---|---:|---:|---:|",
    ])
    for type_name, result in metrics["per_anomaly_type"].items():
        exposure = "held-out (unseen)" if type_name == holdout else "seen"
        lines.append(
            f"| {type_name} | {exposure} | {_fmt(result['detection_rate'])} "
            f"| {_fmt(result['roc_auc'])} | {result['anomaly_count']} |"
        )
    lines.extend([
        "",
        "## Detection rate and normal false-positive rate by pre-fault SNR",
        "",
        "| SNR range | Type | Detection rate | Anomaly frames | Normal FPR | Normal frames |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for range_key, range_label, _ in SNR_RANGES:
        range_result = metrics["by_snr_range"][range_key]
        for type_name in TYPE_NAMES:
            result = range_result["detection_rate_by_type"][type_name]
            lines.append(
                f"| {range_label} | {type_name} | {_fmt(result['detection_rate'])} "
                f"| {result['count']} | {_fmt(range_result['false_positive_rate_normal'])} "
                f"| {range_result['normal_count']} |"
            )
    if metrics["type_classification"] is not None:
        lines.extend([
            "",
            "## Anomaly type classification (anomaly frames only)",
            "",
            "| True / predicted | " + " | ".join(TYPE_NAMES) + " |",
            "|---|" + "---:|" * 5,
        ])
        for row, name in enumerate(TYPE_NAMES):
            values = metrics["type_classification"]["confusion_matrix"][row]
            lines.append(f"| {name} | " + " | ".join(str(value) for value in values) + " |")
        lines.extend(["", "| Type | Accuracy |", "|---|---:|"])
        for name, accuracy in metrics["type_classification"]["accuracy_by_type"].items():
            lines.append(f"| {name} | {_fmt(accuracy)} |")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def evaluate(run_name, device, checkpoint_name="best.pt"):
    run_dir = os.path.join(ROOT, "results", run_name)
    checkpoint_path = (
        checkpoint_name
        if os.path.isabs(checkpoint_name)
        else os.path.join(run_dir, checkpoint_name)
    )
    model, checkpoint = load_model_checkpoint(checkpoint_path, map_location="cpu")
    if not model.use_anomaly:
        raise ValueError(f"{checkpoint_path} was not trained with --use-anomaly")
    model = model.to(device).eval()
    config, holdout, data_path = _read_run_config(run_dir, checkpoint)
    if holdout is not None and holdout not in TYPE_NAMES:
        raise ValueError(f"Unknown held-out anomaly type in run config: {holdout}")

    validation_data, validation_scores = _load_scores(
        model, "simulated", data_path, "val", device, holdout=holdout
    )
    thresholds = _threshold_metrics_on_validation(
        validation_data["anomaly"], validation_scores
    )
    test_data, test_scores = _load_scores(
        model, "simulated", data_path, "test", device
    )
    threshold = thresholds["max_f1"]["threshold"]
    metrics = _compute_test_metrics(
        test_data, test_scores, threshold, model.use_anomaly_type
    )
    metrics["overall"]["roc_auc"] = roc_auc_numpy(
        test_data["anomaly"], test_scores
    )
    metrics["overall"]["pr_auc"] = pr_auc_numpy(
        test_data["anomaly"], test_scores
    )
    metrics["overall_at_95_recall"] = binary_metrics(
        test_data["anomaly"], test_scores,
        thresholds["recall_95"]["threshold"],
    )
    metrics["overall_at_95_recall"]["roc_auc"] = metrics["overall"]["roc_auc"]
    metrics["overall_at_95_recall"]["pr_auc"] = metrics["overall"]["pr_auc"]
    metrics["per_anomaly_type"] = {
        name: {
            **values,
            "exposure": "held-out (unseen)" if name == holdout else "seen",
        }
        for name, values in metrics["per_anomaly_type"].items()
    }
    metrics.update({
        "run_name": run_name,
        "checkpoint": os.path.relpath(checkpoint_path, ROOT).replace(os.sep, "/"),
        "dataset": "simulated",
        "holdout_anomaly_type": holdout,
        "validation_frames_for_thresholds": int(len(validation_scores)),
        "validation_thresholds": thresholds,
        "table_threshold": "max_f1",
        "type_head_enabled": bool(model.use_anomaly_type),
    })
    output_dir = os.path.join(run_dir, "eval_anomaly")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_anomaly_metrics.json"),
              "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    _write_markdown(
        os.path.join(output_dir, "eval_anomaly.md"),
        run_name, holdout, thresholds, metrics,
    )
    _save_plots(output_dir, test_data, test_scores, metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint", default="best.pt")
    args = parser.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    metrics = evaluate(args.run_name, device, args.checkpoint)
    print(
        f"Evaluated {metrics['test_frames']} test frames; "
        f"max-F1 threshold={metrics['validation_thresholds']['max_f1']['threshold']:.6g}; "
        f"test F1={metrics['overall']['f1']:.4f}; "
        f"ROC-AUC={metrics['overall']['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
