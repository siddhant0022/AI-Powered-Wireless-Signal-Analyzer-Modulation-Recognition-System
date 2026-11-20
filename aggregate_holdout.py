"""Aggregate per-type detection results across anomaly holdout runs."""
import argparse
import glob
import json
import os


ROOT = os.path.dirname(os.path.abspath(__file__))


def aggregate(results_dir):
    pattern = os.path.join(results_dir, "holdout_*", "eval_anomaly",
                           "eval_anomaly_metrics.json")
    metric_paths = sorted(glob.glob(pattern))
    if not metric_paths:
        raise FileNotFoundError(f"No holdout evaluation metrics matched {pattern}")

    rows = []
    for path in metric_paths:
        with open(path, encoding="utf-8") as stream:
            metrics = json.load(stream)
        heldout = metrics.get("holdout_anomaly_type")
        for anomaly_type, values in metrics["per_anomaly_type"].items():
            rows.append({
                "run_name": metrics.get("run_name", os.path.basename(
                    os.path.dirname(os.path.dirname(path))
                )),
                "heldout_type": heldout or "none",
                "anomaly_type": anomaly_type,
                "exposure": "unseen" if anomaly_type == heldout else "seen",
                "detection_rate": values["detection_rate"],
                "roc_auc": values["roc_auc"],
            })
    return rows


def write_markdown(rows, output_path):
    lines = [
        "# Anomaly holdout aggregate",
        "",
        "| Run | Held-out type | Evaluated type | Seen/unseen | Detection rate | ROC-AUC vs normal |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        detection = (
            "n/a" if row["detection_rate"] is None
            else f"{row['detection_rate']:.4f}"
        )
        auc = "n/a" if row["roc_auc"] is None else f"{row['roc_auc']:.4f}"
        lines.append(
            f"| {row['run_name']} | {row['heldout_type']} | {row['anomaly_type']} "
            f"| {row['exposure']} | {detection} | {auc} |"
        )
    with open(output_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "results", "holdout_aggregate.md"),
    )
    args = parser.parse_args(argv)
    rows = aggregate(args.results_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_markdown(rows, args.output)
    print(f"Wrote {len(rows)} per-type rows to {args.output}")


if __name__ == "__main__":
    main()
