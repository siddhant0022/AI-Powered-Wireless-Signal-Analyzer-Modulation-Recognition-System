import json

from aggregate_holdout import aggregate, write_markdown


def test_aggregate_holdout_writes_seen_and_unseen_rows(tmp_path):
    metrics_path = (
        tmp_path / "holdout_unknown_ofdm" / "eval_anomaly"
        / "eval_anomaly_metrics.json"
    )
    metrics_path.parent.mkdir(parents=True)
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump({
            "run_name": "holdout_unknown_ofdm",
            "holdout_anomaly_type": "unknown_ofdm",
            "per_anomaly_type": {
                "unknown_ofdm": {"detection_rate": 0.25, "roc_auc": 0.6},
                "clipping": {"detection_rate": 0.75, "roc_auc": 0.9},
            },
        }, stream)

    rows = aggregate(str(tmp_path))
    output = tmp_path / "aggregate.md"
    write_markdown(rows, output)

    assert len(rows) == 2
    assert {row["exposure"] for row in rows} == {"seen", "unseen"}
    assert "Held-out type" in output.read_text(encoding="utf-8")
    assert "0.2500" in output.read_text(encoding="utf-8")
