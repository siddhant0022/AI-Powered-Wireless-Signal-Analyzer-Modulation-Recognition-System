import numpy as np
import pytest

from anomaly_metrics import (
    binary_metrics,
    pr_auc_numpy,
    roc_auc_numpy,
    select_threshold_for_recall,
    select_threshold_max_f1,
)


def test_auc_handles_perfect_random_and_inverted_scores():
    labels = np.array([0, 0, 1, 1])

    assert roc_auc_numpy(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc_numpy(labels, np.array([0.1, 0.8, 0.4, 0.7])) == pytest.approx(0.5)
    assert roc_auc_numpy(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert pr_auc_numpy(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_threshold_selection_and_confusion_metrics():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])

    best = select_threshold_max_f1(labels, scores)
    recall_threshold = select_threshold_for_recall(labels, scores, 0.95)
    applied = binary_metrics(labels, scores, best["threshold"])

    assert best["f1"] == pytest.approx(0.8)
    assert recall_threshold["recall"] == 1.0
    assert applied == best


def test_auc_matches_sklearn_when_available():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    labels = np.array([0, 1, 0, 1, 0, 1, 1])
    scores = np.array([0.2, 0.8, 0.7, 0.6, 0.1, 0.9, 0.4])
    precision, recall, _ = sklearn_metrics.precision_recall_curve(labels, scores)

    assert roc_auc_numpy(labels, scores) == pytest.approx(
        sklearn_metrics.roc_auc_score(labels, scores)
    )
    assert pr_auc_numpy(labels, scores) == pytest.approx(
        sklearn_metrics.auc(recall, precision)
    )
