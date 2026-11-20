"""Pure NumPy binary-classification metrics for anomaly evaluation."""
import numpy as np


def _validate_binary_inputs(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be matching one-dimensional arrays")
    if not labels.size:
        raise ValueError("labels and scores must not be empty")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    return labels.astype(np.int8, copy=False), scores


def binary_metrics(labels, scores, threshold):
    """Return precision, recall, F1, FPR, and counts at a score threshold."""
    labels, scores = _validate_binary_inputs(labels, scores)
    predicted = scores >= threshold
    positive = labels == 1
    negative = ~positive
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & negative))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & negative))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _ranked_counts(labels, scores):
    labels, scores = _validate_binary_inputs(labels, scores)
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    if not positives or not negatives:
        raise ValueError("ROC/PR curves require both positive and negative samples")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    true_positives = np.cumsum(sorted_labels == 1)[group_ends].astype(np.float64)
    false_positives = (group_ends + 1).astype(np.float64) - true_positives
    thresholds = sorted_scores[group_ends]
    return positives, negatives, true_positives, false_positives, thresholds


def roc_curve_numpy(labels, scores):
    """Return FPR, TPR, and descending thresholds with a leading origin."""
    positives, negatives, tp, fp, thresholds = _ranked_counts(labels, scores)
    fpr = np.r_[0.0, fp / negatives]
    tpr = np.r_[0.0, tp / positives]
    thresholds = np.r_[np.inf, thresholds]
    return fpr, tpr, thresholds


def roc_auc_numpy(labels, scores):
    fpr, tpr, _ = roc_curve_numpy(labels, scores)
    return float(np.trapezoid(tpr, fpr))


def precision_recall_curve_numpy(labels, scores):
    """Return precision and recall in increasing-recall order."""
    positives, _, tp, fp, thresholds = _ranked_counts(labels, scores)
    precision = tp / (tp + fp)
    recall = tp / positives
    return (
        np.r_[1.0, precision],
        np.r_[0.0, recall],
        thresholds,
    )


def pr_auc_numpy(labels, scores):
    precision, recall, _ = precision_recall_curve_numpy(labels, scores)
    return float(np.trapezoid(precision, recall))


def select_threshold_max_f1(labels, scores):
    labels, scores = _validate_binary_inputs(labels, scores)
    if not np.any(labels == 1) or not np.any(labels == 0):
        raise ValueError("threshold selection requires positive and negative samples")
    candidates = np.r_[
        np.nextafter(np.max(scores), np.inf),
        np.unique(scores)[::-1],
    ]
    results = [binary_metrics(labels, scores, threshold) for threshold in candidates]
    best_index = max(range(len(results)), key=lambda index: results[index]["f1"])
    return results[best_index]


def select_threshold_for_recall(labels, scores, target_recall=0.95):
    labels, scores = _validate_binary_inputs(labels, scores)
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    if not np.any(labels == 1) or not np.any(labels == 0):
        raise ValueError("threshold selection requires positive and negative samples")
    candidates = np.unique(scores)[::-1]
    for threshold in candidates:
        result = binary_metrics(labels, scores, threshold)
        if result["recall"] >= target_recall:
            return result
    raise ValueError("could not select a threshold meeting target recall")
