"""Pure metric functions for SNR and modulation evaluation."""
import numpy as np


def _paired_float_arrays(true_values, predicted_values):
    true_values = np.asarray(true_values, dtype=np.float64)
    predicted_values = np.asarray(predicted_values, dtype=np.float64)
    if true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("metric inputs must be one-dimensional")
    if true_values.shape != predicted_values.shape:
        raise ValueError("true and predicted arrays must have matching shapes")
    if not true_values.size:
        raise ValueError("metric inputs must not be empty")
    if not np.isfinite(true_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("metric inputs must contain only finite values")
    return true_values, predicted_values


def mean_absolute_error(true_values, predicted_values):
    true_values, predicted_values = _paired_float_arrays(true_values, predicted_values)
    return float(np.mean(np.abs(predicted_values - true_values)))


def root_mean_squared_error(true_values, predicted_values):
    true_values, predicted_values = _paired_float_arrays(true_values, predicted_values)
    return float(np.sqrt(np.mean(np.square(predicted_values - true_values))))


def mean_error_bias(true_values, predicted_values):
    """Return mean(predicted - true), so positive bias means overestimation."""
    true_values, predicted_values = _paired_float_arrays(true_values, predicted_values)
    return float(np.mean(predicted_values - true_values))


def regression_metrics(true_values, predicted_values):
    true_values, predicted_values = _paired_float_arrays(true_values, predicted_values)
    error = predicted_values - true_values
    return {
        "count": int(true_values.size),
        "mae_db": float(np.mean(np.abs(error))),
        "rmse_db": float(np.sqrt(np.mean(np.square(error)))),
        "bias_db": float(np.mean(error)),
    }


def modulation_accuracy_by_snr_range(snr_db, true_labels, predicted_labels):
    snr_db = np.asarray(snr_db)
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    if snr_db.ndim != 1 or true_labels.ndim != 1 or predicted_labels.ndim != 1:
        raise ValueError("SNR and label inputs must be one-dimensional")
    if snr_db.shape != true_labels.shape or true_labels.shape != predicted_labels.shape:
        raise ValueError("SNR and label inputs must have matching shapes")
    if not snr_db.size:
        raise ValueError("metric inputs must not be empty")

    ranges = {
        "snr_ge_10_db": snr_db >= 10,
        "snr_0_to_9_db": (snr_db >= 0) & (snr_db < 10),
        "snr_lt_0_db": snr_db < 0,
    }
    return {
        name: {
            "accuracy": (
                float(np.mean(true_labels[mask] == predicted_labels[mask]))
                if np.any(mask) else None
            ),
            "count": int(mask.sum()),
        }
        for name, mask in ranges.items()
    }


def per_snr_regression_metrics(snr_db, true_values, predicted_values, levels=None):
    snr_db = np.asarray(snr_db)
    true_values = np.asarray(true_values)
    predicted_values = np.asarray(predicted_values)
    if snr_db.ndim != 1 or true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("SNR and regression inputs must be one-dimensional")
    if snr_db.shape != true_values.shape or true_values.shape != predicted_values.shape:
        raise ValueError("SNR and regression inputs must have matching shapes")
    if not snr_db.size:
        raise ValueError("metric inputs must not be empty")
    if levels is None:
        levels = sorted(int(level) for level in np.unique(snr_db))

    metrics = {}
    for level in levels:
        mask = snr_db == level
        metrics[str(int(level))] = (
            regression_metrics(true_values[mask], predicted_values[mask])
            if np.any(mask) else {
                "count": 0, "mae_db": None, "rmse_db": None, "bias_db": None,
            }
        )
    return metrics


def quality_confusion_metrics(true_quality, estimated_quality):
    true_quality = np.asarray(true_quality)
    estimated_quality = np.asarray(estimated_quality)
    if true_quality.ndim != 1 or estimated_quality.ndim != 1:
        raise ValueError("quality inputs must be one-dimensional")
    if true_quality.shape != estimated_quality.shape:
        raise ValueError("quality inputs must have matching shapes")
    if not true_quality.size:
        raise ValueError("quality inputs must not be empty")
    if (
        not np.isin(true_quality, (0, 1, 2)).all()
        or not np.isin(estimated_quality, (0, 1, 2)).all()
    ):
        raise ValueError("quality labels must be 0, 1, or 2")

    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (true_quality, estimated_quality), 1)
    errors = true_quality != estimated_quality
    error_count = int(errors.sum())
    off_by_one = int((errors & (np.abs(true_quality - estimated_quality) == 1)).sum())
    return {
        "accuracy": float(np.mean(~errors)),
        "confusion_matrix": confusion.tolist(),
        "error_count": error_count,
        "errors_off_by_one": off_by_one,
        "fraction_of_errors_off_by_one": (
            float(off_by_one / error_count) if error_count else 0.0
        ),
    }
