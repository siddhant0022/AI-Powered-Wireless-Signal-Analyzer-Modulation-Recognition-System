import numpy as np
import pytest

from snr_metrics import (
    mean_absolute_error,
    mean_error_bias,
    modulation_accuracy_by_snr_range,
    per_snr_regression_metrics,
    quality_confusion_metrics,
    regression_metrics,
    root_mean_squared_error,
)


def test_regression_metrics_mae_rmse_and_bias():
    true = np.array([0.0, 2.0, -1.0])
    estimated = np.array([1.0, 0.0, 0.0])

    assert mean_absolute_error(true, estimated) == pytest.approx(4 / 3)
    assert root_mean_squared_error(true, estimated) == pytest.approx(np.sqrt(2))
    assert mean_error_bias(true, estimated) == pytest.approx(0.0)
    assert regression_metrics(true, estimated) == {
        "count": 3,
        "mae_db": pytest.approx(4 / 3),
        "rmse_db": pytest.approx(np.sqrt(2)),
        "bias_db": pytest.approx(0.0),
    }


def test_regression_metrics_by_true_snr_level():
    true_snr = np.array([-2.0, -2.0, 10.0])
    estimated_snr = np.array([-1.0, -4.0, 12.0])

    metrics = per_snr_regression_metrics(
        true_snr, true_snr, estimated_snr, levels=(-2, 0, 10)
    )

    assert metrics["-2"]["count"] == 2
    assert metrics["-2"]["mae_db"] == pytest.approx(1.5)
    assert metrics["0"] == {
        "count": 0, "mae_db": None, "rmse_db": None, "bias_db": None,
    }
    assert metrics["10"]["bias_db"] == pytest.approx(2.0)


def test_modulation_accuracy_uses_snr_range_boundaries():
    snr = np.array([-2, 0, 8, 9, 10, 18])
    truth = np.array([0, 1, 1, 0, 1, 0])
    prediction = np.array([0, 0, 1, 1, 1, 1])

    result = modulation_accuracy_by_snr_range(snr, truth, prediction)

    assert result["snr_lt_0_db"] == {"accuracy": 1.0, "count": 1}
    assert result["snr_0_to_9_db"] == {"accuracy": pytest.approx(1 / 3), "count": 3}
    assert result["snr_ge_10_db"] == {"accuracy": 0.5, "count": 2}


def test_quality_confusion_and_off_by_one_error_fraction():
    truth = np.array([0, 0, 1, 1, 2])
    estimated = np.array([0, 2, 0, 2, 1])

    result = quality_confusion_metrics(truth, estimated)

    assert result["accuracy"] == pytest.approx(0.2)
    assert result["confusion_matrix"] == [
        [1, 0, 1],
        [1, 0, 1],
        [0, 1, 0],
    ]
    assert result["error_count"] == 4
    assert result["errors_off_by_one"] == 3
    assert result["fraction_of_errors_off_by_one"] == pytest.approx(0.75)
