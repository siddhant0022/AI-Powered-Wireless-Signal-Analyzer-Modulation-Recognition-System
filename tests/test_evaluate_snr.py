import os

import pytest

from evaluate_snr import ROOT, _baseline_from_markdown


@pytest.mark.parametrize(
    ("dataset", "expected_high", "expected_mid", "expected_low"),
    [
        ("radioml", 0.9287, 0.91442, 0.3239),
        ("simulated", 0.8424, 0.81152, 0.3300),
    ],
)
def test_stage_a_markdown_baseline_parser(dataset, expected_high, expected_mid, expected_low):
    path = os.path.join(ROOT, "results", "stageA_summary.md")

    baseline = _baseline_from_markdown(path, dataset)

    assert baseline["snr_ge_10_db"] == pytest.approx(expected_high)
    assert baseline["snr_0_to_9_db"] == pytest.approx(expected_mid)
    assert baseline["snr_lt_0_db"] == pytest.approx(expected_low)
