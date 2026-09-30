from __future__ import annotations

import pytest

from riskgate.evaluation.calibration import calibration_summary, reliability_bins


def test_reliability_bins_cover_unit_interval() -> None:
    import numpy as np

    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_score = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    bins = reliability_bins(y_true, y_score, n_bins=2)
    assert len(bins) == 2
    summary = calibration_summary(bins)
    assert "ece" in summary
    assert summary["ece"] == pytest.approx(0.25, abs=0.05)


def test_calibration_bin_serialization() -> None:
    import numpy as np

    from riskgate.evaluation.calibration import bins_to_json

    bins = reliability_bins(
        np.array([0, 1], dtype=np.int64),
        np.array([0.2, 0.8], dtype=np.float64),
        n_bins=2,
    )
    data = bins_to_json(bins)
    assert isinstance(data, list)
    assert "bin_start" in data[0]
