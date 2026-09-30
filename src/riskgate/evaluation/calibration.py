from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationBin:
    bin_start: float
    bin_end: float
    count: int
    mean_predicted: float
    fraction_positive: float


def reliability_bins(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> list[CalibrationBin]:
    if n_bins < 2:
        msg = "n_bins must be >= 2"
        raise ValueError(msg)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                CalibrationBin(
                    bin_start=lo,
                    bin_end=hi,
                    count=0,
                    mean_predicted=float("nan"),
                    fraction_positive=float("nan"),
                )
            )
            continue
        mean_pred = float(y_score[mask].mean())
        frac_pos = float(y_true[mask].mean())
        bins.append(
            CalibrationBin(
                bin_start=lo,
                bin_end=hi,
                count=count,
                mean_predicted=mean_pred,
                fraction_positive=frac_pos,
            )
        )
    return bins


def calibration_summary(bins: list[CalibrationBin]) -> dict[str, float]:
    """Expected calibration error (ECE) over non-empty bins."""
    total = sum(b.count for b in bins)
    if total == 0:
        return {"ece": float("nan")}
    ece = 0.0
    for b in bins:
        if b.count == 0 or np.isnan(b.mean_predicted) or np.isnan(b.fraction_positive):
            continue
        ece += (b.count / total) * abs(b.mean_predicted - b.fraction_positive)
    return {"ece": float(ece)}


def bins_to_json(bins: list[CalibrationBin]) -> list[dict[str, float | int]]:
    return [asdict(b) for b in bins]
