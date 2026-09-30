from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from riskgate.config import Settings, get_settings


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    expected_cost: float
    tp: int
    fp: int
    tn: int
    fn: int


def expected_cost(tp: int, fp: int, tn: int, fn: int, cost_fp: float, cost_fn: float) -> float:
    _ = tn  # included for clarity in cost accounting
    return cost_fp * fp + cost_fn * fn


def select_threshold_min_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    settings: Settings | None = None,
) -> ThresholdSelection:
    cfg = settings or get_settings()
    if len(y_true) != len(y_score):
        msg = "y_true and y_score length mismatch"
        raise ValueError(msg)

    candidates = np.unique(y_score)
    if len(candidates) > 500:
        candidates = np.quantile(y_score, np.linspace(0.0, 1.0, 500))

    best: ThresholdSelection | None = None
    for thr in candidates:
        y_pred = (y_score >= thr).astype(np.int64)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        cost = expected_cost(tp, fp, tn, fn, cfg.cost_fp, cfg.cost_fn)
        selection = ThresholdSelection(
            threshold=float(thr),
            expected_cost=cost,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )
        if best is None or selection.expected_cost < best.expected_cost:
            best = selection

    assert best is not None
    return best


def selection_to_dict(selection: ThresholdSelection) -> dict[str, float | int]:
    return asdict(selection)
