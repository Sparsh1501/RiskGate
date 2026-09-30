from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Maximum recall achievable at or below the target false-positive rate."""
    y_true = y_true.astype(np.int64, copy=False)
    negatives = int((y_true == 0).sum())
    positives = int((y_true == 1).sum())
    if negatives == 0 or positives == 0:
        return float("nan")

    order = np.argsort(-y_score)
    sorted_labels = y_true[order]

    tp = 0
    fp = 0
    best_recall = 0.0
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / negatives
        if fpr <= target_fpr + 1e-12:
            best_recall = max(best_recall, tp / positives)
    return float(best_recall)
