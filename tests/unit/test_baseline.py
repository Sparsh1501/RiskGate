import numpy as np
import pandas as pd
import pytest

from riskgate.config import Settings
from riskgate.data.schema import TARGET_COLUMN, TIME_COLUMN
from riskgate.evaluation.cost import select_threshold_min_cost
from riskgate.evaluation.metrics import pr_auc, recall_at_fpr
from riskgate.models.rules_baseline import RulesBaselineModel


@pytest.fixture
def settings() -> Settings:
    return Settings(train_frac=0.6, val_frac=0.2, cost_fn=10.0, cost_fp=1.0, target_fpr=0.01)


def test_time_split_preserves_order() -> None:
    from riskgate.data.splits import time_based_split

    frame = pd.DataFrame(
        {
            TIME_COLUMN: [3, 1, 2, 4, 5, 6],
            TARGET_COLUMN: [0, 0, 0, 1, 0, 1],
            "TransactionAmt": [1.0] * 6,
        }
    )
    split = time_based_split(frame, Settings(train_frac=0.5, val_frac=0.25))
    assert split.train[TIME_COLUMN].tolist() == [1, 2, 3]
    assert split.val[TIME_COLUMN].tolist() == [4]
    assert split.test[TIME_COLUMN].tolist() == [5, 6]


def test_select_threshold_prefers_catching_fraud_when_fn_costly(settings: Settings) -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.35, 0.9], dtype=np.int64)
    sel = select_threshold_min_cost(y_true, y_score.astype(np.float64), settings)
    assert sel.threshold <= 0.35


def test_pr_auc_perfect_ranking() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    assert pr_auc(y_true, y_score) == pytest.approx(1.0)


def test_recall_at_fpr() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    y_score = np.array([0.9, 0.8, 0.4, 0.3, 0.85, 0.2], dtype=np.float64)
    recall = recall_at_fpr(y_true, y_score, target_fpr=0.25)
    assert recall == pytest.approx(0.5)


def test_rules_baseline_scores_bounded() -> None:
    train = pd.DataFrame(
        {
            "TransactionAmt": [10.0, 1000.0, 50.0],
            "P_emaildomain": ["a.com", "b.com", "c.com"],
            "R_emaildomain": ["a.com", "x.com", "c.com"],
            "id_01": [1.0, np.nan, 2.0],
        }
    )
    model = RulesBaselineModel.fit(train.assign(**{TARGET_COLUMN: [0, 1, 0]}))
    scores = model.predict_proba(train)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
