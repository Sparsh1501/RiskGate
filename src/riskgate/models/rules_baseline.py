from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from riskgate.data.schema import TARGET_COLUMN


@dataclass(frozen=True)
class RulesThresholds:
    amount_p99: float
    amount_p995: float


@dataclass
class RulesBaselineModel:
    """Interpretable rules baseline; scores in [0, 1]."""

    thresholds: RulesThresholds
    rule_weights: dict[str, float]

    @classmethod
    def fit(cls, train: pd.DataFrame) -> RulesBaselineModel:
        amount_p99 = float(train["TransactionAmt"].quantile(0.99))
        amount_p995 = float(train["TransactionAmt"].quantile(0.995))
        thresholds = RulesThresholds(amount_p99=amount_p99, amount_p995=amount_p995)
        weights = {
            "amount_p995": 0.45,
            "amount_p99": 0.20,
            "email_mismatch": 0.20,
            "identity_missing": 0.15,
        }
        return cls(thresholds=thresholds, rule_weights=weights)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(frame), dtype=np.float64)

        amt = frame["TransactionAmt"].to_numpy(dtype=np.float64, copy=False)
        scores += self.rule_weights["amount_p995"] * (amt >= self.thresholds.amount_p995)
        scores += self.rule_weights["amount_p99"] * (
            (amt >= self.thresholds.amount_p99) & (amt < self.thresholds.amount_p995)
        )

        if "P_emaildomain" in frame.columns and "R_emaildomain" in frame.columns:
            p = frame["P_emaildomain"].astype("string")
            r = frame["R_emaildomain"].astype("string")
            mismatch = p.notna() & r.notna() & (p != r)
            scores += self.rule_weights["email_mismatch"] * mismatch.to_numpy(dtype=np.float64)

        identity_cols = [c for c in frame.columns if c.startswith("id_")]
        if identity_cols:
            missing_rate = frame[identity_cols].isna().mean(axis=1).to_numpy(dtype=np.float64)
            scores += self.rule_weights["identity_missing"] * (missing_rate >= 0.5)

        return np.clip(scores, 0.0, 1.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "thresholds": asdict(self.thresholds),
            "rule_weights": self.rule_weights,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RulesBaselineModel:
        payload = json.loads(path.read_text(encoding="utf-8"))
        thresholds = RulesThresholds(**payload["thresholds"])
        return cls(thresholds=thresholds, rule_weights=payload["rule_weights"])


def train_labels(frame: pd.DataFrame) -> np.ndarray:
    return frame[TARGET_COLUMN].to_numpy(dtype=np.int64, copy=False)
