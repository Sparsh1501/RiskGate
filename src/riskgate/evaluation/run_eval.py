from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from riskgate.config import get_settings
from riskgate.data.load import load_merged_dataset
from riskgate.data.splits import time_based_split
from riskgate.evaluation.calibration import bins_to_json, calibration_summary, reliability_bins
from riskgate.evaluation.cost import ThresholdSelection
from riskgate.evaluation.metrics import pr_auc, recall_at_fpr
from riskgate.models.logistic_regression import load_model, predict_proba
from riskgate.models.rules_baseline import RulesBaselineModel, train_labels


def _load_threshold(path: Path) -> ThresholdSelection:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ThresholdSelection(
        threshold=float(payload["threshold"]),
        expected_cost=float(payload["expected_cost"]),
        tp=int(payload["tp"]),
        fp=int(payload["fp"]),
        tn=int(payload["tn"]),
        fn=int(payload["fn"]),
    )


def _evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: ThresholdSelection,
    target_fpr: float,
) -> dict[str, Any]:
    y_pred = (y_score >= threshold.threshold).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    bins = reliability_bins(y_true, y_score)
    cal = calibration_summary(bins)
    return {
        "model": name,
        "threshold": threshold.threshold,
        "threshold_selected_on": "validation",
        "validation_expected_cost": threshold.expected_cost,
        "test": {
            "pr_auc": pr_auc(y_true, y_score),
            f"recall_at_{target_fpr:.0%}_fpr": recall_at_fpr(y_true, y_score, target_fpr),
            "confusion_at_selected_threshold": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "calibration": {
                "ece": cal["ece"],
                "reliability_bins": bins_to_json(bins),
            },
        },
    }


def main() -> None:
    settings = get_settings()
    artifacts = settings.artifacts_dir
    for required in (
        artifacts / "rules_baseline.json",
        artifacts / "logistic_regression.joblib",
        artifacts / "rules_threshold.json",
        artifacts / "logistic_regression_threshold.json",
        artifacts / "split_metadata.json",
    ):
        if not required.is_file():
            msg = f"Missing {required}. Run `make train-baseline` first."
            raise FileNotFoundError(msg)

    split_meta = json.loads((artifacts / "split_metadata.json").read_text(encoding="utf-8"))
    frame = load_merged_dataset()
    split = time_based_split(frame, settings)
    y_test = train_labels(split.test)

    rules = RulesBaselineModel.load(artifacts / "rules_baseline.json")
    rules_thr = _load_threshold(artifacts / "rules_threshold.json")
    rules_scores = rules.predict_proba(split.test)
    rules_report = _evaluate_model(
        "rules",
        y_test,
        rules_scores,
        rules_thr,
        settings.target_fpr,
    )

    lr = load_model(artifacts / "logistic_regression.joblib")
    lr_thr = _load_threshold(artifacts / "logistic_regression_threshold.json")
    lr_scores = predict_proba(lr, split.test)
    lr_report = _evaluate_model(
        "logistic_regression",
        y_test,
        lr_scores,
        lr_thr,
        settings.target_fpr,
    )

    # Honest error analysis: high-value fraud missed at selected LR threshold
    lr_pred = (lr_scores >= lr_thr.threshold).astype(np.int64)
    missed_fraud = (y_test == 1) & (lr_pred == 0)
    missed_amt = split.test.loc[missed_fraud, "TransactionAmt"]
    error_analysis: dict[str, Any] = {
        "logistic_regression_missed_fraud_count": int(missed_fraud.sum()),
        "logistic_regression_missed_fraud_amount_median": float(missed_amt.median())
        if len(missed_amt)
        else float("nan"),
        "logistic_regression_missed_fraud_amount_p95": float(missed_amt.quantile(0.95))
        if len(missed_amt)
        else float("nan"),
    }

    payload: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "dataset": "ieee-fraud-detection",
        "split_metadata": split_meta,
        "cost_assumptions": {
            "cost_fp": settings.cost_fp,
            "cost_fn": settings.cost_fn,
            "note": "Threshold chosen on validation by minimizing cost_fp*FP + cost_fn*FN.",
        },
        "target_fpr_for_recall_metric": settings.target_fpr,
        "models": {
            "rules": rules_report,
            "logistic_regression": lr_report,
        },
        "error_analysis": error_analysis,
        "limitations": [
            "Public IEEE-CIS labels; no production traffic or issuer feedback loops.",
            "Phase 1 uses static tabular columns only; velocity features added in Phase 2.",
        ],
    }

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.results_dir / "baseline.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
