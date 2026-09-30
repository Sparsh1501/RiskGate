from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from riskgate.config import get_settings
from riskgate.data.load import load_merged_dataset
from riskgate.data.splits import save_split_metadata, time_based_split
from riskgate.evaluation.cost import ThresholdSelection, select_threshold_min_cost
from riskgate.models.logistic_regression import (
    fit_logistic_regression,
    predict_proba,
    save_model,
)
from riskgate.models.rules_baseline import RulesBaselineModel, train_labels


def save_threshold(selection: ThresholdSelection, path: Path, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, **asdict(selection)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    settings = get_settings()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    frame = load_merged_dataset()
    split = time_based_split(frame, settings)
    save_split_metadata(split.bounds, settings.artifacts_dir / "split_metadata.json")

    y_train = train_labels(split.train)

    rules = RulesBaselineModel.fit(split.train)
    rules.save(settings.artifacts_dir / "rules_baseline.json")

    lr = fit_logistic_regression(split.train, y_train)
    save_model(lr, settings.artifacts_dir / "logistic_regression.joblib")

    rules_val_scores = rules.predict_proba(split.val)
    rules_thr = select_threshold_min_cost(
        train_labels(split.val),
        rules_val_scores,
        settings,
    )
    save_threshold(
        rules_thr,
        settings.artifacts_dir / "rules_threshold.json",
        "rules",
    )

    lr_val_scores = predict_proba(lr, split.val)
    lr_thr = select_threshold_min_cost(train_labels(split.val), lr_val_scores, settings)
    save_threshold(
        lr_thr,
        settings.artifacts_dir / "logistic_regression_threshold.json",
        "logistic_regression",
    )

    print(f"Wrote artifacts to {settings.artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
