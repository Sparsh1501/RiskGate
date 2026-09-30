from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from riskgate.data.schema import CATEGORICAL_COLUMNS, NUMERIC_TRANSACTION_COLUMNS
from riskgate.features.tabular import select_model_columns


def build_logistic_pipeline() -> Pipeline:
    numeric = NUMERIC_TRANSACTION_COLUMNS
    categorical = CATEGORICAL_COLUMNS

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric),
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(categorical),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def fit_logistic_regression(train: pd.DataFrame, y: np.ndarray) -> Pipeline:
    x = select_model_columns(train)
    pipeline = build_logistic_pipeline()
    pipeline.fit(x, y)
    return pipeline


def predict_proba(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    x = select_model_columns(frame)
    proba: np.ndarray = model.predict_proba(x)
    positive = proba[:, 1]
    return np.asarray(positive, dtype=np.float64)


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    loaded: Any = joblib.load(path)
    if not isinstance(loaded, Pipeline):
        msg = f"Expected sklearn Pipeline at {path}"
        raise TypeError(msg)
    return loaded
