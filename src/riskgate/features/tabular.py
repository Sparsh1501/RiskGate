"""Phase 1 static/tabular features (shared training path; velocity in Phase 2)."""

from __future__ import annotations

import pandas as pd

from riskgate.data.schema import CATEGORICAL_COLUMNS, NUMERIC_TRANSACTION_COLUMNS


def select_model_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return model matrix columns present in the frame."""
    wanted = list(NUMERIC_TRANSACTION_COLUMNS) + list(CATEGORICAL_COLUMNS)
    missing = [c for c in wanted if c not in frame.columns]
    if missing:
        msg = f"Expected columns missing from dataset: {missing}"
        raise KeyError(msg)
    return frame.loc[:, wanted].copy()
