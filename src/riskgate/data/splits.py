from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from riskgate.config import Settings, get_settings
from riskgate.data.schema import TIME_COLUMN


@dataclass(frozen=True)
class SplitBounds:
    train_end: int
    val_end: int
    train_rows: int
    val_rows: int
    test_rows: int
    train_time_max: int
    val_time_max: int
    test_time_max: int


@dataclass(frozen=True)
class TimeSplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    bounds: SplitBounds


def time_based_split(
    frame: pd.DataFrame,
    settings: Settings | None = None,
) -> TimeSplitResult:
    """Split by ascending TransactionDT; no random shuffling."""
    cfg = settings or get_settings()
    if TIME_COLUMN not in frame.columns:
        msg = f"Missing time column {TIME_COLUMN}"
        raise KeyError(msg)

    ordered = frame.sort_values(TIME_COLUMN).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * cfg.train_frac)
    val_end = train_end + int(n * cfg.val_frac)

    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()

    train_time_max = -1 if len(train) == 0 else int(train[TIME_COLUMN].max())
    val_time_max = -1 if len(val) == 0 else int(val[TIME_COLUMN].max())
    test_time_max = -1 if len(test) == 0 else int(test[TIME_COLUMN].max())

    bounds = SplitBounds(
        train_end=train_end,
        val_end=val_end,
        train_rows=len(train),
        val_rows=len(val),
        test_rows=len(test),
        train_time_max=train_time_max,
        val_time_max=val_time_max,
        test_time_max=test_time_max,
    )
    return TimeSplitResult(train=train, val=val, test=test, bounds=bounds)


def save_split_metadata(bounds: SplitBounds, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        **asdict(bounds),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
