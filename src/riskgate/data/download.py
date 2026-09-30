from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
import pandas as pd
from kagglehub.exceptions import UnauthenticatedError

from riskgate.config import get_settings
from riskgate.data.schema import (
    IDENTITY_FILE,
    JOIN_KEY,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRANSACTION_FILE,
)


def download_competition(data_dir: Path | None = None) -> Path:
    """Download IEEE-CIS via kagglehub and copy CSVs into the project data directory."""
    settings = get_settings()
    target = data_dir or settings.data_dir
    target.mkdir(parents=True, exist_ok=True)

    try:
        cache_path = Path(kagglehub.competition_download("ieee-fraud-detection"))
    except UnauthenticatedError as exc:
        msg = (
            "Kaggle authentication required. Configure ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME and KAGGLE_KEY (see data/README.md)."
        )
        raise SystemExit(msg) from exc
    for name in (TRANSACTION_FILE, IDENTITY_FILE):
        src = cache_path / name
        if not src.is_file():
            msg = f"Expected file missing from kagglehub download: {src}"
            raise FileNotFoundError(msg)
        shutil.copy2(src, target / name)

    return target


def load_merged_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    settings = get_settings()
    root = data_dir or settings.data_dir
    txn_path = root / TRANSACTION_FILE
    id_path = root / IDENTITY_FILE
    if not txn_path.is_file() or not id_path.is_file():
        msg = (
            f"Missing dataset under {root}. Run `make download-data` "
            "(requires Kaggle credentials configured for kagglehub)."
        )
        raise FileNotFoundError(msg)

    transactions = pd.read_csv(txn_path)
    identity = pd.read_csv(id_path)
    merged = transactions.merge(identity, on=JOIN_KEY, how="left", suffixes=("", "_id"))
    merged = merged.sort_values(TIME_COLUMN).reset_index(drop=True)
    if merged[TARGET_COLUMN].isna().any():
        msg = f"Missing labels in {TARGET_COLUMN}"
        raise ValueError(msg)
    return merged


def main() -> None:
    path = download_competition()
    print(f"Copied IEEE-CIS files to: {path.resolve()}")


if __name__ == "__main__":
    main()
