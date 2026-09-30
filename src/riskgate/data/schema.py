"""IEEE-CIS Fraud Detection column conventions."""

from typing import Final

TIME_COLUMN: Final = "TransactionDT"
TARGET_COLUMN: Final = "isFraud"
JOIN_KEY: Final = "TransactionID"

TRANSACTION_FILE: Final = "train_transaction.csv"
IDENTITY_FILE: Final = "train_identity.csv"

# Numeric transaction fields used in Phase 1 tabular models (velocity features come in Phase 2).
NUMERIC_TRANSACTION_COLUMNS: Final[tuple[str, ...]] = (
    "TransactionAmt",
    "dist1",
    "dist2",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D13",
    "D14",
    "D15",
)

CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
)
