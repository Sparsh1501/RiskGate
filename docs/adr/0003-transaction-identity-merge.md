# ADR 0003: Transaction + identity merge

## Context

Identity columns (`id_*`, device fields) live in `train_identity.csv` keyed by `TransactionID`.

## Options

1. Left join identity onto transactions (keep all transactions)
2. Inner join (drop transactions without identity)
3. Keep tables separate with multi-table models

## Decision

**Left join** `train_transaction` with `train_identity` on `TransactionID`. Identity nulls are handled in models (rules: missing-rate signal; LR: imputation).

## Consequences

- Maximizes row count vs inner join (~29% transactions lack identity in IEEE-CIS).
- Slightly noisier identity features for card-not-present fraud signals.
- Same merge logic must be reused in Phase 2 feature processor (documented here to prevent skew).
