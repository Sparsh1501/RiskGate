# ADR 0005: Fraud cost function and threshold selection

## Context

Accuracy and ROC-AUC are misleading on imbalanced fraud data. We need a decision threshold tied to business trade-offs.

## Options

1. Minimize expected cost `cost_fp * FP + cost_fn * FN` on validation scores
2. Fix threshold at recall@1% FPR
3. Maximize F1

## Decision

Default **cost_fn = 100**, **cost_fp = 1** (configurable via `RISKGATE_COST_FN`, `RISKGATE_COST_FP`). Search thresholds on validation predicted scores (up to 500 quantile candidates) and pick minimum expected cost. Report **PR-AUC**, **recall at 1% FPR**, and **calibration (ECE + reliability bins)** on the held-out test split at the validation-selected threshold.

## Consequences

- Assumptions must be stated when quoting decline rates (see `docs/EVALUATION.md`).
- Cost ratios are synthetic until tied to real issuer economics.
- Rules and logistic models each get their own threshold file under `artifacts/`.
