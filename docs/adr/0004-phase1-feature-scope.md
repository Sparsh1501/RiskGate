# ADR 0004: Phase 1 feature scope

## Context

Phase 2 introduces rolling velocity features in Redis/Parquet from a shared definition module. Phase 1 must still produce credible baselines and ablation hooks.

## Options

1. Static IEEE tabular columns only (numeric `C*`, `D*`, `dist*`, amount, card/email/device categoricals)
2. Precompute velocity offline in Phase 1 (duplicates Phase 2 work)
3. Use only anonymous `V*` columns (harder to explain in interviews)

## Decision

Phase 1 uses an explicit list of **static** transaction + identity columns in `riskgate.data.schema` and `riskgate.features.tabular`. Velocity windows (1m/1h/24h) are **out of scope** until Phase 2.

## Consequences

- LightGBM-with-velocity ablation waits for Phase 2/3.
- Logistic regression remains interpretable with imputation + scaling.
- Feature list changes require a new ADR and regenerated `results/baseline.json`.
