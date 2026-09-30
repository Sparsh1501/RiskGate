# Evaluation (Phase 1)

## Reproduce metrics

```bash
make sync
make download-data   # Kaggle credentials required
make train-baseline
make eval            # writes results/baseline.json
```

All headline numbers for Phase 1 must come from `results/baseline.json` (generated, not hand-edited).

## Metrics

| Metric | Why |
|--------|-----|
| **PR-AUC** | Sensitive to rare fraud class |
| **Recall @ 1% FPR** | Operational point for low false-decline regimes |
| **Calibration (ECE + bins)** | Scores must be comparable to cost-based thresholds |

We do **not** headline accuracy or ROC-AUC on this dataset.

## Cost assumptions

- `RISKGATE_COST_FP = 1` — cost of a false decline (good customer blocked)
- `RISKGATE_COST_FN = 100` — cost of missed fraud (issuer + customer harm)

Thresholds are chosen on the **validation** time slice by minimizing `cost_fp * FP + cost_fn * FN`. Test metrics use that fixed threshold.

Adjust costs in `.env` and re-run `make train-baseline && make eval` to see sensitivity.

## Phase 1 models

1. **Rules baseline** — amount percentiles, email domain mismatch, identity missingness
2. **Logistic regression** — imputed numeric + one-hot categoricals, `class_weight=balanced`

Velocity features and LightGBM arrive in later phases.

## Error analysis

`results/baseline.json` includes a small honest slice: count and amount stats for fraud missed by logistic regression at the selected threshold.
