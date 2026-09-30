# ADR 0002: Time-based split

## Context

IEEE-CIS is temporal (`TransactionDT`). Random splits leak future behavior into training and inflate metrics.

## Options

1. Sort by `TransactionDT`, contiguous 70/15/15 fractions
2. Purged k-fold with embargo (more complex, better for small experiments)
3. Random stratified split (rejected for production narrative)

## Decision

Sort all labeled rows by `TransactionDT`, assign the first 70% to train, next 15% to validation, remainder to test. Threshold tuning uses **validation only**; test is reported once in `make eval`.

## Consequences

- Simple, defensible in interviews; matches streaming replay order later.
- Does not model calendar seasonality explicitly; acceptable for portfolio scope.
- Split metadata is written to `artifacts/split_metadata.json` for audit.
