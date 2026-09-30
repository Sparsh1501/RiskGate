# ADR 0001: Python tooling and repo layout

## Context

We need reproducible Phase 1 baselines and a layout that later phases (streaming, API, K8s) can extend without renames.

## Options

1. **Monorepo + `src/riskgate` + uv lockfile**
2. Multi-package repo with separate `training/` and `services/` Python packages
3. Pip + unpinned `requirements.txt`

## Decision

Use a single installable package under `src/riskgate`, Docker Compose for infra, **uv** for locked dependencies (`uv.lock`), **ruff** + **mypy --strict**, **pytest**.

Dataset download via **kagglehub** (`competition_download('ieee-fraud-detection')`), copying CSVs into `data/raw/`.

## Consequences

- Later services can import shared feature definitions from the same package.
- Contributors must install uv and run `uv sync`.
- Kaggle credentials are required locally to download data; CI may skip heavy integration unless secrets are provided.
