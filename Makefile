.PHONY: help sync lint typecheck test up down download-data train-baseline eval

UV ?= uv
PYTHON ?= $(UV) run

help:
	@echo "Targets: sync, lint, typecheck, test, up, down, download-data, train-baseline, eval"

sync:
	$(UV) sync --all-extras --frozen 2>/dev/null || $(UV) sync --all-extras

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

typecheck:
	$(UV) run mypy src/riskgate

test:
	$(UV) run pytest

up:
	docker compose up -d

down:
	docker compose down

download-data:
	$(PYTHON) riskgate-download

train-baseline:
	$(PYTHON) riskgate-train-baseline

eval:
	$(PYTHON) riskgate-eval
