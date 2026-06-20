# loc-lm-bench -- developer entrypoints
SHELL := /bin/bash
PROJECT_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
VENV := $(PROJECT_ROOT)/.venv
PY := $(VENV)/bin/python
PYTHON_VERSION := 3.11

# Milestone 0 artifacts (regeneratable under .data/, gitignored).
GOLDSET := $(PROJECT_ROOT)/.data/llb/goldset/sample_rag_items.jsonl
CORPUS := $(PROJECT_ROOT)/.data/llb/corpus
SQUAD_JSON ?= samples/squad_uk_fixture.json
CORPUS_DIR ?= $(PROJECT_ROOT)/samples/corpus
GOLDSET_N ?= 250

.DEFAULT_GOAL := help
.PHONY: help venv test ci gen-rag-items validate-goldset ingest-squad ingest-uk-squad build-rag-store calibration-worksheet

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

venv: ## Create .venv (uv, py3.11), install deps, seed .env from .env.example
	@command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found -- install from https://docs.astral.sh/uv/"; exit 1; }
	uv venv --python $(PYTHON_VERSION) "$(VENV)"
	uv pip install --python "$(PY)" -e .
	@if [ ! -f "$(PROJECT_ROOT)/.env" ]; then \
		cp "$(PROJECT_ROOT)/.env.example" "$(PROJECT_ROOT)/.env"; \
		echo "[venv] created .env from .env.example"; \
	else \
		echo "[venv] .env already exists, leaving it"; \
	fi
	@echo "[venv] ready: $(VENV)"

test: ## Run the test suite (pytest)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m pytest

ci: ## Lint + unit tests only (no network, GPU, or heavy extras) -- used by GitHub CI
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- create one + install '.[dev]' first"; exit 1; }
	$(VENV)/bin/ruff check src tests
	$(PY) -m pytest

gen-rag-items: ## Generate sample canonical UA RAG gold items into .data/llb/
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	bash "$(PROJECT_ROOT)/scripts/gen_rag_items.sh"

validate-goldset: ## Validate the sample gold set against its corpus (M0 acceptance)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.goldset.validate --goldset "$(GOLDSET)" --corpus-root "$(CORPUS)"

ingest-squad: ## Ingest SQuAD-format UA QA into .data/llb/ (override SQUAD_JSON=path)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.prep.ingest_squad --squad-json "$(SQUAD_JSON)" --out-dir "$(PROJECT_ROOT)/.data/llb"

calibration-worksheet: ## Emit a blank judge-calibration worksheet from the sample gold set
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.judge.calibration worksheet --goldset "$(GOLDSET)" \
		--out "$(PROJECT_ROOT)/.data/llb/calibration_worksheet.csv"

ingest-uk-squad: ## Pull HPLT/ua-squad (GOLDSET_N items, default 250) into .data/llb/ (needs HF_TOKEN in .env + [goldset])
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; \
	$(PY) -m llb.prep.ingest_squad --hf-dataset HPLT/ua-squad --hf-split train \
		--max-items $(GOLDSET_N) --out-name goldset_uk.jsonl --out-dir "$(PROJECT_ROOT)/.data/llb"

build-rag-store: ## Chunk a corpus with all strategies into .data/llb/rag (override CORPUS_DIR=...)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.rag.chunking --corpus-root "$(CORPUS_DIR)" \
		--out-dir "$(PROJECT_ROOT)/.data/llb/rag" --strategy all --size 800 --overlap 120
