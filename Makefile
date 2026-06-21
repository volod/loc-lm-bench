# loc-lm-bench -- developer entrypoints
SHELL := /bin/bash
PROJECT_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
VENV := $(PROJECT_ROOT)/.venv
PY := $(VENV)/bin/python
PYTHON_VERSION := 3.11

# Extras installed by `make venv` -- every declared optional-dependency group, so a fresh
# checkout can run every command without a follow-up `uv pip install`. vLLM/torch/flash-attn
# are deliberately NOT here: they are hardware-matched and built separately (AGENTS.md).
# Override for a lean install, e.g. `make venv EXTRAS=dev`.
EXTRAS ?= rag,eval,track,board,prep,telemetry,goldset,dev

# Milestone 0 artifacts (regeneratable under .data/, gitignored).
GOLDSET := $(PROJECT_ROOT)/.data/llb/goldset/sample_rag_items.jsonl
CORPUS := $(PROJECT_ROOT)/.data/llb/corpus
SQUAD_JSON ?= samples/squad_uk_fixture.json
CORPUS_DIR ?= $(PROJECT_ROOT)/samples/corpus
GOLDSET_N ?= 250

# Milestone 1/2 eval knobs (override on the command line).
MODEL ?= llama3.2:3b
BACKEND ?= ollama
SPLIT ?= final
LIMIT ?= 20
RAG_K ?= 10
MODELS_MANIFEST ?= $(PROJECT_ROOT)/samples/models_uk.yaml
PREP_BACKEND ?= all
# The verified sample set (seeded by `make gen-rag-items` / `make demo-eval`) is the default
# so `make run-eval` works out of the box. Override GOLDSET= to point at another set, e.g.
# the real HPLT/ua-squad set built by `make ingest-uk-squad` (verified=false until reviewed).
GOLDSET ?= $(PROJECT_ROOT)/.data/llb/goldset/sample_rag_items.jsonl

# `make demo-eval` end-to-end pipeline knobs (idempotent; CUDA-free defaults).
ALL_GOLDSET ?= $(PROJECT_ROOT)/.data/llb/goldset/sample_rag_items.jsonl
ALL_CORPUS  := $(PROJECT_ROOT)/.data/llb/corpus
ALL_INDEX   := $(PROJECT_ROOT)/.data/llb/rag/index.faiss
LOG_DIR     := $(PROJECT_ROOT)/.data/llb/logs
PREP_ALL_BACKEND ?= ollama
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000

.DEFAULT_GOAL := help
.PHONY: help venv test format ci gen-rag-items validate-goldset ingest-squad ingest-uk-squad build-rag-store calibration-worksheet build-index validate-retrieval run-eval prep-models list-models build-vllm demo-eval mlflow

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n", $$1, $$2}'

demo-eval: ## End-to-end (idempotent): venv -> gold set -> index -> validate -> prep-models -> run-eval+telemetry; logs to .data/llb/logs/
	@mkdir -p "$(LOG_DIR)"; LOG="$(LOG_DIR)/pipeline-$$(date +%Y%m%d-%H%M%S).log"; \
	echo "[demo-eval] end-to-end pipeline (idempotent); logging to $$LOG"; \
	( \
	  set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; \
	  echo "### [1/6] venv (idempotent; RECREATE_VENV=1 to rebuild)"; \
	  $(MAKE) --no-print-directory venv || exit 1; \
	  echo "### [2/6] seed gold set"; \
	  bash "$(PROJECT_ROOT)/scripts/gen_rag_items.sh" || exit 1; \
	  echo "### [3/6] build index"; \
	  if [ -f "$(ALL_INDEX)" ] && \
	     [ -z "$$(find "$(ALL_CORPUS)" -type f -newer "$(ALL_INDEX)" -print -quit)" ]; \
	  then echo "  cached, skipping -> $(ALL_INDEX)"; \
	  else $(PY) -m llb.main build-index --corpus-root "$(ALL_CORPUS)" || exit 1; fi; \
	  echo "### [4/6] validate retrieval"; \
	  $(PY) -m llb.main validate-retrieval --goldset "$(ALL_GOLDSET)" --k $(RAG_K) \
	    || echo "  WARN: retrieval below the 0.8 gate (non-fatal; continuing)"; \
	  echo "### [5/6] prep models (backend=$(PREP_ALL_BACKEND); cached downloads are skipped)"; \
	  $(MAKE) --no-print-directory prep-models PREP_BACKEND=$(PREP_ALL_BACKEND) || exit 1; \
	  echo "### [6/6] run-eval + telemetry (model=$(MODEL) backend=$(BACKEND))"; \
	  $(PY) -m llb.main run-eval --model "$(MODEL)" --backend "$(BACKEND)" \
	    --goldset "$(ALL_GOLDSET)" --split final --limit $(LIMIT) --telemetry || exit 1; \
	  echo "### pipeline complete"; \
	) 2>&1 | tee "$$LOG"; \
	rc=$${PIPESTATUS[0]}; \
	if [ "$$rc" -eq 0 ]; then \
	  { echo "[demo-eval] OK -- full log: $$LOG"; \
	    echo "[demo-eval] review experiment results: make mlflow"; \
	    echo "[demo-eval] MLflow UI: http://$(MLFLOW_HOST):$(MLFLOW_PORT)"; \
	    echo "[demo-eval] guide: docs/guides/mlflow-analysis.md"; } | tee -a "$$LOG"; \
	else echo "[demo-eval] FAILED (exit $$rc) -- investigate the log: $$LOG" \
	  | tee -a "$$LOG" >&2; exit "$$rc"; fi

mlflow: ## Serve the shared MLflow experiment UI (MLFLOW_HOST=127.0.0.1 MLFLOW_PORT=5000)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main mlflow-ui --host "$(MLFLOW_HOST)" --port "$(MLFLOW_PORT)"

venv: ## Create/update .venv (py3.11) + all extras + .env. Idempotent; RECREATE_VENV=1 to rebuild, EXTRAS= to trim
	@command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found -- install from https://docs.astral.sh/uv/"; exit 1; }
	@if [ -n "$(RECREATE_VENV)" ] && [ -d "$(VENV)" ]; then echo "[venv] RECREATE_VENV set -- removing $(VENV)"; rm -rf "$(VENV)"; fi
	@if [ ! -x "$(PY)" ]; then \
		echo "[venv] creating $(VENV) (py$(PYTHON_VERSION))"; uv venv --python $(PYTHON_VERSION) "$(VENV)"; \
	else \
		echo "[venv] reusing $(VENV) -- updating deps (RECREATE_VENV=1 to rebuild)"; \
	fi
	uv pip install --python "$(PY)" -e ".[$(EXTRAS)]"
	@if [ ! -f "$(PROJECT_ROOT)/.env" ]; then \
		cp "$(PROJECT_ROOT)/.env.example" "$(PROJECT_ROOT)/.env"; \
		echo "[venv] created .env from .env.example"; \
	else \
		echo "[venv] .env already exists, leaving it"; \
	fi
	@echo "[venv] ready: $(VENV) (extras: $(EXTRAS))"
	@echo "[venv] note: vLLM/torch/flash-attn are hardware-matched and installed separately."

test: ## Run the test suite (pytest)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m pytest

format: ## Format Python sources and tests with Ruff
	@test -x "$(VENV)/bin/ruff" || { echo "ERROR: ruff missing -- run 'make venv' first"; exit 1; }
	$(VENV)/bin/ruff format src tests

ci: ## Format check + lint + type check + unit tests -- used by GitHub CI
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- create one + install '.[dev]' first"; exit 1; }
	$(VENV)/bin/ruff format --check src tests
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/mypy
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

build-index: ## M1: chunk + embed the gold-set corpus into a FAISS store (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main build-index --corpus-root "$(PROJECT_ROOT)/.data/llb/corpus"

validate-retrieval: ## M1: recall@k / MRR of the pinned embedding over the gold set (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main validate-retrieval --k $(RAG_K)

run-eval: ## Run the eval on one model; MODEL= BACKEND=ollama|vllm GOLDSET= LIMIT= SPLIT= TELEMETRY=1
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; \
	$(PY) -m llb.main run-eval --model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --split "$(SPLIT)" \
		--limit $(LIMIT) $(if $(TELEMETRY),--telemetry,)

build-vllm: ## Install prebuilt vLLM via uv; VLLM_SOURCE_DIR= builds/caches one checkout wheel
	bash "$(PROJECT_ROOT)/scripts/build_vllm.sh"

prep-models: ## Detect GPU, pull Ollama tags + cache vLLM HF weights (MODELS_MANIFEST=, PREP_BACKEND=, gated needs HF_TOKEN)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; \
	$(PY) -m llb.main prep-models --manifest "$(MODELS_MANIFEST)" --backend "$(PREP_BACKEND)"

list-models: ## List which candidate models can run here (GPU+RAM, KV-cache-aware); CONTEXT= to target a context
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main list-models --manifest "$(MODELS_MANIFEST)" $(if $(CONTEXT),--context $(CONTEXT),)
