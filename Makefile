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

.DEFAULT_GOAL := help
.PHONY: help venv test ci gen-rag-items validate-goldset ingest-squad ingest-uk-squad build-rag-store calibration-worksheet build-index validate-retrieval run-eval prep-models list-models build-vllm

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

venv: ## Create .venv (uv, py3.11), install the package + all extras, seed .env (EXTRAS= to trim)
	@command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found -- install from https://docs.astral.sh/uv/"; exit 1; }
	uv venv --python $(PYTHON_VERSION) "$(VENV)"
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

build-index: ## M1: chunk + embed the gold-set corpus into a FAISS store (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main build-index --corpus-root "$(PROJECT_ROOT)/.data/llb/corpus"

validate-retrieval: ## M1: recall@k / MRR of the pinned embedding over the gold set (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main validate-retrieval --k $(RAG_K)

run-eval: ## Run the eval on one model; MODEL= BACKEND=ollama|vllm LIMIT= SPLIT= TELEMETRY=1
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main run-eval --model "$(MODEL)" --backend "$(BACKEND)" --split "$(SPLIT)" \
		--limit $(LIMIT) $(if $(TELEMETRY),--telemetry,)

build-vllm: ## M2: install vLLM for the host (MAX_JOBS-capped build + wheel cache); GPU host only
	bash "$(PROJECT_ROOT)/scripts/build_vllm.sh"

prep-models: ## Detect GPU, pull Ollama tags + cache vLLM HF weights (MODELS_MANIFEST=, PREP_BACKEND=, gated needs HF_TOKEN)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; \
	$(PY) -m llb.main prep-models --manifest "$(MODELS_MANIFEST)" --backend "$(PREP_BACKEND)"

list-models: ## List which candidate models can run here (GPU+RAM, KV-cache-aware); CONTEXT= to target a context
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main list-models --manifest "$(MODELS_MANIFEST)" $(if $(CONTEXT),--context $(CONTEXT),)
