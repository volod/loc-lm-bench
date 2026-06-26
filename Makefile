# loc-lm-bench -- developer entrypoints
SHELL := /bin/bash
PROJECT_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
VENV := $(PROJECT_ROOT)/.venv
PY := $(VENV)/bin/python
PYTHON_VERSION := 3.11
DATA_DIR ?= $(shell bash -c 'source "$(PROJECT_ROOT)/scripts/shared/common.sh"; llb_load_env; printf "%s" "$$DATA_DIR"')

# Extras installed by `make venv` -- every declared optional-dependency group, so a fresh
# checkout can run every command without a follow-up `uv pip install`. vLLM/torch/flash-attn
# are deliberately NOT here: they are hardware-matched and built separately (AGENTS.md).
# Override for a lean install, e.g. `make venv EXTRAS=dev`.
EXTRAS ?= rag,eval,graph,track,board,prep,telemetry,goldset,dev

# Stable human-reviewed development fixture. Runtime imports adopt matching reviewed ids.
PUBLISHED_GOLDSET_ROOT := $(PROJECT_ROOT)/samples/goldsets/ua_squad_postedited_v1
GOLDSET ?= $(PUBLISHED_GOLDSET_ROOT)/goldset.jsonl
CORPUS ?= $(PUBLISHED_GOLDSET_ROOT)/corpus
SQUAD_JSON ?= samples/squad_uk_fixture.json
CORPUS_DIR ?= $(PROJECT_ROOT)/samples/corpus
GOLDSET_N ?= 250
GOLDSET_MODE ?= development
# M4.4 ontology-assisted draft mode (GOLDSET_MODE=draft over CORPUS).
DRAFT_MODEL ?= llama3.2:3b
DRAFT_ENDPOINT ?= local
DRAFT_MAX_ITEMS ?= 60

# Milestone 1/2 eval knobs (override on the command line).
MODEL ?= llama3.2:3b
BACKEND ?= ollama
SPLIT ?= final
LIMIT ?= 20
RAG_K ?= 10
MODELS_MANIFEST ?= $(PROJECT_ROOT)/samples/models_uk.yaml
PREP_BACKEND ?= all
# `make demo-eval` end-to-end pipeline knobs (idempotent; CUDA-free defaults).
ALL_GOLDSET ?= $(GOLDSET)
ALL_CORPUS  ?= $(CORPUS)
LOG_DIR     := $(DATA_DIR)/llb/logs
PREP_ALL_BACKEND ?= ollama
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000
# Judge knobs (M3.8). JUDGE_MODEL is the model id exposed by a LOCAL OpenAI-compatible endpoint
# (no data egress + reproducible; bias documented in current.md); JUDGE_BASE_URL points at it.
# Default = the Ollama gemma3:27b judge on :11434 (the default BACKEND=ollama candidate runs there
# too). Alternatives by GPU tier (override JUDGE_MODEL + JUDGE_BASE_URL):
#   12 GB GPU: ollama_chat/gemma-4-e4b-it                       (GGUF/CPU offload; the 12B won't fit)
#   16 GB GPU: hosted_vllm/google/gemma-4-12B-it-qat-w4a16-ct   (vLLM on :8000)
#   32 GB GPU: hosted_vllm/google/gemma-4-12B-it                (bf16, higher fidelity + co-host headroom)
# Set JUDGE_MODEL empty to skip the judge.
# JUDGE_RHO is the calibration Spearman rho (from `make calibration-score`); set it on `run-eval`
# to ENABLE the gated judge in a scored run -- the judge enters the ranking blend only when
# JUDGE_RHO >= 0.6 and the decision is recorded in the run manifest. Unset -> no judge (default).
# LLB_EMBED_DEVICE pins the sentence-transformers embedder to the CPU by default so the GPU stays
# free for a co-resident local judge/candidate (a big judge + a GPU embedder OOMs 16 GB); override
# with LLB_EMBED_DEVICE=cuda when nothing else needs the GPU.
# Calibration worksheets carry irreducibly-human ratings, kept in TWO roots:
#   - PERMANENT: the tracked root calibration/ dir -- committed, so they survive a clone.
#   - TEMPORARY: $(DATA_DIR)/llb/calibration -- gitignored (generated/in-progress sets).
# CAL_NAME labels the use case (one worksheet per goldset). A worksheet AUTO-ROUTES by name:
# names listed in CAL_PERMANENT go to root calibration/, everything else to the temp dir -- so a
# new/generated set stays local by default. To persist one: copy it into calibration/ and add its
# name to CAL_PERMANENT (or commit it directly). CAL_DIR overrides the routing explicitly.
CAL_PERMANENT ?= ua_squad_postedited_v1
CAL_NAME ?= ua_squad_postedited_v1
CAL_DIR ?= $(if $(filter $(CAL_NAME),$(CAL_PERMANENT)),calibration,$(DATA_DIR)/llb/calibration)
CAL_WS ?= $(CAL_DIR)/$(CAL_NAME).csv
RATINGS ?= $(CAL_WS)
JUDGE_MODEL ?= gemma3:27b
JUDGE_BASE_URL ?= http://localhost:11434/v1
JUDGE_RHO ?=
LLB_EMBED_DEVICE ?= cpu
export LLB_EMBED_DEVICE
APT_PROFILE ?= production

.DEFAULT_GOAL := help
.PHONY: help venv apt-deps test test-fast format ci gen-rag-items validate-goldset ingest-squad ingest-uk-squad build-rag-store calibration-worksheet calibration-run calibration-rate calibration-score judge-experiment build-index validate-retrieval run-eval prep-models list-models build-vllm demo-eval mlflow detect-gpu-vram gen-serving-config

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n", $$1, $$2}'

demo-eval: ## End-to-end: venv -> committed gold set -> index -> validate -> prep-models -> run-eval+telemetry
	@source "$(PROJECT_ROOT)/scripts/shared/common.sh"; \
	llb_ensure_env || exit 0; \
	mkdir -p "$(LOG_DIR)"; LOG="$(LOG_DIR)/pipeline-$$(date +%Y%m%d-%H%M%S).log"; \
	echo "[demo-eval] end-to-end pipeline (idempotent); logging to $$LOG"; \
	( \
	  llb_load_env; \
	  echo "### [1/6] venv (idempotent; RECREATE_VENV=1 to rebuild)"; \
	  $(MAKE) --no-print-directory venv || exit 1; \
	  echo "### [2/6] validate committed published gold set"; \
	  $(PY) -m llb.goldset.validate --goldset "$(ALL_GOLDSET)" \
	    --corpus-root "$(ALL_CORPUS)" || exit 1; \
	  echo "### [3/6] build index"; \
	  $(PY) -m llb.main build-index --corpus-root "$(ALL_CORPUS)" || exit 1; \
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

venv: ## Create/update .venv (py3.11) + apt deps + all extras + .env. Idempotent; RECREATE_VENV=1 to rebuild, EXTRAS= to trim, SKIP_APT=1 to skip apt
	@command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found -- install from https://docs.astral.sh/uv/"; exit 1; }
	@SKIP_APT="$(SKIP_APT)" bash "$(PROJECT_ROOT)/scripts/install_apt_deps.sh" production
	@case ",$(EXTRAS)," in *,dev,*|*,dev) SKIP_APT="$(SKIP_APT)" bash "$(PROJECT_ROOT)/scripts/install_apt_deps.sh" dev ;; esac
	@if [ -n "$(RECREATE_VENV)" ] && [ -d "$(VENV)" ]; then echo "[venv] RECREATE_VENV set -- removing $(VENV)"; rm -rf "$(VENV)"; fi
	@if [ ! -x "$(PY)" ]; then \
		echo "[venv] creating $(VENV) (py$(PYTHON_VERSION))"; uv venv --python $(PYTHON_VERSION) "$(VENV)"; \
	else \
		echo "[venv] reusing $(VENV) -- updating deps (RECREATE_VENV=1 to rebuild)"; \
	fi
	@UV_LINK_MODE="$(UV_LINK_MODE)" bash -c 'source "$(PROJECT_ROOT)/scripts/shared/common.sh"; llb_export_uv_link_mode; echo "[venv] uv link mode: $${UV_LINK_MODE:-default (cache + checkout share a device)}"; uv pip install --python "$(PY)" -e ".[$(EXTRAS)]"'
	@echo "[venv] ready: $(VENV) (extras: $(EXTRAS))"
	@echo "[venv] note: vLLM/torch/flash-attn are hardware-matched and installed separately."
	@bash -c 'source "$(PROJECT_ROOT)/scripts/shared/common.sh"; llb_ensure_env || true'

apt-deps: ## Install apt packages (APT_PROFILE=production|dev|all; SKIP_APT=1 to skip; APT_DRY_RUN=1 to list only)
	@SKIP_APT="$(SKIP_APT)" APT_DRY_RUN="$(APT_DRY_RUN)" bash "$(PROJECT_ROOT)/scripts/install_apt_deps.sh" "$(APT_PROFILE)"

# Two test groups (markers registered in pyproject.toml):
#   `make test`      -- FULL local suite: every test, including the `slow` ones (real Optuna
#                       sweeps, embedder/model loads, deepeval, subprocess builds).
#   `make ci` / `test-fast` -- LIGHTWEIGHT suite (`-m "not slow"`) so GitHub CI stays fast.
NOT_SLOW := -m "not slow"

test: ## Run the FULL test suite locally (pytest, includes slow tests)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m pytest

test-fast: ## Run the lightweight test suite (skips slow tests; mirrors CI)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m pytest $(NOT_SLOW)

format: ## Format Python sources and tests with Ruff
	@test -x "$(VENV)/bin/ruff" || { echo "ERROR: ruff missing -- run 'make venv' first"; exit 1; }
	$(VENV)/bin/ruff format src tests

ci: ## Format check + lint + type check + LIGHTWEIGHT unit tests -- used by GitHub CI
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- create one + install '.[dev]' first"; exit 1; }
	$(VENV)/bin/ruff format --check src tests
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/mypy
	$(PY) -m pytest $(NOT_SLOW)

gen-rag-items: ## Generate sample canonical UA RAG gold items into .data/llb/
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	bash "$(PROJECT_ROOT)/scripts/gen_rag_items.sh"

validate-goldset: ## Validate GOLDSET against CORPUS (defaults to the committed fixture)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.goldset.validate --goldset "$(GOLDSET)" --corpus-root "$(CORPUS)"

ingest-squad: ## Ingest local SQuAD QA; matching reviewed ids are verified (SQUAD_JSON=path)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.prep.ingest_squad --squad-json "$(SQUAD_JSON)"

calibration-worksheet: ## Emit a blank judge-calibration worksheet from GOLDSET
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.judge.calibration worksheet --goldset "$(GOLDSET)" \
		--out "$(CAL_WS)"

calibration-run: ## Run MODEL on the calibration split -> filled worksheet (model_answer + judge_rating if JUDGE_MODEL set)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main run-eval --model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --split calibration --worksheet "$(CAL_WS)" \
		$(if $(JUDGE_MODEL),--judge-model "$(JUDGE_MODEL)",) \
		$(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",)

calibration-rate: ## Interactively fill human ratings/answers in CAL_WS (judge_rating hidden; SHOW_JUDGE=1 to reveal, START=N, CLEAR=1 to reset)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.judge.calibration rate --worksheet "$(CAL_WS)" $(if $(START),--start $(START)) $(if $(SHOW_JUDGE),--show-judge) $(if $(CLEAR),--clear)

calibration-score: ## Score a filled worksheet: rho + bootstrap CI + trust decision (RATINGS=path, gate rho>=0.6)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.judge.calibration score --ratings "$(RATINGS)"

judge-experiment: ## Run fixed UA judge cases against a local OpenAI-compatible endpoint
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main judge-experiment --judge-model "$(JUDGE_MODEL)" \
		$(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",)

ingest-uk-squad: ## Development utility: GOLDSET_MODE=development|skeleton|draft (draft is M4)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@echo "[ingest-uk-squad] mode=$(GOLDSET_MODE)"; \
	case "$(GOLDSET_MODE)" in \
	  development) \
	    set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; \
	    $(PY) -m llb.prep.ingest_squad --pinned-development-source \
	      --max-items $(GOLDSET_N) \
	      --out-name goldset_uk_development.jsonl ;; \
	  skeleton) \
	    $(PY) -m llb.prep.goldset_skeleton ;; \
	  draft) \
	    set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; \
	    $(PY) -m llb.main prepare-goldset-draft --corpus-root "$(CORPUS)" \
	      --model "$(DRAFT_MODEL)" --endpoint "$(DRAFT_ENDPOINT)" --max-items $(DRAFT_MAX_ITEMS) ;; \
	  *) \
	    echo "ERROR: GOLDSET_MODE must be development, skeleton, or draft" >&2; exit 2 ;; \
	esac

build-rag-store: ## Chunk a corpus with all strategies into DATA_DIR/llb/rag (CORPUS_DIR=...)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.rag.chunking --corpus-root "$(CORPUS_DIR)" \
		--out-dir "$(DATA_DIR)/llb/rag" --strategy all --size 800 --overlap 120

build-index: ## M1: chunk + embed CORPUS into the FAISS store (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main build-index --corpus-root "$(CORPUS)"

build-graph: ## M6: build the GraphRAG store from an M4.4 draft bundle (BUNDLE=...; needs ".[graph]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<prepare-goldset dir> (extraction.jsonl + corpus/)"; exit 1; }
	$(PY) -m llb.main build-graph --bundle "$(BUNDLE)"

validate-retrieval: ## M1: recall@k / MRR of the pinned embedding over the gold set (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main validate-retrieval --goldset "$(GOLDSET)" --k $(RAG_K)

run-eval: ## Run the eval; MODEL= BACKEND=ollama|vllm GOLDSET= LIMIT= SPLIT= TELEMETRY=1 JUDGE_RHO= (enables the gated judge)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; \
	$(PY) -m llb.main run-eval --model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --split "$(SPLIT)" \
		--limit $(LIMIT) $(if $(TELEMETRY),--telemetry) $(if $(JUDGE_RHO),--judge-rho $(JUDGE_RHO) --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)"))

build-vllm: ## Install prebuilt vLLM via uv; VLLM_SOURCE_DIR= builds/caches one checkout wheel
	bash "$(PROJECT_ROOT)/scripts/build_vllm.sh"

build-llamacpp: ## Build CUDA llama-server for the M4.5 launcher; CUDA_ARCH=/LLAMACPP_REF= override
	bash "$(PROJECT_ROOT)/scripts/build_llamacpp.sh"

prep-models: ## Detect GPU, pull Ollama tags + cache vLLM HF weights (MODELS_MANIFEST=, PREP_BACKEND=, gated needs HF_TOKEN)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; \
	$(PY) -m llb.main prep-models --manifest "$(MODELS_MANIFEST)" --backend "$(PREP_BACKEND)"

list-models: ## List which candidate models can run here (GPU+RAM, KV-cache-aware); CONTEXT= to target a context
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main list-models --manifest "$(MODELS_MANIFEST)" $(if $(CONTEXT),--context $(CONTEXT),)

detect-gpu-vram: ## Print supported GPU VRAM tier (12/16/24/32 GiB) from nvidia-smi
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main detect-gpu-vram

gen-serving-config: ## Emit serve + run-eval artifacts under .data/llb/serving/; GPU_GB=12|16|24|32 overrides detect
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main gen-serving-config $(if $(GPU_GB),--gpu-gb $(GPU_GB),)
