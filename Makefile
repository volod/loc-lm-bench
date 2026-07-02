# loc-lm-bench -- developer entrypoints
SHELL := /bin/bash
PROJECT_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
VENV := $(PROJECT_ROOT)/.venv
PY := $(VENV)/bin/python
PYTHON_VERSION := 3.13
DATA_DIR ?= $(shell bash -c 'source "$(PROJECT_ROOT)/scripts/shared/common.sh"; llb_load_env; printf "%s" "$$DATA_DIR"')

# Tool caches live under one $DATA_DIR/cache/<tool> tree (not the project root), so the root stays
# clean and `rm -rf $DATA_DIR` clears every temporary artifact in one shot. These env vars override
# the static pyproject defaults, so the caches follow a custom DATA_DIR. deepeval reads
# DEEPEVAL_CACHE_FOLDER (its `.deepeval` keystore) and DEEPEVAL_RESULTS_FOLDER.
LLB_CACHE_DIR := $(DATA_DIR)/cache
export RUFF_CACHE_DIR := $(LLB_CACHE_DIR)/ruff
export MYPY_CACHE_DIR := $(LLB_CACHE_DIR)/mypy
export DEEPEVAL_CACHE_FOLDER := $(LLB_CACHE_DIR)/deepeval
export DEEPEVAL_RESULTS_FOLDER := $(LLB_CACHE_DIR)/deepeval/results
PYTEST_CACHE_OPT := -o cache_dir=$(LLB_CACHE_DIR)/pytest

# Extras installed by `make venv` -- the standard local workflow groups, so a fresh checkout can
# run the normal test/eval/vector-store commands without a follow-up `uv pip install`.
# vLLM/torch/flash-attn are deliberately NOT in pyproject extras: they are hardware-matched
# (AGENTS.md) and installed by scripts/build_vllm.sh after the editable install on CUDA hosts.
# CrewAI remains a dedicated environment because its pins conflict with dev/RAG extras.
# Override for a lean install, e.g. `make venv EXTRAS=dev`.
EXTRAS ?= rag,rag-chroma,rag-qdrant,eval,graph,track,board,viz,prep,telemetry,goldset,dev
VENV_INSTALL_VLLM ?= auto

# Stable human-reviewed development fixture. Runtime imports adopt matching reviewed ids.
PUBLISHED_GOLDSET_ROOT := $(PROJECT_ROOT)/samples/goldsets/ua_squad_postedited_v1
GOLDSET ?= $(PUBLISHED_GOLDSET_ROOT)/goldset.jsonl
CORPUS ?= $(PUBLISHED_GOLDSET_ROOT)/corpus
SQUAD_JSON ?= samples/squad_uk_fixture.json
CORPUS_DIR ?= $(PROJECT_ROOT)/samples/corpus
PDF_DIR ?= $(DATA_DIR)/quickstart-pdf-corpus
PDF_OUT_DIR ?=
PDF_MIN_CHARS ?=
PDF_PARSER ?= auto
PDF_REFRESH ?=
GOLDSET_N ?= 250
GOLDSET_MODE ?= development
# Ontology-assisted draft mode (GOLDSET_MODE=draft over CORPUS).
DRAFT_MODEL ?= gemma4:e4b
DRAFT_ENDPOINT ?= local
DRAFT_BACKEND ?= ollama
DRAFT_BASE_URL ?=
DRAFT_MAX_ITEMS ?= 60
DRAFT_CORPUS ?= $(CORPUS)
DRAFT_DOC_LIMIT ?=
DRAFT_EXTRACT_MAX_CHARS ?=
DRAFT_EXTRACT_CHUNK_OVERLAP ?=
DRAFT_CONCURRENCY ?=
DRAFT_MAX_TOKENS ?= 4096
DRAFT_TEMPERATURE ?= 0
DRAFT_TIMEOUT ?= 300
DRAFT_NO_THINK ?= 0
DRAFT_NUM_CTX ?=
DRAFT_VLLM_PORT ?= 8000
DRAFT_VLLM_GPU_MEMORY_UTILIZATION ?= 0.85
DRAFT_VLLM_MAX_MODEL_LEN ?=
DRAFT_VLLM_DTYPE ?= auto
DRAFT_VLLM_QUANTIZATION ?=
DRAFT_VLLM_STARTUP_TIMEOUT ?= 600
DRAFT_EXTRACTOR ?= llm
DRAFT_OUT_DIR ?=
DRAFT_VERIFY_N ?= 0
DRAFT_RETRIEVAL_INDEX_DIR ?=
DRAFT_RETRIEVAL_K ?= $(RAG_K)
DRAFT_DROP_NONRETRIEVABLE_NEEDLES ?= 0

# RAG/vLLM eval knobs (override on the command line). SMOKE_MODEL is intentionally small
# and should be used for connectivity checks only, not leaderboard or extended tests.
SMOKE_MODEL ?= llama3.2:3b
MODEL ?= $(SMOKE_MODEL)
BACKEND ?= ollama
SPLIT ?= final
LIMIT ?= 20
RAG_K ?= 10
MODELS_MANIFEST ?= $(PROJECT_ROOT)/samples/models_uk.yaml
PREP_BACKEND ?= all
SERVING_TIER_JSON ?=
LLB_OLLAMA_PULL_TIMEOUT_S ?= 1800
PROMPT_SYSTEM_CORPUS ?= $(CORPUS)
PROMPT_SYSTEM_OUT_DIR ?=
PROMPT_SYSTEM_RUN_DIR ?= $(PROMPT_SYSTEM_OUT_DIR)
PROMPT_SYSTEM_ID ?=
PROMPT_PACKAGE ?=
PROMPT_SYSTEM_CONTEXT_WINDOW ?= 8192
PROMPT_SYSTEM_CHUNK_TOKENS ?= 1024
PROMPT_SYSTEM_ANSWER_TOKENS ?= 512
PROMPT_SYSTEM_MAX_PASSAGES ?= 12
PROMPT_SYSTEM_ROLE ?=
PROMPT_SYSTEM_INSTRUCTION ?=
PROMPT_SYSTEM_ACTION ?= summary
PROMPT_SYSTEM_NOTE ?=
PROMPT_SYSTEM_LANE ?= rag
PROMPT_SYSTEM_HARNESS ?=
AGENTIC_TASKS ?= $(PROJECT_ROOT)/samples/agentic_tasks_uk.json
AGENTIC_MAX_STEPS ?= 6
AGENTIC_HARNESS ?= loop
AGENTIC_HARNESSES ?= loop langgraph crewai
AGENTIC_BASE_URL ?=
# `make demo-eval` end-to-end pipeline knobs (idempotent; CUDA-free defaults).
ALL_GOLDSET ?= $(GOLDSET)
ALL_CORPUS  ?= $(CORPUS)
LOG_DIR     := $(DATA_DIR)/llb/logs
PREP_ALL_BACKEND ?= ollama
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000
BOARD_HOST ?= 127.0.0.1
BOARD_PORT ?= 8501
RECOMMEND_MIN_CASES ?= 1
RECOMMEND_GPU_GB ?=
RECOMMEND_MIN_TOK_S ?=
RECOMMEND_JSON_OUT ?=
RECOMMEND_NO_CHART ?=
SWEEP_ID ?= run1
SWEEP_MAX_MODEL_LEN ?= 8192
SWEEP_OFFLINE ?=
SWEEP_LIMIT ?=
# Default RAG grid: sweep retrieval depth so the best top_k is DEMONSTRATED per model, not assumed.
# The best depth varies by model (e.g. MamayLM-12B peaks at top_k=3, mistral at top_k=8), and top_k
# is in the cell fingerprint so re-runs resume. Set SWEEP_RAG_GRID= (empty) to disable the grid.
SWEEP_RAG_GRID ?= top_k=3,5,8
PIPELINE_TOP_N ?= 2
PIPELINE_TRIALS ?= 20
PIPELINE_OFFLINE ?=
# Judge knobs (judge calibration gate). JUDGE_MODEL is the model id exposed by a LOCAL OpenAI-compatible endpoint
# (no data egress + reproducible; bias documented in current.md); JUDGE_BASE_URL points at it.
# Default = the Ollama gemma3:27b judge on :11434 (the default BACKEND=ollama candidate runs there
# too). Alternatives by GPU tier (override JUDGE_MODEL + JUDGE_BASE_URL):
#   12 GB GPU: gemma-4-e4b-it                                  (GGUF/CPU offload; the 12B won't fit)
#   16 GB GPU: google/gemma-4-12B-it-qat-w4a16-ct              (vLLM on :8000)
#   32 GB GPU: google/gemma-4-12B-it                           (bf16, higher fidelity + co-host headroom)
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
# Data-verification knobs (the new-goldset flow: cross-check -> human verification gate sample-verify). BUNDLE is a
# draft dir (goldset.jsonl + corpus/) under $(DATA_DIR)/prepare-goldset/<ts>/; the sample worksheet
# + accepted-ledger default beside it. CROSS_CHECK_MODEL is the SECOND-frontier verifier (must
# differ from the drafter). VERIFY_N sizes the stratified sample; VERIFY_TOLERANCE is the accepted
# reject rate. Full workflow: docs/guides/goldset-from-scratch.md.
BUNDLE ?=
CROSS_CHECK_MODEL ?=
VERIFY_WS ?= $(if $(BUNDLE),$(BUNDLE)/verify_sample.csv,)
VERIFY_N ?= 30
VERIFY_SEED ?= 13
VERIFY_TOLERANCE ?= 0.05
# Composite-headline pipeline knobs. Each verification ref must point at a reviewed
# verify_sample.csv, a sample_manifest.json that points to one, or an accepted-ledger bundle.
COMPOSITE_SAMPLE_VERIFICATION_ROOT ?= $(PROJECT_ROOT)/samples/verification/composite_samples
COMPOSITE_TEXT_ANALYSIS_BUNDLE ?= $(PROJECT_ROOT)/samples/text_analysis_bundle_uk
COMPOSITE_SUMMARIZATION_CASES ?= $(PROJECT_ROOT)/samples/summarization_cases_uk.json
COMPOSITE_STRUCTURED_CASES ?= $(PROJECT_ROOT)/samples/structured_cases_uk.json
COMPOSITE_SECURITY_CASES ?= $(PROJECT_ROOT)/samples/security_cases_uk.json
COMPOSITE_AGENTIC_TASKS ?= $(PROJECT_ROOT)/samples/agentic_tasks_uk.json
COMPOSITE_TOOLING_CATALOG ?= $(PROJECT_ROOT)/samples/tooling_cases_uk.json
COMPOSITE_VERIFICATION_REF ?=
COMPOSITE_TEXT_ANALYSIS_VERIFICATION_REF ?= $(if $(COMPOSITE_VERIFICATION_REF),$(COMPOSITE_VERIFICATION_REF),$(COMPOSITE_SAMPLE_VERIFICATION_ROOT)/text_analysis/sample_manifest.json)
COMPOSITE_SUMMARIZATION_VERIFICATION_REF ?= $(if $(COMPOSITE_VERIFICATION_REF),$(COMPOSITE_VERIFICATION_REF),$(COMPOSITE_SAMPLE_VERIFICATION_ROOT)/summarization/sample_manifest.json)
COMPOSITE_STRUCTURED_VERIFICATION_REF ?= $(if $(COMPOSITE_VERIFICATION_REF),$(COMPOSITE_VERIFICATION_REF),$(COMPOSITE_SAMPLE_VERIFICATION_ROOT)/structured/sample_manifest.json)
COMPOSITE_SECURITY_VERIFICATION_REF ?= $(if $(COMPOSITE_VERIFICATION_REF),$(COMPOSITE_VERIFICATION_REF),$(COMPOSITE_SAMPLE_VERIFICATION_ROOT)/security/sample_manifest.json)
COMPOSITE_AGENTIC_VERIFICATION_REF ?= $(if $(COMPOSITE_VERIFICATION_REF),$(COMPOSITE_VERIFICATION_REF),$(COMPOSITE_SAMPLE_VERIFICATION_ROOT)/agentic/sample_manifest.json)
COMPOSITE_TOOLING_VERIFICATION_REF ?= $(if $(COMPOSITE_VERIFICATION_REF),$(COMPOSITE_VERIFICATION_REF),$(COMPOSITE_SAMPLE_VERIFICATION_ROOT)/tooling/sample_manifest.json)
COMPOSITE_BASE_URL ?=
COMPOSITE_REAL_CORPUS ?=
SECURITY_CASES ?= $(COMPOSITE_SECURITY_CASES)
SECURITY_VERIFICATION_REF ?= $(COMPOSITE_SECURITY_VERIFICATION_REF)
SECURITY_DATA_VERIFIED ?= 1
SECURITY_MODEL ?= hf.co/INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-GGUF:Q4_K_M
SECURITY_BACKEND ?= ollama
SECURITY_BASE_URL ?=
SECURITY_MAX_MODEL_LEN ?=
JUDGE_MODEL ?= gemma3:27b
JUDGE_BASE_URL ?= http://localhost:11434/v1
JUDGE_RHO ?=
LLB_EMBED_DEVICE ?= cpu
export LLB_EMBED_DEVICE
APT_PROFILE ?= production
# Platform and vector-store matrix knobs. Defaults target the common Gemma 4 E4B logical base that fits this
# 16 GB CUDA host across all three backend families. Override these to run a larger common base.
PLATFORM_MATRIX_GOLDSET ?= $(GOLDSET)
PLATFORM_MATRIX_SPLIT ?= final
PLATFORM_MATRIX_LIMIT ?= 20
PLATFORM_MATRIX_MAX_MODEL_LEN ?= 8192
PLATFORM_MATRIX_GPU_MEMORY_UTILIZATION ?= 0.80
PLATFORM_MATRIX_OLLAMA_MODEL ?= gemma4:e4b
PLATFORM_MATRIX_VLLM_MODEL ?= google/gemma-4-E4B-it-qat-w4a16-ct
PLATFORM_MATRIX_LLAMACPP_MODEL ?= hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf:q4_0-it
PLATFORM_MATRIX_LLAMACPP_GPU_LAYERS ?= -1
PLATFORM_MATRIX_BACKENDS ?= ollama vllm llamacpp
PLATFORM_MATRIX_STRICT ?= 0

# README quickstart orchestration knobs. The public interface is the quickstart-* make targets;
# scripts/quickstart.sh owns consistent logging and step summaries for all grouped commands.
QUICKSTART_ROOT ?= $(DATA_DIR)
QUICKSTART_LOG_DIR ?= $(DATA_DIR)/llb/logs/quickstart
QUICKSTART_UV_CACHE_DIR ?= $(QUICKSTART_ROOT)/uv-cache
QUICKSTART_A_DATA_DIR ?= $(QUICKSTART_ROOT)/quickstart-leaderboard
QUICKSTART_A_GOLDSET ?= $(GOLDSET)
QUICKSTART_A_CORPUS ?= $(CORPUS)
QUICKSTART_A_SWEEP_ID ?= qs-committed
QUICKSTART_SKIP_APT ?= 1
QUICKSTART_SETUP_VENV ?= auto
QUICKSTART_PREP_MODELS ?= 1
QUICKSTART_PREP_SERVING_TARGETS ?= 1
QUICKSTART_RUN_SWEEP ?= 1
QUICKSTART_RUN_PLATFORM_MATRIX ?= 1
QUICKSTART_RUN_SECURITY ?= 1
QUICKSTART_RECOMMEND_MIN_CASES ?= $(RECOMMEND_MIN_CASES)
QUICKSTART_SWEEP_LIMIT ?= $(LIMIT)
QUICKSTART_GPU_GB ?=
QUICKSTART_PROMPT_DIR ?= $(QUICKSTART_A_DATA_DIR)/prompt-system/quickstart
QUICKSTART_PROMPT_ID ?=
QUICKSTART_SECURITY_MODEL ?= $(SECURITY_MODEL)
QUICKSTART_SECURITY_BACKEND ?= $(SECURITY_BACKEND)
QUICKSTART_SECURITY_CASES ?= $(SECURITY_CASES)
QUICKSTART_SECURITY_VERIFICATION_REF ?= $(SECURITY_VERIFICATION_REF)
QUICKSTART_PDF_SOURCE ?= $(QUICKSTART_ROOT)/quickstart-pdf-corpus
QUICKSTART_PDF_MD ?= $(QUICKSTART_ROOT)/quickstart-pdf-corpus-md
QUICKSTART_PDF_RAG_DATA ?= $(QUICKSTART_ROOT)/quickstart-pdf-corpus-rag
QUICKSTART_PDF_DRAFT_MD ?= $(QUICKSTART_ROOT)/quickstart-pdf-corpus-draft-md
QUICKSTART_PDF_DRAFT ?= $(QUICKSTART_ROOT)/quickstart-pdf-corpus-draft
QUICKSTART_PDF_GRAPH_DATA ?= $(QUICKSTART_ROOT)/quickstart-pdf-corpus-graph
QUICKSTART_PDF_LEADERBOARD_DATA ?= $(QUICKSTART_ROOT)/quickstart-pdf-corpus-leaderboard
QUICKSTART_PDF_MODEL_BENCH_DATA ?= $(QUICKSTART_A_DATA_DIR)
QUICKSTART_PDF_ACCEPTED ?= $(QUICKSTART_PDF_DRAFT)/accepted
QUICKSTART_PDF_DRAFT_DOCS ?= all
QUICKSTART_DRAFT_MODEL ?= auto
QUICKSTART_DRAFT_ENDPOINT ?= local
QUICKSTART_DRAFT_BACKEND ?= ollama
QUICKSTART_DRAFT_BASE_URL ?=
QUICKSTART_DRAFT_MAX_ITEMS ?= 180
QUICKSTART_DRAFT_VERIFY_N ?= 40
QUICKSTART_DRAFT_TIMEOUT ?= 900
QUICKSTART_DRAFT_MAX_TOKENS ?= 4096
QUICKSTART_DRAFT_TEMPERATURE ?= 0
# Right-sized Ollama context for drafting: extraction windows are bounded (12k chars), so the
# modelfile default (often 128k+) only wastes VRAM and forces CPU offload on 16 GB hosts.
QUICKSTART_DRAFT_NUM_CTX ?= 16384
QUICKSTART_DRAFT_VLLM_PORT ?= 8000
QUICKSTART_DRAFT_VLLM_GPU_MEMORY_UTILIZATION ?= 0.85
QUICKSTART_DRAFT_VLLM_MAX_MODEL_LEN ?=
QUICKSTART_DRAFT_VLLM_DTYPE ?= auto
QUICKSTART_DRAFT_VLLM_QUANTIZATION ?=
QUICKSTART_DRAFT_VLLM_STARTUP_TIMEOUT ?= 600
QUICKSTART_DRAFT_EXTRACT_MAX_CHARS ?=
QUICKSTART_DRAFT_EXTRACT_CHUNK_OVERLAP ?=
QUICKSTART_DRAFT_CONCURRENCY ?=
QUICKSTART_MODEL_SELECTION ?= auto
QUICKSTART_ASSUME_YES ?= 0
QUICKSTART_PDF_MIN_CHARS ?= 500
QUICKSTART_PDF_PARSER ?= auto
export QUICKSTART_ROOT QUICKSTART_LOG_DIR QUICKSTART_UV_CACHE_DIR QUICKSTART_A_DATA_DIR QUICKSTART_A_GOLDSET
export QUICKSTART_A_CORPUS QUICKSTART_A_SWEEP_ID QUICKSTART_SKIP_APT QUICKSTART_SETUP_VENV
export QUICKSTART_PREP_MODELS QUICKSTART_PREP_SERVING_TARGETS QUICKSTART_RUN_SWEEP
export QUICKSTART_RUN_PLATFORM_MATRIX QUICKSTART_RUN_SECURITY QUICKSTART_RECOMMEND_MIN_CASES
export QUICKSTART_SWEEP_LIMIT QUICKSTART_GPU_GB
export QUICKSTART_PROMPT_DIR QUICKSTART_PROMPT_ID QUICKSTART_SECURITY_MODEL
export QUICKSTART_SECURITY_BACKEND QUICKSTART_SECURITY_CASES QUICKSTART_SECURITY_VERIFICATION_REF
export QUICKSTART_PDF_SOURCE QUICKSTART_PDF_MD QUICKSTART_PDF_RAG_DATA
export QUICKSTART_PDF_DRAFT_MD QUICKSTART_PDF_DRAFT QUICKSTART_PDF_GRAPH_DATA
export QUICKSTART_PDF_LEADERBOARD_DATA QUICKSTART_PDF_MODEL_BENCH_DATA QUICKSTART_PDF_ACCEPTED
export QUICKSTART_PDF_DRAFT_DOCS QUICKSTART_DRAFT_MODEL QUICKSTART_DRAFT_ENDPOINT
export QUICKSTART_DRAFT_BACKEND QUICKSTART_DRAFT_BASE_URL QUICKSTART_DRAFT_MAX_ITEMS QUICKSTART_DRAFT_VERIFY_N
export QUICKSTART_DRAFT_TIMEOUT QUICKSTART_DRAFT_MAX_TOKENS QUICKSTART_DRAFT_TEMPERATURE
export QUICKSTART_DRAFT_NUM_CTX
export QUICKSTART_DRAFT_VLLM_PORT QUICKSTART_DRAFT_VLLM_GPU_MEMORY_UTILIZATION
export QUICKSTART_DRAFT_VLLM_MAX_MODEL_LEN QUICKSTART_DRAFT_VLLM_DTYPE
export QUICKSTART_DRAFT_VLLM_QUANTIZATION QUICKSTART_DRAFT_VLLM_STARTUP_TIMEOUT
export QUICKSTART_DRAFT_EXTRACT_MAX_CHARS QUICKSTART_DRAFT_EXTRACT_CHUNK_OVERLAP
export QUICKSTART_DRAFT_CONCURRENCY
export QUICKSTART_MODEL_SELECTION QUICKSTART_ASSUME_YES QUICKSTART_PDF_MIN_CHARS
export QUICKSTART_PDF_PARSER
export MODELS_MANIFEST RAG_K SPLIT HF_HUB_OFFLINE SECURITY_CASES SECURITY_VERIFICATION_REF
export SECURITY_DATA_VERIFIED SECURITY_MODEL SECURITY_BACKEND SECURITY_BASE_URL SECURITY_MAX_MODEL_LEN
export SERVING_TIER_JSON LLB_OLLAMA_PULL_TIMEOUT_S

.DEFAULT_GOAL := help
.PHONY: help venv apt-deps test test-fast format ci gen-rag-items pdf-to-markdown validate-goldset ingest-squad ingest-uk-squad prepare-goldset-draft build-rag-store calibration-worksheet calibration-run calibration-rate calibration-score cross-check-goldset verify-sample verify-review verify-accept judge-experiment build-index validate-retrieval compare-retrieval run-eval sweep pipeline board recommend prompt-system-prepare prompt-system-review prompt-system-compare bench-security bench-agentic agentic-harness-compare composite-headline platform-matrix prep-models prep-serving-targets list-models build-vllm demo-eval mlflow detect-gpu-vram gen-serving-config quickstart-goldset quickstart-goldset-setup quickstart-goldset-rag quickstart-goldset-models quickstart-goldset-eval quickstart-goldset-security quickstart-goldset-prompt quickstart-pdf-corpus quickstart-pdf-corpus-convert quickstart-pdf-corpus-index quickstart-pdf-corpus-draft quickstart-pdf-corpus-graph quickstart-pdf-corpus-validate quickstart-pdf-corpus-review quickstart-pdf-corpus-accept quickstart-pdf-corpus-score

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-34s %s\n", $$1, $$2}'

quickstart-goldset: ## Quickstart all-in-one: committed goldset -> RAG -> model prep -> sweep -> backend matrix -> security -> prompts
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset

quickstart-goldset-setup: ## Quickstart group: venv, CUDA tier detection, serving config generation
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset-setup

quickstart-goldset-rag: ## Quickstart group: build and validate committed-goldset RAG index
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset-rag

quickstart-goldset-models: ## Quickstart group: list and prepare model candidates
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset-models

quickstart-goldset-eval: ## Quickstart group: model-family sweep and backend platform matrix
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset-eval

quickstart-goldset-security: ## Quickstart group: run model security tests as a separate benchmark tier
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset-security

quickstart-goldset-prompt: ## Quickstart group: prompt candidates; set QUICKSTART_PROMPT_ID=<id> to pin and score
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset-prompt

quickstart-pdf-corpus: ## Quickstart all-in-one: PDF corpus -> RAG -> full goldset/ontology draft -> graph -> validation
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus

quickstart-pdf-corpus-convert: ## Quickstart group: convert QUICKSTART_PDF_SOURCE PDFs to markdown/citations
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-convert

quickstart-pdf-corpus-index: ## Quickstart group: build full PDF-corpus RAG index
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-index

quickstart-pdf-corpus-draft: ## Quickstart group: select drafter and draft full unverified PDF goldset/ontology
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-draft

quickstart-pdf-corpus-graph: ## Quickstart group: build graph artifacts from the draft bundle
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-graph

quickstart-pdf-corpus-validate: ## Quickstart group: validate draft structure and retrieval metrics
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-validate

quickstart-pdf-corpus-review: ## Quickstart human gate: review verify_sample.csv interactively
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-review

quickstart-pdf-corpus-accept: ## Quickstart human gate: emit accepted ledger after review
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-accept

quickstart-pdf-corpus-score: ## Quickstart continuation: score accepted PDF corpus/goldset
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-score

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

board: ## Serve the Streamlit leaderboard (BOARD_HOST=127.0.0.1 BOARD_PORT=8501)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main board --host "$(BOARD_HOST)" --port "$(BOARD_PORT)"

recommend: ## Summarize a sweep into host-adaptive picks + chart (RECOMMEND_MIN_CASES= RECOMMEND_GPU_GB= RECOMMEND_MIN_TOK_S=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main recommend --min-cases "$(RECOMMEND_MIN_CASES)" \
		$(if $(RECOMMEND_GPU_GB),--gpu-gb $(RECOMMEND_GPU_GB),) \
		$(if $(RECOMMEND_MIN_TOK_S),--min-tokens-per-s $(RECOMMEND_MIN_TOK_S),) \
		$(if $(RECOMMEND_JSON_OUT),--json-out "$(RECOMMEND_JSON_OUT)",) \
		$(if $(filter 1 true yes,$(RECOMMEND_NO_CHART)),--no-chart,)

venv: ## Create/update .venv + extras + vLLM on CUDA hosts; VENV_INSTALL_VLLM=0 to skip
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
	@case "$(VENV_INSTALL_VLLM)" in \
	  0|false|no) echo "[venv] vLLM install skipped (VENV_INSTALL_VLLM=$(VENV_INSTALL_VLLM))" ;; \
	  1|true|yes) echo "[venv] installing vLLM binary wheels (forced)"; bash "$(PROJECT_ROOT)/scripts/build_vllm.sh" ;; \
	  auto) \
	    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then \
	      echo "[venv] CUDA host detected; installing vLLM binary wheels"; \
	      bash "$(PROJECT_ROOT)/scripts/build_vllm.sh"; \
	    else \
	      echo "[venv] vLLM install skipped (no CUDA GPU detected; set VENV_INSTALL_VLLM=1 to force)"; \
	    fi ;; \
	  *) echo "ERROR: VENV_INSTALL_VLLM must be auto, 1, or 0 (got $(VENV_INSTALL_VLLM))" >&2; exit 2 ;; \
	esac
	@bash -c 'source "$(PROJECT_ROOT)/scripts/shared/common.sh"; llb_ensure_env || true'

apt-deps: ## Install apt packages (APT_PROFILE=production|dev|all; SKIP_APT=1 to skip; APT_DRY_RUN=1 to list only)
	@SKIP_APT="$(SKIP_APT)" APT_DRY_RUN="$(APT_DRY_RUN)" bash "$(PROJECT_ROOT)/scripts/install_apt_deps.sh" "$(APT_PROFILE)"

# Two test groups (markers registered in pyproject.toml):
#   `make test`      -- FULL local suite: every test, including the `slow` ones (real Optuna
#                       sweeps, embedder/model loads, deepeval, subprocess builds).
#   `make ci` / `test-fast` -- LIGHTWEIGHT suite (`-m "not slow"`) so GitHub CI stays fast.
NOT_SLOW := -m "not slow"

# Markdown docs linted by `make lint-md` (override to narrow scope, e.g. MD_PATHS=docs/design).
MD_PATHS ?= README.md AGENTS.md CLAUDE.md GEMINI.md docs

test: ## FULL local precommit flow: pytest (incl. slow) + markdown lint (NOT run in GitHub CI)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m pytest $(PYTEST_CACHE_OPT)
	$(MAKE) lint-md

test-fast: ## Run the lightweight test suite (skips slow tests; mirrors CI)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m pytest $(PYTEST_CACHE_OPT) $(NOT_SLOW)

format: ## Format Python sources and tests with Ruff
	@test -x "$(VENV)/bin/ruff" || { echo "ERROR: ruff missing -- run 'make venv' first"; exit 1; }
	$(VENV)/bin/ruff format src tests

ci: ## Format check + lint + type check + LIGHTWEIGHT unit tests -- used by GitHub CI
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- create one + install '.[dev]' first"; exit 1; }
	$(VENV)/bin/ruff format --check src tests
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/mypy --python-version $(PYTHON_VERSION)
	$(PY) -m pytest $(PYTEST_CACHE_OPT) $(NOT_SLOW)

# Fix findings BY HAND. Do NOT run `pymarkdown fix` -- it corrupts prose on this version (AGENTS.md).
lint-md: ## Lint Markdown docs with pymarkdown (config in pyproject; MD_PATHS overrides scope)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m pymarkdown scan -r --respect-gitignore $(MD_PATHS)

gen-rag-items: ## Generate sample canonical UA RAG gold items into .data/llb/
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	bash "$(PROJECT_ROOT)/scripts/gen_rag_items.sh"

pdf-to-markdown: ## Convert PDF_DIR to markdown corpus (default DATA_DIR/quickstart-pdf-corpus; PDF_OUT_DIR=, PDF_MIN_CHARS=, PDF_PARSER=auto, PDF_REFRESH=1 reconverts unchanged PDFs)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@args=(); \
	if [ -n "$(PDF_OUT_DIR)" ]; then args+=("$(PDF_OUT_DIR)"); fi; \
	if [ -n "$(PDF_MIN_CHARS)" ]; then args+=(--min-chars "$(PDF_MIN_CHARS)"); fi; \
	if [ -n "$(PDF_PARSER)" ]; then args+=(--parser "$(PDF_PARSER)"); fi; \
	if [ -n "$(PDF_REFRESH)" ]; then args+=(--refresh); fi; \
	$(PY) -m llb.main pdf-to-markdown "$(PDF_DIR)" "$${args[@]}"

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

cross-check-goldset: ## Data gate: a SECOND frontier re-confirms grounding/support on a draft BUNDLE (CROSS_CHECK_MODEL=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<draft dir with goldset.jsonl + corpus/>"; exit 1; }
	@test -n "$(CROSS_CHECK_MODEL)" || { echo "ERROR: set CROSS_CHECK_MODEL=<second-frontier id, != the drafter>"; exit 1; }
	$(PY) -m llb.main cross-check-goldset --goldset "$(BUNDLE)/goldset.jsonl" --corpus "$(BUNDLE)/corpus" --model "$(CROSS_CHECK_MODEL)"

verify-sample: ## human verification gate: draw a stratified sample from a draft BUNDLE -> verification worksheet (VERIFY_N=, VERIFY_SEED=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<draft dir with goldset.jsonl + corpus/>"; exit 1; }
	$(PY) -m llb.goldset.verify sample --bundle "$(BUNDLE)" --out "$(VERIFY_WS)" -n $(VERIFY_N) --seed $(VERIFY_SEED)

verify-review: ## human verification gate: interactively verify the sampled items (VERIFY_WS=path, SHOW_CROSSCHECK=1 to reveal, START=N, CLEAR=1 to reset)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.goldset.verify review --worksheet "$(VERIFY_WS)" $(if $(START),--start $(START)) $(if $(SHOW_CROSSCHECK),--show-crosscheck) $(if $(CLEAR),--clear)

verify-accept: ## human verification gate: acceptance report + emit the accepted-ledger bundle (VERIFY_WS=, BUNDLE=, VERIFY_TOLERANCE=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<the draft dir the sample came from>"; exit 1; }
	$(PY) -m llb.goldset.verify accept --worksheet "$(VERIFY_WS)" --bundle "$(BUNDLE)" --tolerance $(VERIFY_TOLERANCE)

judge-experiment: ## Run fixed UA judge cases against a local OpenAI-compatible endpoint
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main judge-experiment --judge-model "$(JUDGE_MODEL)" \
		$(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",)

ingest-uk-squad: ## Development utility: GOLDSET_MODE=development|skeleton|draft (draft is robust backend prep)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@echo "[ingest-uk-squad] mode=$(GOLDSET_MODE)"; \
	case "$(GOLDSET_MODE)" in \
	  development) \
	    set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	    $(PY) -m llb.prep.ingest_squad --pinned-development-source \
	      --max-items $(GOLDSET_N) \
	      --out-name goldset_uk_development.jsonl ;; \
	  skeleton) \
	    $(PY) -m llb.prep.goldset_skeleton ;; \
	  draft) \
	    set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	    $(MAKE) --no-print-directory prepare-goldset-draft DRAFT_CORPUS="$(CORPUS)" ;; \
	  *) \
	    echo "ERROR: GOLDSET_MODE must be development, skeleton, or draft" >&2; exit 2 ;; \
	esac

prepare-goldset-draft: ## Ontology-assisted draft bundle; use DRAFT_DOC_LIMIT=1 for PDF probe
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	args=( \
	  --corpus-root "$(DRAFT_CORPUS)" \
	  --model "$(DRAFT_MODEL)" \
	  --endpoint "$(DRAFT_ENDPOINT)" \
	  --backend "$(DRAFT_BACKEND)" \
	  --max-items "$(DRAFT_MAX_ITEMS)" \
	  --extractor "$(DRAFT_EXTRACTOR)" \
	  --max-tokens "$(DRAFT_MAX_TOKENS)" \
	  --temperature "$(DRAFT_TEMPERATURE)" \
	  --timeout "$(DRAFT_TIMEOUT)" \
	  --verification-sample-size "$(DRAFT_VERIFY_N)" \
	); \
	if [ -n "$(DRAFT_BASE_URL)" ]; then args+=(--base-url "$(DRAFT_BASE_URL)"); fi; \
	if [ -n "$(DRAFT_VLLM_PORT)" ]; then args+=(--vllm-port "$(DRAFT_VLLM_PORT)"); fi; \
	if [ -n "$(DRAFT_VLLM_GPU_MEMORY_UTILIZATION)" ]; then args+=(--vllm-gpu-memory-utilization "$(DRAFT_VLLM_GPU_MEMORY_UTILIZATION)"); fi; \
	if [ -n "$(DRAFT_VLLM_MAX_MODEL_LEN)" ]; then args+=(--vllm-max-model-len "$(DRAFT_VLLM_MAX_MODEL_LEN)"); fi; \
	if [ -n "$(DRAFT_VLLM_DTYPE)" ]; then args+=(--vllm-dtype "$(DRAFT_VLLM_DTYPE)"); fi; \
	if [ -n "$(DRAFT_VLLM_QUANTIZATION)" ]; then args+=(--vllm-quantization "$(DRAFT_VLLM_QUANTIZATION)"); fi; \
	if [ -n "$(DRAFT_VLLM_STARTUP_TIMEOUT)" ]; then args+=(--vllm-startup-timeout "$(DRAFT_VLLM_STARTUP_TIMEOUT)"); fi; \
	if [ -n "$(DRAFT_DOC_LIMIT)" ]; then args+=(--doc-limit "$(DRAFT_DOC_LIMIT)"); fi; \
	if [ -n "$(DRAFT_EXTRACT_MAX_CHARS)" ]; then args+=(--extract-max-chars "$(DRAFT_EXTRACT_MAX_CHARS)"); fi; \
	if [ -n "$(DRAFT_EXTRACT_CHUNK_OVERLAP)" ]; then args+=(--extract-chunk-overlap "$(DRAFT_EXTRACT_CHUNK_OVERLAP)"); fi; \
	if [ -n "$(DRAFT_CONCURRENCY)" ]; then args+=(--concurrency "$(DRAFT_CONCURRENCY)"); fi; \
	if [ -n "$(DRAFT_OUT_DIR)" ]; then args+=(--out-dir "$(DRAFT_OUT_DIR)"); fi; \
	if [ -n "$(DRAFT_RETRIEVAL_INDEX_DIR)" ]; then args+=(--retrieval-index-dir "$(DRAFT_RETRIEVAL_INDEX_DIR)" --retrieval-k "$(DRAFT_RETRIEVAL_K)"); fi; \
	if [ "$(DRAFT_DROP_NONRETRIEVABLE_NEEDLES)" = "1" ]; then args+=(--drop-nonretrievable-needles); fi; \
	if [ "$(DRAFT_NO_THINK)" = "1" ]; then args+=(--no-think); fi; \
	if [ -n "$(DRAFT_NUM_CTX)" ]; then args+=(--num-ctx "$(DRAFT_NUM_CTX)"); fi; \
	$(PY) -m llb.main prepare-goldset-draft "$${args[@]}"

build-rag-store: ## Chunk a corpus with all strategies into DATA_DIR/llb/rag (CORPUS_DIR=...)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.rag.chunking --corpus-root "$(CORPUS_DIR)" \
		--out-dir "$(DATA_DIR)/llb/rag" --strategy all --size 800 --overlap 120

build-index: ## RAG core: chunk + embed CORPUS into the FAISS store (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main build-index --corpus-root "$(CORPUS)"

build-graph: ## GraphRAG backend: build the GraphRAG store from an ontology-assisted draft bundle (BUNDLE=...; needs ".[graph]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<prepare-goldset dir> (extraction.jsonl + corpus/)"; exit 1; }
	$(PY) -m llb.main build-graph --bundle "$(BUNDLE)"

validate-retrieval: ## RAG core: recall@k / MRR of the pinned embedding over the gold set (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main validate-retrieval --goldset "$(GOLDSET)" --k $(RAG_K)

compare-retrieval: ## GraphRAG backend: compare faiss vs both graph strategies' recall@k/MRR on the gold set
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main compare-retrieval --goldset "$(GOLDSET)" --k $(RAG_K)

run-eval: ## Run the eval; MODEL= BACKEND= GOLDSET= SPLIT= PROMPT_SYSTEM_ID= PROMPT_PACKAGE=
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main run-eval --model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --split "$(SPLIT)" \
		--limit $(LIMIT) $(if $(TELEMETRY),--telemetry) \
		$(if $(PROMPT_SYSTEM_ID),--prompt-system "$(PROMPT_SYSTEM_ID)",) \
		$(if $(PROMPT_PACKAGE),--prompt-package "$(PROMPT_PACKAGE)",) \
		$(if $(JUDGE_RHO),--judge-rho $(JUDGE_RHO) --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)"))

sweep: ## Run isolated candidate sweep (SWEEP_ID= MODELS_MANIFEST= SPLIT= GOLDSET= SWEEP_LIMIT= SWEEP_RAG_GRID=top_k=3,5,8)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main sweep --manifest "$(MODELS_MANIFEST)" --split "$(SPLIT)" \
		--goldset "$(GOLDSET)" --sweep-id "$(SWEEP_ID)" \
		--max-model-len "$(SWEEP_MAX_MODEL_LEN)" $(if $(SWEEP_OFFLINE),--offline,) \
		$(if $(SWEEP_LIMIT),--limit "$(SWEEP_LIMIT)",) \
		$(if $(SWEEP_RAG_GRID),--rag-grid "$(SWEEP_RAG_GRID)",)

pipeline: ## Select public-screen finalists, tune, and print the final board
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main pipeline --manifest "$(MODELS_MANIFEST)" --goldset "$(GOLDSET)" \
		--top-n "$(PIPELINE_TOP_N)" --trials "$(PIPELINE_TRIALS)" \
		$(if $(PIPELINE_OFFLINE),--offline,)

prompt-system-prepare: ## Generate reviewable RAG prompt-system candidates
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main prompt-system-prepare --corpus-root "$(PROMPT_SYSTEM_CORPUS)" \
		--context-window "$(PROMPT_SYSTEM_CONTEXT_WINDOW)" \
		--chunk-tokens "$(PROMPT_SYSTEM_CHUNK_TOKENS)" \
		--answer-tokens "$(PROMPT_SYSTEM_ANSWER_TOKENS)" \
		--max-passages "$(PROMPT_SYSTEM_MAX_PASSAGES)" \
		$(if $(PROMPT_SYSTEM_OUT_DIR),--out-dir "$(PROMPT_SYSTEM_OUT_DIR)",) \
		$(if $(PROMPT_SYSTEM_ROLE),--role "$(PROMPT_SYSTEM_ROLE)",) \
		$(if $(PROMPT_SYSTEM_INSTRUCTION),--instruction "$(PROMPT_SYSTEM_INSTRUCTION)",)

prompt-system-review: ## Review prompt-system candidates (PROMPT_SYSTEM_RUN_DIR= PROMPT_SYSTEM_ACTION= PROMPT_SYSTEM_ID=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(PROMPT_SYSTEM_RUN_DIR)" || { echo "ERROR: set PROMPT_SYSTEM_RUN_DIR=<run-dir>"; exit 1; }
	$(PY) -m llb.main prompt-system-review --run-dir "$(PROMPT_SYSTEM_RUN_DIR)" \
		--action "$(PROMPT_SYSTEM_ACTION)" \
		$(if $(PROMPT_SYSTEM_ID),--id "$(PROMPT_SYSTEM_ID)",) \
		$(if $(PROMPT_SYSTEM_NOTE),--note "$(PROMPT_SYSTEM_NOTE)",)

prompt-system-compare: ## Rank one model across prompt-system-tagged runs
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main prompt-system-compare --model "$(MODEL)" \
		--lane "$(PROMPT_SYSTEM_LANE)" \
		$(if $(PROMPT_SYSTEM_HARNESS),--harness "$(PROMPT_SYSTEM_HARNESS)",)

bench-security: ## Security benchmark: ASR/defense/refusal metrics for SECURITY_MODEL/SECURITY_BACKEND
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-security --cases "$(SECURITY_CASES)" \
		--model "$(SECURITY_MODEL)" --backend "$(SECURITY_BACKEND)" \
		$(if $(SECURITY_BASE_URL),--base-url "$(SECURITY_BASE_URL)",) \
		$(if $(SECURITY_MAX_MODEL_LEN),--max-model-len "$(SECURITY_MAX_MODEL_LEN)",) \
		$(if $(filter 1 true yes,$(SECURITY_DATA_VERIFIED)),--data-verified,) \
		$(if $(SECURITY_VERIFICATION_REF),--verification-ref "$(SECURITY_VERIFICATION_REF)",) \
		$(if $(JUDGE_RHO),--judge-rho "$(JUDGE_RHO)" --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",),)

bench-agentic: ## Run one agentic harness cell (AGENTIC_HARNESS=loop|langgraph|crewai)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic --tasks "$(AGENTIC_TASKS)" \
		--model "$(MODEL)" --backend "$(BACKEND)" --max-steps "$(AGENTIC_MAX_STEPS)" \
		--harness "$(AGENTIC_HARNESS)" \
		$(if $(AGENTIC_BASE_URL),--base-url "$(AGENTIC_BASE_URL)",) \
		$(if $(JUDGE_RHO),--judge-rho "$(JUDGE_RHO)" --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",),)

agentic-harness-compare: ## Run loop/langgraph/crewai agentic cells, then compare harnesses
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@for harness in $(AGENTIC_HARNESSES); do \
		$(MAKE) --no-print-directory bench-agentic AGENTIC_HARNESS="$$harness" || exit 1; \
	done
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-compare --model "$(MODEL)"

composite-headline: ## Run the verified category suite for MODEL, then require a clean bench-composite preflight
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(COMPOSITE_TEXT_ANALYSIS_BUNDLE)" || { echo "ERROR: set COMPOSITE_TEXT_ANALYSIS_BUNDLE=<verified text-analysis bundle>"; exit 1; }
	@test -n "$(COMPOSITE_TEXT_ANALYSIS_VERIFICATION_REF)" || { echo "ERROR: set COMPOSITE_TEXT_ANALYSIS_VERIFICATION_REF or COMPOSITE_VERIFICATION_REF"; exit 1; }
	@test -n "$(COMPOSITE_SUMMARIZATION_VERIFICATION_REF)" || { echo "ERROR: set COMPOSITE_SUMMARIZATION_VERIFICATION_REF or COMPOSITE_VERIFICATION_REF"; exit 1; }
	@test -n "$(COMPOSITE_STRUCTURED_VERIFICATION_REF)" || { echo "ERROR: set COMPOSITE_STRUCTURED_VERIFICATION_REF or COMPOSITE_VERIFICATION_REF"; exit 1; }
	@test -n "$(COMPOSITE_SECURITY_VERIFICATION_REF)" || { echo "ERROR: set COMPOSITE_SECURITY_VERIFICATION_REF or COMPOSITE_VERIFICATION_REF"; exit 1; }
	@test -n "$(COMPOSITE_AGENTIC_VERIFICATION_REF)" || { echo "ERROR: set COMPOSITE_AGENTIC_VERIFICATION_REF or COMPOSITE_VERIFICATION_REF"; exit 1; }
	@test -n "$(COMPOSITE_TOOLING_VERIFICATION_REF)" || { echo "ERROR: set COMPOSITE_TOOLING_VERIFICATION_REF or COMPOSITE_VERIFICATION_REF"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-text-analysis --bundle "$(COMPOSITE_TEXT_ANALYSIS_BUNDLE)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		$(if $(COMPOSITE_BASE_URL),--base-url "$(COMPOSITE_BASE_URL)",) \
		$(if $(COMPOSITE_REAL_CORPUS),--real-corpus,) \
		$(if $(JUDGE_RHO),--judge-rho "$(JUDGE_RHO)" --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",),) \
		--data-verified --verification-ref "$(COMPOSITE_TEXT_ANALYSIS_VERIFICATION_REF)" && \
	$(PY) -m llb.main bench-summarization --cases "$(COMPOSITE_SUMMARIZATION_CASES)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		$(if $(COMPOSITE_BASE_URL),--base-url "$(COMPOSITE_BASE_URL)",) \
		$(if $(JUDGE_RHO),--judge-rho "$(JUDGE_RHO)" --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",),) \
		--data-verified --verification-ref "$(COMPOSITE_SUMMARIZATION_VERIFICATION_REF)" && \
	$(PY) -m llb.main bench-structured --cases "$(COMPOSITE_STRUCTURED_CASES)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		$(if $(COMPOSITE_BASE_URL),--base-url "$(COMPOSITE_BASE_URL)",) \
		--data-verified --verification-ref "$(COMPOSITE_STRUCTURED_VERIFICATION_REF)" && \
	$(PY) -m llb.main bench-security --cases "$(COMPOSITE_SECURITY_CASES)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		$(if $(COMPOSITE_BASE_URL),--base-url "$(COMPOSITE_BASE_URL)",) \
		$(if $(JUDGE_RHO),--judge-rho "$(JUDGE_RHO)" --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",),) \
		--data-verified --verification-ref "$(COMPOSITE_SECURITY_VERIFICATION_REF)" && \
	$(PY) -m llb.main bench-agentic --tasks "$(COMPOSITE_AGENTIC_TASKS)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		$(if $(COMPOSITE_BASE_URL),--base-url "$(COMPOSITE_BASE_URL)",) \
		$(if $(JUDGE_RHO),--judge-rho "$(JUDGE_RHO)" --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",),) \
		--data-verified --verification-ref "$(COMPOSITE_AGENTIC_VERIFICATION_REF)" && \
	$(PY) -m llb.main bench-tooling --catalog "$(COMPOSITE_TOOLING_CATALOG)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		$(if $(COMPOSITE_BASE_URL),--base-url "$(COMPOSITE_BASE_URL)",) \
		--data-verified --verification-ref "$(COMPOSITE_TOOLING_VERIFICATION_REF)" && \
	$(PY) -m llb.main bench-composite

platform-matrix: ## Run same logical model base across Ollama, vLLM, and llama.cpp with telemetry
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	HF_HUB_OFFLINE="$(HF_HUB_OFFLINE)" $(MAKE) --no-print-directory build-index
	@set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	wants_backend() { case " $(PLATFORM_MATRIX_BACKENDS) " in *" $$1 "*) return 0 ;; *) return 1 ;; esac; }; \
	record_failure() { failed=1; echo "[platform-matrix] failed $$1 (continuing; set PLATFORM_MATRIX_STRICT=1 to fail fast)"; }; \
	ran=0; failed=0; \
	if wants_backend ollama; then \
	  echo "[platform-matrix] run ollama model=$(PLATFORM_MATRIX_OLLAMA_MODEL)"; \
	  if $(PY) -m llb.main run-eval --model "$(PLATFORM_MATRIX_OLLAMA_MODEL)" --backend ollama \
	    --goldset "$(PLATFORM_MATRIX_GOLDSET)" --split "$(PLATFORM_MATRIX_SPLIT)" --limit "$(PLATFORM_MATRIX_LIMIT)" \
	    --telemetry; then ran=$$((ran + 1)); else record_failure ollama; fi; \
	fi; \
	if wants_backend vllm; then \
	  if [ -x "$(VENV)/bin/vllm" ] || command -v vllm >/dev/null 2>&1; then \
	    echo "[platform-matrix] run vllm model=$(PLATFORM_MATRIX_VLLM_MODEL)"; \
	    if $(PY) -m llb.main run-eval --model "$(PLATFORM_MATRIX_VLLM_MODEL)" --backend vllm \
	      --goldset "$(PLATFORM_MATRIX_GOLDSET)" --split "$(PLATFORM_MATRIX_SPLIT)" --limit "$(PLATFORM_MATRIX_LIMIT)" \
	      --telemetry --max-model-len "$(PLATFORM_MATRIX_MAX_MODEL_LEN)" \
	      --gpu-memory-utilization "$(PLATFORM_MATRIX_GPU_MEMORY_UTILIZATION)" --evict; then ran=$$((ran + 1)); else record_failure vllm; fi; \
	  else \
	    echo "[platform-matrix] skipped vllm: vllm executable not found (run make build-vllm)"; \
	    [ "$(PLATFORM_MATRIX_STRICT)" = "1" ] && failed=1; \
	  fi; \
	fi; \
	if wants_backend llamacpp; then \
	  llama_bin="$$DATA_DIR/llb/llamacpp/build/bin/llama-server"; \
	  if [ -x "$$llama_bin" ] || command -v llama-server >/dev/null 2>&1; then \
	    echo "[platform-matrix] run llamacpp model=$(PLATFORM_MATRIX_LLAMACPP_MODEL)"; \
	    if $(PY) -m llb.main run-eval --model "$(PLATFORM_MATRIX_LLAMACPP_MODEL)" --backend llamacpp \
	      --goldset "$(PLATFORM_MATRIX_GOLDSET)" --split "$(PLATFORM_MATRIX_SPLIT)" --limit "$(PLATFORM_MATRIX_LIMIT)" \
	      --telemetry --max-model-len "$(PLATFORM_MATRIX_MAX_MODEL_LEN)" \
	      --gpu-layers "$(PLATFORM_MATRIX_LLAMACPP_GPU_LAYERS)"; then ran=$$((ran + 1)); else record_failure llamacpp; fi; \
	  else \
	    echo "[platform-matrix] skipped llamacpp: llama-server not found (run make build-llamacpp)"; \
	    [ "$(PLATFORM_MATRIX_STRICT)" = "1" ] && failed=1; \
	  fi; \
	fi; \
	if [ "$$ran" -eq 0 ]; then echo "ERROR: platform-matrix produced no successful backend rows" >&2; exit 1; fi; \
	if [ "$(PLATFORM_MATRIX_STRICT)" = "1" ] && [ "$$failed" -ne 0 ]; then exit 1; fi; \
	echo "[platform-matrix] successful backend rows: $$ran"

build-vllm: ## Install prebuilt vLLM via uv; VLLM_SOURCE_DIR= builds/caches one checkout wheel
	bash "$(PROJECT_ROOT)/scripts/build_vllm.sh"

build-llamacpp: ## Build CUDA llama-server for the llama.cpp launcher; CUDA_ARCH=/LLAMACPP_REF= override
	bash "$(PROJECT_ROOT)/scripts/build_llamacpp.sh"

prep-models: ## Detect GPU, pull Ollama tags + cache vLLM HF weights (MODELS_MANIFEST=, PREP_BACKEND=, gated needs HF_TOKEN)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main prep-models --manifest "$(MODELS_MANIFEST)" --backend "$(PREP_BACKEND)"

prep-serving-targets: ## Pull/cache models referenced by generated serving tier.json (SERVING_TIER_JSON=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(SERVING_TIER_JSON)" || { echo "ERROR: set SERVING_TIER_JSON=<llb/serving/gpu-*/tier.json>"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main prep-serving-targets --tier-json "$(SERVING_TIER_JSON)" --backend "$(PREP_BACKEND)"

list-models: ## List which candidate models can run here (GPU+RAM, KV-cache-aware); CONTEXT= to target a context
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main list-models --manifest "$(MODELS_MANIFEST)" $(if $(CONTEXT),--context $(CONTEXT),)

detect-gpu-vram: ## Print supported GPU VRAM tier (12/16/24/32 GiB) from nvidia-smi
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main detect-gpu-vram

gen-serving-config: ## Emit serve + run-eval artifacts under .data/llb/serving/; GPU_GB=12|16|24|32 overrides detect
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main gen-serving-config $(if $(GPU_GB),--gpu-gb $(GPU_GB),)
