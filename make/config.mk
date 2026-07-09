# Shared variables and exported environment for loc-lm-bench make targets.
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
# External prompt-02 SQuAD draft -> curated canonical goldset + RAG index.
SQUAD_DRAFT_INPUT_DIR ?=
SQUAD_DRAFT_INPUTS ?=
SQUAD_DRAFT_CORPUS ?=
SQUAD_DRAFT_OUT_DIR ?= $(DATA_DIR)/external-squad-rag
SQUAD_DRAFT_CURATED ?= $(SQUAD_DRAFT_OUT_DIR)/goldset.curated.json
SQUAD_DRAFT_GOLDSET_NAME ?= squad_uk.jsonl
SQUAD_DRAFT_SEMANTIC ?= 1
# curate-drafts: merge/dedup/filter externally drafted artifacts before import.
CURATE_KIND ?= squad
CURATE_INPUTS ?=
CURATE_OUT ?=
CURATE_CORPUS ?=
CURATE_DEDUP_AGAINST ?=
CURATE_SEMANTIC ?= 1
CORPUS_DIR ?= $(PROJECT_ROOT)/samples/corpus
PDF_DIR ?= $(DATA_DIR)/quickstart-pdf-corpus
PDF_OUT_DIR ?=
PDF_MIN_CHARS ?=
PDF_PARSER ?= auto
PDF_REFRESH ?=
# Unified mixed txt/md/pdf ingest (llb ingest-corpus).
CORPUS_ROOT ?= $(CORPUS_DIR)
CORPUS_OUT_DIR ?=
CORPUS_MIN_CHARS ?= 500
CORPUS_PARSER ?= auto
CORPUS_REFRESH ?=
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
DRAFT_VLLM_CPU_OFFLOAD_GB ?=
DRAFT_VLLM_KV_OFFLOADING_SIZE_GB ?=
DRAFT_VLLM_DTYPE ?= auto
DRAFT_VLLM_QUANTIZATION ?=
DRAFT_VLLM_STARTUP_TIMEOUT ?= 600
DRAFT_EXTRACTOR ?= llm
DRAFT_OUT_DIR ?=
DRAFT_RESUME ?=
DRAFT_VERIFY_N ?= 0
DRAFT_RETRIEVAL_INDEX_DIR ?=
DRAFT_RETRIEVAL_K ?= $(RAG_K)
DRAFT_DROP_NONRETRIEVABLE_NEEDLES ?= 0
DRAFT_REQUIRE_PASSED_GATES ?= 0
# yield-max knobs: per-stratum coverage target, multi-hop chain drafting, prior-bundle dedup.
DRAFT_COVERAGE_TARGET ?=
DRAFT_MULTI_HOP ?= 0
DRAFT_MULTI_HOP_MAX_PATHS ?=
DRAFT_DEDUP_AGAINST ?=
DRAFT_GRAPH_DIR ?=
COVERAGE_JSON ?=
COVERAGE_TEXT ?=

# RAG/vLLM eval knobs (override on the command line). SMOKE_MODEL is intentionally small
# and should be used for connectivity checks only, not leaderboard or extended tests.
SMOKE_MODEL ?= llama3.2:3b
MODEL ?= $(SMOKE_MODEL)
BACKEND ?= ollama
SPLIT ?= final
LIMIT ?= 20
RESUME ?=
RAG_K ?= 10
# Lost-in-the-middle probe (rerank-context-order): fixed context size for probe-context-position.
PROBE_K ?= 5
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
EXTERNAL_RAG_ANSWERS ?=
EXTERNAL_RAG_CSV ?=
EXTERNAL_RAG_REPORT ?=
EXTERNAL_RAG_ANSWER_FIELD ?=
EXTERNAL_RAG_SOURCES_FIELD ?=
EXTERNAL_RAG_ERROR_FIELD ?=
EXTERNAL_RAG_SOURCE_LIMIT ?= 3
EXTERNAL_RAG_LABEL ?=
EXTERNAL_RAG_KEEP_SOURCE_FOOTER ?=
EXTERNAL_RAG_START ?=
EXTERNAL_RAG_CLEAR ?=
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
FINETUNE_CAMPAIGN_MODELS ?= $(MODEL)
FINETUNE_CAMPAIGN_ROUNDS ?= 1
FINETUNE_CAMPAIGN_LIMIT ?=
FINETUNE_CAMPAIGN_OUT ?=
FINETUNE_CAMPAIGN_RESUME ?=
FINETUNE_CAMPAIGN_MANIFEST ?= $(MODELS_MANIFEST)
# Judge knobs (judge calibration gate). JUDGE_MODEL is the model id exposed by a LOCAL OpenAI-compatible endpoint
# (no data egress + reproducible; bias documented in current.md); JUDGE_BASE_URL points at it.
# Default = the Ollama gemma3:27b judge on :11434 (the default BACKEND=ollama candidate runs there
# too). Alternatives by GPU tier (override JUDGE_MODEL + JUDGE_BASE_URL):
#   12 GB GPU: google/gemma-4-12B-it-qat-w4a16-ct              (vLLM + CPU/KV offload on :8000)
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
# reject rate. Full workflow: docs/guides/data-prep/goldset-from-scratch.md.
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
QUICKSTART_DRAFT_VLLM_CPU_OFFLOAD_GB ?=
QUICKSTART_DRAFT_VLLM_KV_OFFLOADING_SIZE_GB ?=
QUICKSTART_DRAFT_VLLM_DTYPE ?= auto
QUICKSTART_DRAFT_VLLM_QUANTIZATION ?=
QUICKSTART_DRAFT_VLLM_STARTUP_TIMEOUT ?= 600
QUICKSTART_DRAFT_EXTRACT_MAX_CHARS ?=
QUICKSTART_DRAFT_EXTRACT_CHUNK_OVERLAP ?=
QUICKSTART_DRAFT_CONCURRENCY ?=
QUICKSTART_MODEL_SELECTION ?= gemma4
QUICKSTART_ASSUME_YES ?= 0
QUICKSTART_PDF_MIN_CHARS ?= 500
QUICKSTART_PDF_PARSER ?= auto
# Mixed-corpus quickstart (txt/md/pdf via ingest-corpus). Shares the draft/model knobs above.
QUICKSTART_CORPUS_SRC ?= $(QUICKSTART_ROOT)/quickstart-corpus
QUICKSTART_CORPUS_MD ?= $(QUICKSTART_ROOT)/quickstart-corpus-md
QUICKSTART_CORPUS_RAG_DATA ?= $(QUICKSTART_ROOT)/quickstart-corpus-rag
QUICKSTART_CORPUS_DRAFT ?= $(QUICKSTART_ROOT)/quickstart-corpus-draft
QUICKSTART_CORPUS_GRAPH_DATA ?= $(QUICKSTART_ROOT)/quickstart-corpus-graph
QUICKSTART_CORPUS_MIN_CHARS ?= 500
QUICKSTART_CORPUS_PARSER ?= auto
QUICKSTART_CORPUS_RESUME ?=
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
export QUICKSTART_DRAFT_VLLM_MAX_MODEL_LEN QUICKSTART_DRAFT_VLLM_CPU_OFFLOAD_GB
export QUICKSTART_DRAFT_VLLM_KV_OFFLOADING_SIZE_GB QUICKSTART_DRAFT_VLLM_DTYPE
export QUICKSTART_DRAFT_VLLM_QUANTIZATION QUICKSTART_DRAFT_VLLM_STARTUP_TIMEOUT
export QUICKSTART_DRAFT_EXTRACT_MAX_CHARS QUICKSTART_DRAFT_EXTRACT_CHUNK_OVERLAP
export QUICKSTART_DRAFT_CONCURRENCY
export QUICKSTART_MODEL_SELECTION QUICKSTART_ASSUME_YES QUICKSTART_PDF_MIN_CHARS
export QUICKSTART_PDF_PARSER
export QUICKSTART_CORPUS_SRC QUICKSTART_CORPUS_MD QUICKSTART_CORPUS_RAG_DATA
export QUICKSTART_CORPUS_DRAFT QUICKSTART_CORPUS_GRAPH_DATA QUICKSTART_CORPUS_MIN_CHARS
export QUICKSTART_CORPUS_PARSER QUICKSTART_CORPUS_RESUME
export MODELS_MANIFEST RAG_K SPLIT HF_HUB_OFFLINE SECURITY_CASES SECURITY_VERIFICATION_REF
export SECURITY_DATA_VERIFIED SECURITY_MODEL SECURITY_BACKEND SECURITY_BASE_URL SECURITY_MAX_MODEL_LEN
export SERVING_TIER_JSON LLB_OLLAMA_PULL_TIMEOUT_S
