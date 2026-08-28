# Shared variables and exported environment for loc-lm-bench make targets.
# Tool caches live under one $DATA_DIR/cache/<tool> tree (not the project root), so the root stays
# clean and `rm -rf $DATA_DIR` clears every temporary artifact in one shot. These env vars are the
# ONLY place the cache locations are set for make-driven runs -- pyproject.toml deliberately holds
# no static cache path, since a config file cannot read .env and would not follow a custom
# DATA_DIR. `llb_export_tool_caches` in scripts/shared/common.sh exports the same values for
# shell-driven runs (Make cannot source that file). deepeval reads DEEPEVAL_CACHE_FOLDER (its
# `.deepeval` keystore) and DEEPEVAL_RESULTS_FOLDER.
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
EXTRAS ?= rag,rag-chroma,rag-qdrant,eval,graph,linkage,ontology,track,board,viz,prep,telemetry,goldset,cutoff,dev
VENV_INSTALL_VLLM ?= auto
# `make venv` installs with `uv sync`, so the local venv holds exactly the uv.lock versions GitHub
# CI installs (.github/workflows/ci.yml) -- an unpinned `uv pip install` on either side is how a
# lint/type error reaches CI that no local `make ci` can reproduce. uv.lock is refreshed
# automatically when pyproject.toml dependencies change; COMMIT the result, since CI syncs
# `--locked` and fails on a stale lock. VENV_LOCKED=1 makes `make venv` fail the same way instead
# of relocking.
VENV_LOCKED ?=
# EXTRAS is a comma list (pip syntax); uv sync takes one --extra per group.
UV_SYNC_EXTRAS = $(foreach extra,$(subst $(comma), ,$(EXTRAS)),--extra $(extra))
UV_SYNC_LOCK_FLAG = $(if $(filter 1 true yes,$(VENV_LOCKED)),--locked,)

# Stable human-reviewed development fixture. Runtime imports adopt matching reviewed ids.
PUBLISHED_GOLDSET_ROOT := $(PROJECT_ROOT)/samples/goldsets/ua_squad_postedited_v1
GOLDSET ?= $(PUBLISHED_GOLDSET_ROOT)/goldset.jsonl
CORPUS ?= $(PUBLISHED_GOLDSET_ROOT)/corpus
SQUAD_JSON ?= samples/data-prep/squad_uk_fixture.json
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
DRAFT_FRONTIER_MODEL ?=
DRAFT_FRONTIER_STAGE ?= both
DRAFT_LOCAL_MODEL ?=
DRAFT_EGRESS_CONSENT ?= 0
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
DRAFT_MAX_USD ?=
DRAFT_MAX_CALLS ?= 100
DRAFT_NO_THINK ?= 1
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
DRAFT_VERIFY_N ?=
DRAFT_VERIFY_DERIVE ?= 0
DRAFT_VERIFY_CONFIDENCE ?= 0.95
DRAFT_VERIFY_PRECISION ?= 0.10
DRAFT_RETRIEVAL_INDEX_DIR ?=
DRAFT_RETRIEVAL_K ?= $(RAG_K)
DRAFT_DROP_NONRETRIEVABLE_NEEDLES ?= 0
DRAFT_REQUIRE_PASSED_GATES ?= 0
# yield-max knobs: per-stratum coverage target, multi-hop drafting, chain fixtures,
# and prior-bundle dedup.
DRAFT_COVERAGE_TARGET ?=
DRAFT_MULTI_HOP ?= 0
DRAFT_MULTI_HOP_ONLY ?= 0
DRAFT_CHAINS ?= 0
DRAFT_MULTI_HOP_MAX_PATHS ?=
DRAFT_MULTI_HOP_BRIDGE_FILL ?= 0
DRAFT_MULTI_HOP_PATH_STRATIFIED ?= 0
DRAFT_MULTI_HOP_RELATION_PAIR_TARGET ?= 1
DRAFT_MULTI_HOP_DOCUMENT_MODE_TARGET ?= 1
DRAFT_MULTI_HOP_SOURCE_DOCUMENT_TARGET ?= 1
DRAFT_REUSE_EXTRACTION_BUNDLE ?=
DRAFT_CARRY_FORWARD_MULTI_HOP ?= 0
DRAFT_DEDUP_AGAINST ?=
DRAFT_DEDUP_LINKAGE_SHADOW ?= 0
DRAFT_GRAPH_DIR ?=
MULTIHOP_DRAFT_PRIOR_BUNDLE ?=
MULTIHOP_DRAFT_DEDUP_AGAINST ?= $(MULTIHOP_DRAFT_PRIOR_BUNDLE)
MULTIHOP_DRAFT_OUT ?=
MULTIHOP_DRAFT_MODEL ?= $(DRAFT_MODEL)
MULTIHOP_DRAFT_MAX_PATHS ?= 120
MULTIHOP_DRAFT_PATH_STRATIFIED ?= 1
MULTIHOP_DRAFT_RELATION_PAIR_TARGET ?= 1
MULTIHOP_DRAFT_DOCUMENT_MODE_TARGET ?= 1
MULTIHOP_DRAFT_SOURCE_DOCUMENT_TARGET ?= 1
MULTIHOP_DRAFT_MIN_HEADROOM_FRACTION ?= 0.15
COVERAGE_JSON ?=
COVERAGE_TEXT ?=
# Exact shared-seed local/frontier drafting comparison.
DRAFT_COMPARE_CORPUS ?= $(DRAFT_CORPUS)
DRAFT_COMPARE_SEEDS ?= 20
DRAFT_COMPARE_FRONTIER_MODEL ?=
DRAFT_COMPARE_LOCAL_MODEL ?= $(DRAFT_MODEL)
DRAFT_COMPARE_LOCAL_BACKEND ?= $(DRAFT_BACKEND)
DRAFT_COMPARE_LOCAL_BASE_URL ?= $(DRAFT_BASE_URL)
DRAFT_COMPARE_MAX_USD ?=
DRAFT_COMPARE_MAX_CALLS ?= 100
DRAFT_COMPARE_OUT_DIR ?=
DRAFT_COMPARE_LOCAL_VERIFICATION ?=
DRAFT_COMPARE_FRONTIER_VERIFICATION ?=
# Bounded real-provider probe over the committed two-document synthetic UA corpus.
FRONTIER_UA_PROBE_CORPUS := $(PROJECT_ROOT)/samples/text_analysis_bundle_uk/corpus
FRONTIER_UA_PROBE_SEEDS ?= 12
FRONTIER_UA_PROBE_LOCAL_MODEL ?= $(DRAFT_COMPARE_LOCAL_MODEL)
FRONTIER_UA_PROBE_FRONTIER_MODEL ?=
FRONTIER_UA_PROBE_MAX_USD ?=
FRONTIER_UA_PROBE_MAX_CALLS ?= $(FRONTIER_UA_PROBE_SEEDS)
FRONTIER_UA_PROBE_OUT_DIR ?=
# Sequential local Qwen/Gemma comparison; blank models select from the detected GPU tier.
LOCAL_DRAFT_COMPARE_CORPUS := $(PROJECT_ROOT)/samples/text_analysis_bundle_uk/corpus
LOCAL_DRAFT_COMPARE_SEEDS ?= 12
LOCAL_DRAFT_COMPARE_BASELINE_MODEL ?=
LOCAL_DRAFT_COMPARE_PROBE_MODEL ?=
LOCAL_DRAFT_COMPARE_OUT_DIR ?=
COMPARE_ANALYZE_JSON ?=
COMPARE_REQUIRE_GATES ?=

# RAG/vLLM eval knobs (override on the command line). SMOKE_MODEL is intentionally small
# and should be used for connectivity checks only, not leaderboard or extended tests.
SMOKE_MODEL ?= llama3.2:3b
MODEL ?= $(SMOKE_MODEL)
BACKEND ?= ollama
# Bounded, resumable open-model downloads.
MODEL_DOWNLOAD_ID ?=
MODEL_DOWNLOAD_TARGET ?=
MODEL_DOWNLOAD_PROVIDER ?= huggingface
MODEL_DOWNLOAD_REVISION ?=
MODEL_DOWNLOAD_CHUNK_MIB ?= 64
MODEL_DOWNLOAD_SESSION_GIB ?= 64
MODEL_DOWNLOAD_MAX_MIBPS ?=
MODEL_DOWNLOAD_BANDWIDTH_FRACTION ?= 0.8
MODEL_DOWNLOAD_TIMEOUT_SECONDS ?= 60
MODEL_DOWNLOAD_RETRIES ?= 5
MODEL_DOWNLOAD_MAX_RATE_WAIT_SECONDS ?= 900
MODEL_DOWNLOAD_MIN_FREE_GIB ?= 1
MODEL_DOWNLOAD_MIN_FREE_PERCENT ?= 5
MODEL_DOWNLOAD_VERIFY_COMPLETED ?=
MODEL_DOWNLOAD_VERIFY_ONLY ?=
MODEL_DOWNLOAD_DRY_RUN ?=
SPLIT ?= final
LIMIT ?= 20
RESUME ?=
RAG_K ?= 10
# Graph-vector fusion evidence: fixed graph shares, the routed row's non-zero graph share, swept
# per-lane candidate depths ('k' == the scored cutoff), swept span-identity policies ('exact'
# and/or 'overlap'), swept merge thresholds for a folding policy ('1.0' == containment only), and
# the question type the verdict is decided on (compare-graph-fusion).
# Empty knobs fall back to the command defaults.
GRAPH_WEIGHTS ?=
ROUTED_GRAPH_WEIGHT ?=
GRAPH_FUSION_CANDIDATES ?=
GRAPH_FUSION_SPAN_IDENTITY ?=
GRAPH_FUSION_SPAN_MERGE_RATIO ?=
GRAPH_STRATEGIES ?=
FUSION_FOCUS_SLICE ?=
FUSION_BOOTSTRAP_RESAMPLES ?=
FUSION_HIDE_ROUTING_SIDECAR ?=
FUSION_HEURISTIC_LONG_QUESTION_WORDS ?=
FUSION_HEURISTIC_MIN_LINKED_ENTITIES ?=
# Per-hop multi-hop retrievability probe (probe-multihop-hops): the budget grid the all-spans@k
# curve is read at (the SMALLEST is the operating budget the diagnosis is stated against), how
# deep a labeled span is searched for before a query counts as unable to reach it, and the lane
# to probe when it is not the one the config names. Empty knobs fall back to the command defaults.
HOP_PROBE_BUDGETS ?=
HOP_PROBE_DEPTH ?=
HOP_PROBE_BACKEND ?=
HOP_PROBE_STRATEGY ?=
HOP_PROBE_OUT_DIR ?=
# Query-prep inputs are shared by validate-retrieval and the paired per-hop probe.
QUERY_PREP ?=
QUERY_PREP_MODEL ?=
QUERY_PREP_BACKEND ?=
# Sidecar-free fusion-router calibration: tune these policy grids on the named tuning split, then
# evaluate only the frozen policy on the held-out final split.
ROUTING_LONG_WORD_GRID ?= 10,12,14,16,18,20
ROUTING_ENTITY_GRID ?= 0,1,2
ROUTING_TUNING_SPLIT ?= tuning
ROUTING_FINAL_SPLIT ?= final
ROUTING_GRAPH_STRATEGY ?= global_community
ROUTING_GRAPH_WEIGHT ?= 0.3
ROUTING_CANDIDATES ?= 50
ROUTING_SPAN_IDENTITY ?= overlap
ROUTING_OUT_DIR ?=
# End-to-end answer quality of two retrieval lanes (compare-answer-quality): the scored lanes
# (compare-graph-fusion row labels; the FIRST is the baseline), the sweep comparison.json whose
# verdict names them instead, and the full split by default.
ANSWER_QUALITY_LANES ?=
FUSION_COMPARISON ?=
ANSWER_QUALITY_LIMIT ?=
ANSWER_QUALITY_OUT_DIR ?=
# A RECORDED answer-quality comparison.json to re-render under the current report from the run
# bundles it named, with no model call. Every other variable above is ignored when it is set.
ANSWER_QUALITY_BUNDLES ?=
# Retrieval budgets every lane is scored at, smallest first (e.g. 10,50). Empty scores the
# config's own top_k once; a second budget adds the reading that says whether a budget-driven
# coverage gain converts into answers, and what the extra context costs.
ANSWER_QUALITY_BUDGETS ?=
# Set to 1 to twin every scored lane with a `<lane>+headers` copy whose prompt restores a table
# chunk's recorded header row. The twins retrieve identically, so the delta is answer-quality only;
# it needs a store built with CHUNK_STRATEGY=table to fire at all.
ANSWER_QUALITY_RESTORE_HEADERS ?=
# Set to 1 only to score a drafted (not human-accepted) ledger; artifacts record the grounding.
INCLUDE_DRAFTED ?=
# Embedder adoption bar (compare-embedder-adoption): the two encoders, each with the data root
# whose llb/rag store was built with it, and the top_k x reranker grid the answers are scored on.
# The default grid is the shipped budget plus a small one, with the pinned cross-encoder off and on
# -- the two knobs that decide whether first-hit RANK binds at all.
# EMBED_BASELINE is shared with compare-embeddings, where it is the incumbent every candidate is
# paired against; the value below is the shipped `RunConfig.embedding_model`.
EMBED_BASELINE ?= intfloat/multilingual-e5-base
EMBED_CANDIDATE ?= BAAI/bge-m3
EMBED_BASELINE_DATA_DIR ?=
EMBED_CANDIDATE_DATA_DIR ?=
# Embedder fine-tuning (finetune-embedder): the encoder adapted to the operator's own corpus, and
# the knobs of that training. FT_EMBED_BASE defaults to EMBED_BASELINE, the shipped RunConfig
# embedding_model -- the incumbent a tuned row has to beat in the bake-off's paired verdict to be
# worth adopting at all. FT_EMBED_BATCH is not only a speed knob: the contrastive objective draws
# its in-batch negatives from the batch, so a smaller value is a weaker objective.
FT_EMBED_BASE ?= $(EMBED_BASELINE)
FT_EMBED_OUT ?=
FT_EMBED_SEED ?=
FT_EMBED_NEGATIVES ?=
FT_EMBED_MAX_POSITIVES ?=
FT_EMBED_EPOCHS ?=
FT_EMBED_BATCH ?=
FT_EMBED_MINI_BATCH ?=
FT_EMBED_LR ?=
# Encoder throughput decomposition (compare-embeddings --encoder-throughput): cold load vs
# first-pass compile+encode vs warm steady encode. EMBED_ENCODER_THROUGHPUT=1 enables it;
# EMBED_ENCODER_COMPARE_CPU=1 also profiles CPU beside CUDA (only the literal 1 enables it;
# EMBED_ENCODER_COMPARE_CPU=0 is off, matching other Make opt-in flags).
EMBED_ENCODER_THROUGHPUT ?=
EMBED_ENCODER_PRECISION ?= 0.05
EMBED_ENCODER_MIN_WARM ?= 3
EMBED_ENCODER_MAX_WARM ?= 10
EMBED_ENCODER_MAX_WARM_SECONDS ?= 180
EMBED_ENCODER_COMPARE_CPU ?=
# Opt into bake-off candidates that ship their own modelling code (trust_remote_code), e.g.
# Alibaba-NLP/gte-multilingual-base and jinaai/jina-embeddings-v3. Only the literal 1 enables it;
# without it those roster rows are SKIPPED and the reason is recorded in the report.
EMBED_ALLOW_REMOTE_CODE ?=
# Reranker bake-off (compare-rerankers): the incumbent cross-encoder every candidate is paired
# against, the roster (empty = the default UA candidate set), and the cost side of the choice.
# RERANK_GENERATOR_VRAM_MB declares how much VRAM the generator holds while serving; without it the
# footprints are still measured and the fit gate does not run. Only the literal 1 enables
# RERANK_ALLOW_REMOTE_CODE, without which the jina / gte candidates are SKIPPED and recorded.
RERANK_BASELINE ?= BAAI/bge-reranker-v2-m3
RERANK_MODELS ?=
RERANK_ADOPTION_BARS ?=
RERANK_ALLOW_REMOTE_CODE ?=
RERANK_BATCH_SIZE ?=
RERANK_DTYPE ?=
RERANK_GENERATOR_VRAM_MB ?=
RERANK_RESAMPLES ?=
RERANK_CONFIDENCE ?=
COMPARE_RERANKERS_OUT ?=
ADOPTION_TOP_KS ?= 10,3
ADOPTION_RERANKERS ?= off,on
ADOPTION_LIMIT ?=
ADOPTION_OUT_DIR ?=
# Cross-model reading (compare-adoption-models): two finished sweep comparison.json files (or their
# dirs) and where to write the per-cell agreement report.
ADOPTION_REPORT_A ?=
ADOPTION_REPORT_B ?=
ADOPTION_CROSS_OUT ?=
# Roster reading (compare-adoption-roster): three or more sweep dirs, the DECLARED per-model
# property file the separation test reads, and the cell whose answer gain it explains.
ADOPTION_REPORTS ?=
ADOPTION_PROFILES ?=
ADOPTION_FOCUS_CELL ?=
ADOPTION_ROSTER_OUT ?=
# Screen cost study (compare-adoption-screen): the item counts to measure, how many subsamples per
# count, and the agreement a count must reach before it is called a usable screen.
ADOPTION_SCREEN_SIZES ?=
ADOPTION_SCREEN_DRAWS ?=
ADOPTION_SCREEN_TARGET ?=
ADOPTION_SCREEN_OUT ?=
# Query robustness probe: full split by default, bounded answers, deterministic character noise.
QUERY_ROBUSTNESS_LIMIT ?=
QUERY_ROBUSTNESS_TYPO_RATE ?= 0.08
QUERY_ROBUSTNESS_MAX_TOKENS ?= 96
QUERY_ROBUSTNESS_CLASSES ?=
# Restoration-constraint sweep grid (restoration-constraint-threshold-sweep). One factor at a
# time by default; SWEEP_FULL_GRID=1 measures the whole product instead.
SWEEP_SURFACE_DISTANCES ?= 0,1
SWEEP_SHORT_CUTOFFS ?= 3,4,5
SWEEP_RANK_ORDERS ?= morphology,context
SWEEP_FULL_GRID ?=
LANGUAGE_FIXTURE ?=
# Dense-lane casing (normalize-casefold-dense-lane-cost): set to 1 to keep the raw question's
# capitalization on the case-sensitive dense encoder; the lexical lane stays casefolded.
QUERY_PREP_DENSE_CASE ?=
# Lost-in-the-middle probe (rerank-context-order): fixed context size for probe-context-position.
PROBE_K ?= 5
MODELS_MANIFEST ?= $(PROJECT_ROOT)/samples/configs/models_uk.yaml
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
PROMPT_SYSTEM_ONTOLOGY_BUNDLE ?=
PROMPT_SYSTEM_GRAPH_DIR ?=
PROMPT_SYSTEM_TREE_DEPTHS ?= 1,2,3
PROMPT_SYSTEM_TREE_BUDGETS ?= 128,256
PROMPT_SYSTEM_ACTION ?= summary
PROMPT_SYSTEM_NOTE ?=
PROMPT_SYSTEM_LANE ?= rag
PROMPT_SYSTEM_HARNESS ?=
AGENTIC_TASKS ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_tasks_uk.json
AGENTIC_MAX_STEPS ?= 6
AGENTIC_HARNESS ?= loop
AGENTIC_HARNESSES ?= loop langgraph crewai
AGENTIC_CONTEXT_POLICY ?= full
AGENTIC_BASE_URL ?=
AGENTIC_MAX_MODEL_LEN ?=

# Agent loop-policy grid. The exact legacy baseline (6,answer,allow) must remain in every sweep so
# all completion and cost deltas stay paired against shipped behavior.
AGENT_LOOP_TASKS ?= $(AGENTIC_TASKS)
AGENT_MAX_STEPS ?= 4,6,10
AGENT_MALFORMED_POLICY ?= answer,repair_once,strict
AGENT_REPEATED_CALL_POLICY ?= allow,noop
AGENT_REPEAT_FEEDBACK ?= current
AGENT_LOOP_MAX_PROMPT_CHARS ?=
AGENT_LOOP_BASE_URL ?= $(AGENTIC_BASE_URL)
AGENT_LOOP_MAX_MODEL_LEN ?= $(AGENTIC_MAX_MODEL_LEN)
AGENT_LOOP_POWER_DESIGN ?=
AGENT_LOOP_FEEDBACK_DESIGN ?=
AGENT_LOOP_MODEL_FAMILY ?=
AGENT_LOOP_REPEAT_POWER_TASKS ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_loop_repeat_power_uk.json
AGENT_LOOP_REPEAT_POWER_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_loop_repeat_power_design.json
AGENT_LOOP_REPEAT_POWER_ROSTER ?= gemma=hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M qwen=qwen3:14b
AGENT_LOOP_FEEDBACK_DESIGN_FILE ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_loop_feedback_localization_design.json
AGENT_LOOP_FEEDBACK_GENERALIZATION_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_loop_feedback_generalization_design.json
AGENT_LOOP_FEEDBACK_GENERALIZATION_TASKS ?= $(AGENT_LOOP_REPEAT_POWER_TASKS)
AGENT_LOOP_FEEDBACK_ADAPTATION_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_loop_feedback_family_adaptation_design.json
AGENT_LOOP_FEEDBACK_ADAPTATION_TASKS ?= $(AGENT_LOOP_REPEAT_POWER_TASKS)
AGENT_LOOP_FEEDBACK_TRANSFER_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_loop_feedback_task_family_transfer_design.json
AGENT_LOOP_FEEDBACK_TRANSFER_TASKS ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_loop_feedback_task_family_transfer.json
AGENT_LOOP_FEEDBACK_AUTHORITY_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_loop_feedback_controller_authority_design.json
AGENT_LOOP_FEEDBACK_AUTHORITY_TASKS ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_loop_feedback_controller_authority.json
AGENT_LOOP_CONTROLLER_CHANNEL_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_controller_channel_authority_design.json
AGENT_LOOP_CONTROLLER_CHANNEL_TASKS ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_controller_channel_authority.json
AGENT_LOOP_CONTROLLER_CHANNEL_CROSS_MODEL_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_controller_channel_cross_model_design.json
AGENT_LOOP_CONTROLLER_CHANNEL_CROSS_MODEL_TASKS ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_controller_channel_cross_model.json
AGENT_LOOP_CONTROLLER_PREAMBLE_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_controller_preamble_placement_design.json
AGENT_LOOP_CONTROLLER_PREAMBLE_TASKS ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_controller_channel_authority.json

# Agent context-management policies (bench-agentic-context): rank how the agent LOOP spends its
# context window for one fixed model. `full` is the baseline every other policy is paired against.
AGENT_CONTEXT_POLICIES ?= full,observation_cap,keep_last_n,compact
AGENT_CONTEXT_TASKS ?= $(AGENTIC_TASKS)
AGENT_CONTEXT_MAX_STEPS ?= $(AGENTIC_MAX_STEPS)
AGENT_CONTEXT_OBSERVATION_CAP_CHARS ?= 800
# Fraction of the observation-cap budget kept from the HEAD (rest from the tail).
AGENT_CONTEXT_OBSERVATION_HEAD_SHARE ?= 0.6
# Also trims live steps under the compact policy (same cap + aggregate header).
AGENT_CONTEXT_KEEP_LAST_N ?= 3
AGENT_CONTEXT_COMPACT_SHARE ?= 0.5
# Leave empty to resolve the served model's usable window; set to force a prompt-char budget
# (a small value is how the per-step overflow guard is exercised on purpose).
AGENT_CONTEXT_MAX_PROMPT_CHARS ?=
AGENT_CONTEXT_BASE_URL ?= $(AGENTIC_BASE_URL)
AGENT_CONTEXT_MAX_MODEL_LEN ?=
# Constant-sweep task set (large observations); defaults to the same agentic task path.
AGENT_CONTEXT_SWEEP_TASKS ?= $(AGENT_CONTEXT_TASKS)
AGENT_CONTEXT_SWEEP_MAX_STEPS ?= $(AGENT_CONTEXT_MAX_STEPS)
AGENT_CONTEXT_SWEEP_AXES ?= observation_cap_chars,observation_head_share,keep_last_n
AGENT_CONTEXT_SWEEP_MAX_PROMPT_CHARS ?= $(AGENT_CONTEXT_MAX_PROMPT_CHARS)
AGENT_CONTEXT_SWEEP_BASE_URL ?= $(AGENT_CONTEXT_BASE_URL)
AGENT_CONTEXT_SWEEP_MAX_MODEL_LEN ?= $(AGENT_CONTEXT_MAX_MODEL_LEN)
# keep_last_n long-transcript lane: multi-step medium-observation tasks + raised step budget.
AGENT_CONTEXT_KEEP_LONG_TASKS ?= $(DATA_DIR)/agentic-context/tasks_long_transcript.json
AGENT_CONTEXT_KEEP_LONG_MAX_STEPS ?= 12
AGENT_CONTEXT_KEEP_LONG_AXES ?= keep_last_n
AGENT_CONTEXT_KEEP_LONG_FROM_SEARCH ?= $(DATA_DIR)/agentic-context/tasks_24_ua_squad.json
AGENT_CONTEXT_KEEP_LONG_MAX_MATCH_DOCS ?= 6
AGENT_CONTEXT_KEEP_LONG_MAX_OTHER_DOCS ?= 6
AGENT_CONTEXT_KEEP_LONG_MAX_DOC_CHARS ?= 180
# Active compact-vs-cap long-transcript lane. The tighter prompt budget keeps cap usable while
# ensuring the shipped compact_share trigger is crossed after live observation trimming.
AGENT_CONTEXT_COMPACT_LONG_TASKS ?= $(AGENT_CONTEXT_KEEP_LONG_TASKS)
AGENT_CONTEXT_COMPACT_LONG_MAX_STEPS ?= 12
AGENT_CONTEXT_COMPACT_LONG_MAX_PROMPT_CHARS ?= 16000
AGENT_CONTEXT_COMPACT_LONG_COMPACT_SHARE ?= 0.5
AGENT_CONTEXT_COMPACT_LONG_BASE_URL ?= $(AGENT_CONTEXT_BASE_URL)
AGENT_CONTEXT_COMPACT_LONG_MAX_MODEL_LEN ?= $(AGENT_CONTEXT_MAX_MODEL_LEN)
# Memory-dependent compact-vs-cap lane: early read-once fact plus externally checked later work.
AGENT_CONTEXT_COMPACT_MEMORY_TASKS ?= $(DATA_DIR)/agentic-compact-vs-cap/tasks_memory_dependent.json
AGENT_CONTEXT_COMPACT_MEMORY_N_TASKS ?= 8
AGENT_CONTEXT_COMPACT_MEMORY_DEPTH ?= 8
AGENT_CONTEXT_COMPACT_MEMORY_PAD_CHARS ?= 1200
AGENT_CONTEXT_COMPACT_MEMORY_MAX_STEPS ?= 14
AGENT_CONTEXT_COMPACT_MEMORY_MAX_PROMPT_CHARS ?= 8000
AGENT_CONTEXT_COMPACT_MEMORY_COMPACT_SHARE ?= 0.5
AGENT_CONTEXT_COMPACT_MEMORY_MIN_COMPACTION_RATE ?= 0.75
AGENT_CONTEXT_COMPACT_MEMORY_BASE_URL ?= $(AGENT_CONTEXT_BASE_URL)
AGENT_CONTEXT_COMPACT_MEMORY_MAX_MODEL_LEN ?= $(AGENT_CONTEXT_MAX_MODEL_LEN)
AGENT_CONTEXT_COMPACT_MEMORY_TRANSFER_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_compact_memory_transfer_design.json
AGENT_CONTEXT_COMPACT_MEMORY_REPLICATION_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_compact_memory_replication_design.json
AGENT_CONTEXT_COMPACT_MEMORY_BOUNDARY_SURFACE_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_compact_memory_boundary_surface_design.json
AGENT_CONTEXT_COMPACT_TRIGGER_COLLAPSE_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_compact_trigger_guard_collapse_design.json
AGENT_CONTEXT_COMPACT_FOLD_STEP_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_compact_fold_step_crossover_design.json
AGENT_CONTEXT_COMPACT_REPEATED_FOLD_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_compact_repeated_fold_completion_design.json
AGENT_CONTEXT_COMPACT_SUMMARY_INPUT_CAP_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_compact_summary_input_cap_design.json
AGENT_CONTEXT_COMPACT_WINDOW_ELISION_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_compact_window_elision_design.json
AGENT_CONTEXT_COMPACT_WINDOW_ELISION_TRANSFER_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_compact_window_elision_transfer_design.json
AGENT_CONTEXT_COMPACT_CROSSOVER_RESTATEMENT_DESIGN ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_compact_crossover_restatement_design.json
AGENT_CONTEXT_COMPACT_CROSSOVER_RESTATEMENT_SURFACE ?=
KNOWLEDGE_CUTOFF_EVENTS ?=
KNOWLEDGE_CUTOFF_DATASET ?= apoorvumang/knowledge-cutoff-benchmark
KNOWLEDGE_CUTOFF_REVISION ?= main
KNOWLEDGE_CUTOFF_THRESHOLD ?= 0.5
KNOWLEDGE_CUTOFF_TRIALS ?= 200
KNOWLEDGE_CUTOFF_SEED ?= 42
KNOWLEDGE_CUTOFF_LIMIT ?=
KNOWLEDGE_CUTOFF_BASE_URL ?=
KNOWLEDGE_CUTOFF_MAX_MODEL_LEN ?=
KNOWLEDGE_CUTOFF_UA_TRANSLATOR_MODEL ?= $(MODEL)
KNOWLEDGE_CUTOFF_UA_TRANSLATOR_BACKEND ?= $(BACKEND)
KNOWLEDGE_CUTOFF_UA_TRANSLATOR_BASE_URL ?=
KNOWLEDGE_CUTOFF_UA_TRANSLATOR_GPU_MEMORY ?= 0.95
KNOWLEDGE_CUTOFF_UA_BUNDLE ?= $(DATA_DIR)/knowledge-cutoff-ua/$(KNOWLEDGE_CUTOFF_REVISION)
KNOWLEDGE_CUTOFF_UA_REVIEWER ?=
KNOWLEDGE_CUTOFF_UA_REVISIONS ?=
KNOWLEDGE_CUTOFF_BILINGUAL_GPU_MEMORY ?= 0.9
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
JOINT_SEARCH_CANDIDATES ?= $(MODELS_MANIFEST)
JOINT_SEARCH_TRIALS ?= 20
JOINT_SEARCH_SCREEN_LIMIT ?= 8
JOINT_SEARCH_MIN_FINALISTS ?= 2
JOINT_SEARCH_OBJECTIVES ?= quality,latency
JOINT_SEARCH_RUN_ID ?=
JOINT_SEARCH_OFFLINE ?=
JOINT_SEARCH_CORPUS ?=
JOINT_SEARCH_LIMIT ?=
JOINT_SEARCH_NO_ISOLATE ?=

# Research-scale roster confirmation (`joint-search-long-run`). The two POWER_* run bundles supply
# the paired variance the tuning-screen size is derived from; the incumbent is the default model
# the adopt-or-retain verdict is measured against.
LONG_RUN_INCUMBENT ?=
LONG_RUN_POWER_REFERENCE ?=
LONG_RUN_POWER_BASELINE ?=
LONG_RUN_MIN_GAIN ?= 0.05
LONG_RUN_TARGET_POWER ?= 0.80
LONG_RUN_CONFIDENCE ?= 0.95
LONG_RUN_TRIAL_BUDGET ?= 30
LONG_RUN_TRIAL_BLOCK ?= 5
LONG_RUN_STABILITY_BLOCKS ?= 2
LONG_RUN_STABILITY_AGREEMENT ?= 1.0
LONG_RUN_RUN_ID ?=
LONG_RUN_PUBLIC_LIMIT ?=
# Unload Ollama's resident models before a vLLM finalist's public screen launches. On by default:
# the confirmation run owns the host for hours, the resident models are the ones IT loaded, and a
# vLLM engine refuses to start on the VRAM an Ollama keep-alive still holds. Set empty to opt out.
LONG_RUN_PUBLIC_EVICT ?= 1
LONG_RUN_OFFLINE ?=
LONG_RUN_NO_ISOLATE ?=
# Autonomous corpus-to-recommendation pipeline. The default drafter is a 12B Ukrainian model that
# fits the 16 GiB development host; override it for another host tier or installed local tag.
AUTO_RAG_DRAFT_MODEL ?= hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M
AUTO_RAG_RUN_ID ?=
AUTO_RAG_CANDIDATES ?= $(MODELS_MANIFEST)
AUTO_RAG_CANDIDATE_MODELS ?=
AUTO_RAG_JUDGE_MODEL ?=
AUTO_RAG_JUDGE_BASE_URL ?=
AUTO_RAG_MAX_ITEMS ?= 60
AUTO_RAG_DOC_LIMIT ?=
AUTO_RAG_TRIALS ?= 20
AUTO_RAG_SCREEN_LIMIT ?= 8
AUTO_RAG_MIN_FINALISTS ?= 2
AUTO_RAG_EVAL_LIMIT ?=
AUTO_RAG_MAX_MODEL_LEN ?= 8192
AUTO_RAG_PARITY_CHECK ?=
SCORER_POLICY ?= auto
SCORER_EGRESS_CONSENT ?=
SCORER_MAX_USD ?=
SCORER_MAX_CALLS ?=
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

# Frontier judge authorization (`make frontier-judge-agreement`). FRONTIER_JUDGE_MODELS is a
# comma-separated list of litellm ids; each needs its provider key in .env. The lane SENDS the
# worksheet answers to those providers and SPENDS money, so it refuses to run without
# FRONTIER_EGRESS_CONSENT=1 and at least one of FRONTIER_MAX_USD / FRONTIER_MAX_CALLS. Keep the
# worksheet on public/committed fixture data: private corpora must not leave the host.
FRONTIER_JUDGE_MODELS ?=
FRONTIER_EGRESS_CONSENT ?=
FRONTIER_MAX_USD ?=
FRONTIER_MAX_CALLS ?=
FRONTIER_JUDGE_THRESHOLD ?= 0.6
FRONTIER_JUDGE_LIMIT ?=
FRONTIER_JUDGE_OUT ?=
# Data-verification knobs (the new-goldset flow: cross-check -> human verification gate sample-verify). BUNDLE is a
# draft dir (goldset.jsonl + corpus/) under $(DATA_DIR)/prepare-goldset/<ts>/; the sample worksheet
# + accepted-ledger default beside it. CROSS_CHECK_MODEL is the SECOND-frontier verifier (must
# differ from the drafter). VERIFY_N sizes the stratified sample; VERIFY_TOLERANCE is the accepted
# reject rate. Full workflow: docs/guides/data-prep/goldset-from-scratch.md.
BUNDLE ?=
CROSS_CHECK_MODEL ?=
VERIFY_WS ?= $(if $(BUNDLE),$(BUNDLE)/verify_sample.csv,)
VERIFY_N ?=
VERIFY_SEED ?= 13
VERIFY_CONFIDENCE ?= 0.95
VERIFY_PRECISION ?= 0.10
VERIFY_EXPECTED_REJECT_RATE ?= 0.50
VERIFY_TOLERANCE ?= 0.05
# VERIFY_MERGE=1 enlarges an existing worksheet additively (decided rows preserved byte-for-byte,
# never re-shown); VERIFY_ORDER=confidence reviews least-confident items first (cross-check
# verdict + retrieval rank decide the queue; the CSV row order never changes).
VERIFY_MERGE ?=
VERIFY_ORDER ?=
VERIFY_KIND ?= auto
# Multi-annotator gate: VERIFY_ANNOTATORS=k writes the SAME stratified sample as k per-reviewer
# worksheets (verify_sample.r<i>.csv); verify-adjudicate then reports agreement (Cohen/Fleiss
# kappa) and draws disagreements into adjudication.csv. VERIFY_ACCEPT_POLICY picks the acceptance
# arithmetic (global | per-stratum | weighted); VERIFY_STRATUM_TOLERANCES overrides per-stratum
# thresholds as space-separated "provenance|split|doc=tol" pairs.
VERIFY_ANNOTATORS ?=
VERIFY_ACCEPT_POLICY ?=
VERIFY_STRATUM_TOLERANCES ?=
CHAIN_CORPUS ?=
CHAIN_BUNDLE ?=
CHAIN_WS ?= $(if $(CHAIN_BUNDLE),$(CHAIN_BUNDLE)/verify_chains.csv,)
CHAIN_FIXTURE ?=
CHAIN_VERIFY_N ?=
CHAIN_MAX_PATHS ?= 80
CHAIN_MIN_ACCEPTED ?=
CHAIN_MIN_RETENTION_FRACTION ?= 0.50
# Composite-headline pipeline knobs. Each verification ref must point at a reviewed
# verify_sample.csv, a sample_manifest.json that points to one, or an accepted-ledger bundle.
COMPOSITE_SAMPLE_VERIFICATION_ROOT ?= $(PROJECT_ROOT)/samples/verification/composite_samples
COMPOSITE_TEXT_ANALYSIS_BUNDLE ?= $(PROJECT_ROOT)/samples/text_analysis_bundle_uk
COMPOSITE_SUMMARIZATION_CASES ?= $(PROJECT_ROOT)/samples/benchmarks/summarization_cases_uk.json
COMPOSITE_STRUCTURED_CASES ?= $(PROJECT_ROOT)/samples/benchmarks/structured_cases_uk.json
COMPOSITE_SECURITY_CASES ?= $(PROJECT_ROOT)/samples/benchmarks/security_cases_uk.json
COMPOSITE_AGENTIC_TASKS ?= $(PROJECT_ROOT)/samples/benchmarks/agentic_tasks_uk.json
COMPOSITE_TOOLING_CATALOG ?= $(PROJECT_ROOT)/samples/benchmarks/tooling_cases_uk.json
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

# Corpus-specific derived security cases (derive-security-cases / derive-security-worksheet).
# SECURITY_DERIVE_CASES defaults to the committed regression fixture so the worksheet + verified
# bench run are runnable out of the box; point it at a fresh derive output for a real corpus.
SECURITY_DERIVE_CASES ?= $(PROJECT_ROOT)/samples/benchmarks/security_cases_derived_uk.json
SECURITY_DERIVE_WORKSHEET ?=

JUDGE_MODEL ?= gemma3:27b
JUDGE_BASE_URL ?= http://localhost:11434/v1
JUDGE_RHO ?=

# Context-policy benchmark (bench-chain-context): rank context-management policies for one model
# over a verified chain set. CHAIN_CONTEXT_CHAINS/CORPUS default to the committed fixture.
CHAIN_CONTEXT_CHAINS ?= $(PROJECT_ROOT)/samples/goldsets/chain_context_uk_v1/chains.jsonl
CHAIN_CONTEXT_CORPUS ?= $(PROJECT_ROOT)/samples/goldsets/chain_context_uk_v1/corpus
CHAIN_CONTEXT_POLICIES ?= fresh,history,summary,roles
CHAIN_CONTEXT_TOP_K ?= 4
CHAIN_CONTEXT_MODEL ?= $(MODEL)
CHAIN_CONTEXT_BACKEND ?= $(BACKEND)
CHAIN_CONTEXT_BASE_URL ?=
CHAIN_CONTEXT_MAX_MODEL_LEN ?=
CHAIN_CONTEXT_INDEX_DIR ?=
CHAIN_CONTEXT_DATA_VERIFIED ?=
CHAIN_CONTEXT_VERIFICATION_REF ?=
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
QUICKSTART_DRAFT_VERIFY_N ?=
QUICKSTART_DRAFT_VERIFY_CONFIDENCE ?= 0.95
QUICKSTART_DRAFT_VERIFY_PRECISION ?= 0.10
QUICKSTART_DRAFT_TIMEOUT ?= 900
QUICKSTART_DRAFT_MAX_USD ?=
QUICKSTART_DRAFT_MAX_CALLS ?= 1000
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
QUICKSTART_MODEL_SELECTION ?= auto
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
export QUICKSTART_DRAFT_VERIFY_CONFIDENCE QUICKSTART_DRAFT_VERIFY_PRECISION
export QUICKSTART_DRAFT_TIMEOUT QUICKSTART_DRAFT_MAX_TOKENS QUICKSTART_DRAFT_TEMPERATURE
export QUICKSTART_DRAFT_MAX_USD QUICKSTART_DRAFT_MAX_CALLS
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
