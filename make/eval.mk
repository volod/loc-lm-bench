# RAG evaluation, prompt-system, benchmark, and pipeline targets.
##@ Evaluation and Pipelines

.PHONY: \
	build-rag-store build-index build-graph validate-retrieval compare-retrieval \
	compare-embeddings run-eval probe-context-position analyze-misses score-external-rag sweep pipeline prompt-system-prepare prompt-system-review \
	export-finetune-set finetune-adapter self-improve finetune-campaign \
	register-adapter list-adapters serve-adapter gc-adapters \
	prompt-system-compare bench-security bench-agentic agentic-harness-compare \
	composite-headline platform-matrix

build-rag-store: ## Chunk a corpus with all strategies into DATA_DIR/llb/rag (CORPUS_DIR=...)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.rag.chunking --corpus-root "$(CORPUS_DIR)" \
		--out-dir "$(DATA_DIR)/llb/rag" --strategy all --size 800 --overlap 120

build-index: ## RAG core: chunk + embed CORPUS into the FAISS store (CHUNK_STRATEGY= EMBEDDING_MODEL= RETRIEVAL_MODE=hybrid LEMMATIZE=1 to override; needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main build-index --corpus-root "$(CORPUS)" \
		$(if $(CHUNK_STRATEGY),--strategy "$(CHUNK_STRATEGY)",) \
		$(if $(EMBEDDING_MODEL),--embedding-model "$(EMBEDDING_MODEL)",) \
		$(if $(RETRIEVAL_MODE),--retrieval-mode "$(RETRIEVAL_MODE)",) \
		$(if $(LEMMATIZE),--lemmatize,)

build-graph: ## GraphRAG backend: build the GraphRAG store from an ontology-assisted draft bundle (BUNDLE=...; needs ".[graph]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<prepare-goldset dir> (extraction.jsonl + corpus/)"; exit 1; }
	$(PY) -m llb.main build-graph --bundle "$(BUNDLE)"

validate-retrieval: ## RAG core: recall@k / MRR of the pinned embedding over the gold set; QUERY_PREP=normalize,typos,glossary QUERY_PREP_AB=1 QUERY_GLOSSARY= for the query-side A/B (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main validate-retrieval --goldset "$(GOLDSET)" --k $(RAG_K) \
		$(if $(QUERY_PREP),--query-prep "$(QUERY_PREP)",) \
		$(if $(QUERY_GLOSSARY),--query-glossary "$(QUERY_GLOSSARY)",) \
		$(if $(QUERY_PREP_AB),--query-prep-ab,)

compare-retrieval: ## Compare faiss vs graph backends' recall@k/MRR on the gold set; CHUNK_STRATEGIES=... ranks chunkers, HYBRID=1 ranks dense vs hybrid(+lemmas) + oracle-doc headroom, RERANKER=<hf-id> adds reranked twin rows (RERANK_CANDIDATES=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main compare-retrieval --goldset "$(GOLDSET)" --k $(RAG_K) \
		$(if $(CHUNK_STRATEGIES),--strategies "$(CHUNK_STRATEGIES)",) \
		$(if $(HYBRID),--hybrid,) \
		$(if $(FUSION_WEIGHT),--fusion-weight $(FUSION_WEIGHT),) \
		$(if $(RERANKER),--reranker "$(RERANKER)",) \
		$(if $(RERANK_CANDIDATES),--rerank-candidates $(RERANK_CANDIDATES),)

compare-embeddings: ## embedding-bakeoff-uk: rank UA embedders (recall@k/MRR + throughput) on GOLDSET; MODELS= EMBED_API_MODEL= (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main compare-embeddings --goldset "$(GOLDSET)" --k $(RAG_K) \
		$(if $(MODELS),--models "$(MODELS)",) \
		$(if $(EMBED_API_MODEL),--api-model "$(EMBED_API_MODEL)" --data-classification "$(EMBED_DATA_CLASSIFICATION)" $(if $(EMBED_MAX_USD),--max-usd $(EMBED_MAX_USD),),)

run-eval: ## Run the eval; MODEL= BACKEND= GOLDSET= SPLIT= RETRIEVAL_MODE=hybrid ACL_LABEL=tag RERANKER= CONTEXT_ORDER= QUERY_PREP=normalize,typos,glossary QUERY_GLOSSARY= CITED_ANSWERS=1 SCORE_GROUNDEDNESS=1 INSUFFICIENT_CONTEXT_PROBES=n PROMPT_SYSTEM_ID= PROMPT_PACKAGE= RESUME=<run-dir>
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main run-eval --model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --split "$(SPLIT)" \
		$(if $(RETRIEVAL_MODE),--retrieval-mode "$(RETRIEVAL_MODE)",) \
		$(if $(ACL_LABEL),--acl "$(ACL_LABEL)",) \
		$(if $(FUSION_WEIGHT),--fusion-weight $(FUSION_WEIGHT),) \
		$(if $(RERANKER),--reranker "$(RERANKER)",) \
		$(if $(RERANK_CANDIDATES),--rerank-candidates $(RERANK_CANDIDATES),) \
		$(if $(CONTEXT_ORDER),--context-order "$(CONTEXT_ORDER)",) \
		$(if $(QUERY_PREP),--query-prep "$(QUERY_PREP)",) \
		$(if $(QUERY_GLOSSARY),--query-glossary "$(QUERY_GLOSSARY)",) \
		$(if $(CITED_ANSWERS),--cited-answers,) \
		$(if $(SCORE_GROUNDEDNESS),--score-groundedness,) \
		$(if $(INSUFFICIENT_CONTEXT_PROBES),--insufficient-context-probes $(INSUFFICIENT_CONTEXT_PROBES),) \
		--limit $(LIMIT) $(if $(TELEMETRY),--telemetry) \
		$(if $(RESUME),--resume "$(RESUME)",) \
		$(if $(PROMPT_SYSTEM_ID),--prompt-system "$(PROMPT_SYSTEM_ID)",) \
		$(if $(PROMPT_PACKAGE),--prompt-package "$(PROMPT_PACKAGE)",) \
		$(if $(JUDGE_RHO),--judge-rho $(JUDGE_RHO) --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)"))

probe-context-position: ## Lost-in-the-middle probe: gold chunk at head/middle/tail at fixed k -> per-model context-order recommendation (MODEL= BACKEND= GOLDSET= PROBE_K= SPLIT= LIMIT=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main probe-context-position --model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --split "$(SPLIT)" --k $(PROBE_K) \
		$(if $(LIMIT),--limit $(LIMIT),)

analyze-misses: ## Miss analysis: classify + cluster one run's misses (RUN_DIR=<bundle>; PROBE_TOP_K=3,8 re-runs the miss subset; MISS_THRESHOLD= ANALYZE_GOLDSET=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(RUN_DIR)" || { echo "ERROR: set RUN_DIR=<run-eval bundle dir>"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main analyze-misses --run-dir "$(RUN_DIR)" \
		$(if $(ANALYZE_GOLDSET),--goldset "$(ANALYZE_GOLDSET)",) \
		$(if $(MISS_THRESHOLD),--miss-threshold $(MISS_THRESHOLD),) \
		$(if $(PROBE_TOP_K),--probe-top-k "$(PROBE_TOP_K)",)

export-finetune-set: ## Export tuning-split SFT/DPO records (RUN_DIR=<tuning-run> GOLDSET= OUT_DIR= MISSES=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(RUN_DIR)" || { echo "ERROR: set RUN_DIR=<tuning run-eval bundle dir>"; exit 1; }
	@test -n "$(OUT_DIR)" || { echo "ERROR: set OUT_DIR=<dataset dir>"; exit 1; }
	$(PY) -m llb.main export-finetune-set --run-dir "$(RUN_DIR)" --goldset "$(GOLDSET)" \
		--out "$(OUT_DIR)" $(if $(MISSES),--misses "$(MISSES)",)

finetune-adapter: ## Train a LoRA/QLoRA adapter (DATASET=<export dir> MODEL=<base> ADAPTER_OUT= TRAINER=auto|fake)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(DATASET)" || { echo "ERROR: set DATASET=<export-finetune-set dir>"; exit 1; }
	@test -n "$(MODEL)" || { echo "ERROR: set MODEL=<base model>"; exit 1; }
	$(PY) -m llb.main finetune-adapter --dataset "$(DATASET)" --model "$(MODEL)" \
		$(if $(ADAPTER_OUT),--out "$(ADAPTER_OUT)",) $(if $(TRAINER),--trainer "$(TRAINER)",)

self-improve: ## Local self-improvement loop (MODEL= BACKEND= GOLDSET= ROUNDS=2 LIMIT= TRAINER=auto|fake)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main self-improve --model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --rounds "$(ROUNDS)" \
		$(if $(LIMIT),--limit "$(LIMIT)",) \
		$(if $(SELF_IMPROVE_OUT),--out-dir "$(SELF_IMPROVE_OUT)",) \
		$(if $(SELF_IMPROVE_RESUME),--resume "$(SELF_IMPROVE_RESUME)",) \
		$(if $(TRAINER),--trainer "$(TRAINER)",)

finetune-campaign: ## Multi-model adapter campaign (MODELS=<csv> BACKEND= GOLDSET= ROUNDS=1 TRAINER=auto|fake)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main finetune-campaign --models "$(or $(MODELS),$(FINETUNE_CAMPAIGN_MODELS))" \
		--backend "$(BACKEND)" --goldset "$(GOLDSET)" --corpus "$(CORPUS)" \
		--rounds "$(or $(ROUNDS),$(FINETUNE_CAMPAIGN_ROUNDS))" \
		$(if $(FINETUNE_CAMPAIGN_LIMIT),--limit "$(FINETUNE_CAMPAIGN_LIMIT)",) \
		$(if $(FINETUNE_CAMPAIGN_OUT),--out-dir "$(FINETUNE_CAMPAIGN_OUT)",) \
		$(if $(FINETUNE_CAMPAIGN_RESUME),--resume "$(FINETUNE_CAMPAIGN_RESUME)",) \
		$(if $(FINETUNE_CAMPAIGN_MANIFEST),--manifest "$(FINETUNE_CAMPAIGN_MANIFEST)",) \
		$(if $(TRAINER),--trainer "$(TRAINER)",)

register-adapter: ## Register an adapter trained outside the loop (ADAPTER_DIR=<dir> GOLDSET= CORPUS= SOURCE_RUN=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(ADAPTER_DIR)" || { echo "ERROR: set ADAPTER_DIR=<adapter dir>"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main register-adapter --adapter-dir "$(ADAPTER_DIR)" \
		$(if $(GOLDSET),--goldset "$(GOLDSET)",) $(if $(CORPUS),--corpus "$(CORPUS)",) \
		$(if $(SOURCE_RUN),--source-run "$(SOURCE_RUN)",)

list-adapters: ## List registered adapters with base model, eval evidence, and staleness verdict (ADAPTERS_JSON=1)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main list-adapters $(if $(ADAPTERS_JSON),--json,)

serve-adapter: ## Serve a registered adapter (ADAPTER=<id> BACKEND=vllm|ollama|llamacpp SERVE_SMOKE=1 to probe and exit)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(ADAPTER)" || { echo "ERROR: set ADAPTER=<adapter id> (see 'make list-adapters')"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main serve-adapter --adapter "$(ADAPTER)" \
		$(if $(BACKEND),--backend "$(BACKEND)",) $(if $(SERVE_SMOKE),--smoke,)

gc-adapters: ## Delete superseded adapters no run bundle cites (GC_FORCE=1 overrides citations; GC_DRY_RUN=1 previews)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main gc-adapters $(if $(GC_FORCE),--force,) $(if $(GC_DRY_RUN),--dry-run,)

score-external-rag: ## Human-score answered external RAG JSONL; final CSV/report after all rows are scored (EXTERNAL_RAG_ANSWERS=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(EXTERNAL_RAG_ANSWERS)" || { echo "ERROR: set EXTERNAL_RAG_ANSWERS=<answered-jsonl>"; exit 1; }
	@set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main score-external-rag --answers "$(EXTERNAL_RAG_ANSWERS)" \
		--source-limit "$(EXTERNAL_RAG_SOURCE_LIMIT)" \
		$(if $(EXTERNAL_RAG_CSV),--csv-out "$(EXTERNAL_RAG_CSV)",) \
		$(if $(EXTERNAL_RAG_REPORT),--report-out "$(EXTERNAL_RAG_REPORT)",) \
		$(if $(EXTERNAL_RAG_ANSWER_FIELD),--answer-field "$(EXTERNAL_RAG_ANSWER_FIELD)",) \
		$(if $(EXTERNAL_RAG_SOURCES_FIELD),--sources-field "$(EXTERNAL_RAG_SOURCES_FIELD)",) \
		$(if $(EXTERNAL_RAG_ERROR_FIELD),--error-field "$(EXTERNAL_RAG_ERROR_FIELD)",) \
		$(if $(EXTERNAL_RAG_LABEL),--label "$(EXTERNAL_RAG_LABEL)",) \
		$(if $(EXTERNAL_RAG_START),--start "$(EXTERNAL_RAG_START)",) \
		$(if $(EXTERNAL_RAG_CLEAR),--clear,) \
		$(if $(EXTERNAL_RAG_KEEP_SOURCE_FOOTER),--keep-source-footer,)

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
