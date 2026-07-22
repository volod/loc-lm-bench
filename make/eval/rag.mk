## RAG stores, retrieval evaluation, scored runs, probes, and miss analysis.

.PHONY: build-rag-store build-index build-graph refresh-index validate-retrieval \
	compare-retrieval compare-graph-fusion compare-embeddings run-eval bench-query-robustness \
	probe-context-position analyze-misses

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

refresh-index: ## Incrementally refresh built stores after corpus edits + drift report (CORPUS= GOLDSET= RETUNE_THRESHOLD= SKIP_GRAPH=1 GRAPH_EXTRACTION=<jsonl>)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main refresh-index \
		$(if $(CORPUS),--corpus-root "$(CORPUS)",) \
		$(if $(GOLDSET),--goldset "$(GOLDSET)",) \
		--k $(RAG_K) \
		$(if $(RETUNE_THRESHOLD),--retune-threshold $(RETUNE_THRESHOLD),) \
		$(if $(SKIP_GRAPH),--skip-graph,) \
		$(if $(GRAPH_EXTRACTION),--graph-extraction "$(GRAPH_EXTRACTION)",)

validate-retrieval: ## RAG recall/MRR; QUERY_PREP=... QUERY_PREP_MODEL= QUERY_PREP_BACKEND=ollama QUERY_PREP_AB=1 QUERY_PREP_OUT= for model-backed A/B
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main validate-retrieval $(if $(CONFIG),--config "$(CONFIG)",) \
		--goldset "$(GOLDSET)" --k $(RAG_K) $(if $(SPLIT),--split "$(SPLIT)",) \
		$(if $(RETRIEVAL_BACKEND),--retrieval-backend "$(RETRIEVAL_BACKEND)",) \
		$(if $(RETRIEVAL_STRATEGY),--retrieval-strategy "$(RETRIEVAL_STRATEGY)",) \
		$(if $(GRAPH_WEIGHT),--graph-weight $(GRAPH_WEIGHT),) \
		$(if $(QUERY_PREP),--query-prep "$(QUERY_PREP)",) \
		$(if $(QUERY_GLOSSARY),--query-glossary "$(QUERY_GLOSSARY)",) \
		$(if $(QUERY_PREP_TYPO_GUARD),--query-prep-typo-guard,) \
		$(if $(QUERY_PREP_MODEL),--query-prep-model "$(QUERY_PREP_MODEL)",) \
		$(if $(QUERY_PREP_BACKEND),--query-prep-backend "$(QUERY_PREP_BACKEND)",) \
		$(if $(QUERY_PREP_AB),--query-prep-ab,) \
		$(if $(QUERY_PREP_OUT),--out "$(QUERY_PREP_OUT)",)

compare-retrieval: ## Compare vector, graph, and fused recall@k/MRR; GRAPH_WEIGHT= controls the fused graph share; CHUNK_STRATEGIES=..., HYBRID=1, RERANKER= are optional lanes
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main compare-retrieval $(if $(CONFIG),--config "$(CONFIG)",) \
		--goldset "$(GOLDSET)" --k $(RAG_K) $(if $(SPLIT),--split "$(SPLIT)",) \
		$(if $(CHUNK_STRATEGIES),--strategies "$(CHUNK_STRATEGIES)",) \
		$(if $(HYBRID),--hybrid,) \
		$(if $(FUSION_WEIGHT),--fusion-weight $(FUSION_WEIGHT),) \
		$(if $(GRAPH_WEIGHT),--graph-weight $(GRAPH_WEIGHT),) \
		$(if $(RERANKER),--reranker "$(RERANKER)",) \
		$(if $(RERANK_CANDIDATES),--rerank-candidates $(RERANK_CANDIDATES),) \
		$(if $(COMPARE_RETRIEVAL_OUT),--out "$(COMPARE_RETRIEVAL_OUT)",)

compare-graph-fusion: ## Sweep the graph share of graph-vector fusion and decide it on the multi-hop slice (GOLDSET= GRAPH_WEIGHTS= GRAPH_STRATEGIES= FUSION_FOCUS_SLICE= FUSION_OUT_DIR=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main compare-graph-fusion $(if $(CONFIG),--config "$(CONFIG)",) \
		--goldset "$(GOLDSET)" --k $(RAG_K) $(if $(SPLIT),--split "$(SPLIT)",) \
		$(if $(GRAPH_WEIGHTS),--graph-weights "$(GRAPH_WEIGHTS)",) \
		$(if $(GRAPH_STRATEGIES),--graph-strategies "$(GRAPH_STRATEGIES)",) \
		$(if $(FUSION_FOCUS_SLICE),--focus-slice "$(FUSION_FOCUS_SLICE)",) \
		$(if $(FUSION_BOOTSTRAP_RESAMPLES),--resamples $(FUSION_BOOTSTRAP_RESAMPLES),) \
		$(if $(FUSION_OUT_DIR),--out-dir "$(FUSION_OUT_DIR)",)

compare-embeddings: ## embedding-bakeoff-uk: rank UA embedders (recall@k/MRR + throughput) on GOLDSET; MODELS= EMBED_API_MODEL= (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main compare-embeddings --goldset "$(GOLDSET)" --k $(RAG_K) \
		$(if $(MODELS),--models "$(MODELS)",) \
		$(if $(EMBED_API_MODEL),--api-model "$(EMBED_API_MODEL)" --data-classification "$(EMBED_DATA_CLASSIFICATION)" $(if $(EMBED_MAX_USD),--max-usd $(EMBED_MAX_USD),),)

run-eval: ## Run the eval; MODEL= BACKEND= GOLDSET= SPLIT= RETRIEVAL_BACKEND=fused GRAPH_WEIGHT=0.3 RETRIEVAL_MODE=hybrid ACL_LABEL=tag RERANKER= CONTEXT_ORDER= QUERY_PREP=... RESUME=<run-dir>
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main run-eval $(if $(CONFIG),--config "$(CONFIG)",) \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --split "$(SPLIT)" \
		$(if $(RETRIEVAL_BACKEND),--retrieval-backend "$(RETRIEVAL_BACKEND)",) \
		$(if $(RETRIEVAL_STRATEGY),--retrieval-strategy "$(RETRIEVAL_STRATEGY)",) \
		$(if $(RETRIEVAL_MODE),--retrieval-mode "$(RETRIEVAL_MODE)",) \
		$(if $(ACL_LABEL),--acl "$(ACL_LABEL)",) \
		$(if $(FUSION_WEIGHT),--fusion-weight $(FUSION_WEIGHT),) \
		$(if $(GRAPH_WEIGHT),--graph-weight $(GRAPH_WEIGHT),) \
		$(if $(RERANKER),--reranker "$(RERANKER)",) \
		$(if $(RERANK_CANDIDATES),--rerank-candidates $(RERANK_CANDIDATES),) \
		$(if $(CONTEXT_ORDER),--context-order "$(CONTEXT_ORDER)",) \
		$(if $(QUERY_PREP),--query-prep "$(QUERY_PREP)",) \
		$(if $(QUERY_GLOSSARY),--query-glossary "$(QUERY_GLOSSARY)",) \
		$(if $(QUERY_PREP_TYPO_GUARD),--query-prep-typo-guard,) \
		$(if $(CITED_ANSWERS),--cited-answers,) \
		$(if $(SCORE_GROUNDEDNESS),--score-groundedness,) \
		$(if $(INSUFFICIENT_CONTEXT_PROBES),--insufficient-context-probes $(INSUFFICIENT_CONTEXT_PROBES),) \
		--limit $(LIMIT) $(if $(TELEMETRY),--telemetry) \
		$(if $(RESUME),--resume "$(RESUME)",) \
		$(if $(PROMPT_SYSTEM_ID),--prompt-system "$(PROMPT_SYSTEM_ID)",) \
		$(if $(PROMPT_PACKAGE),--prompt-package "$(PROMPT_PACKAGE)",) \
		$(if $(JUDGE_RHO),--judge-rho $(JUDGE_RHO) --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)"))

bench-query-robustness: ## Noisy UA queries vs clean RAG + normalize,typos recovery (MODEL= BACKEND= GOLDSET= CORPUS= SPLIT= QUERY_ROBUSTNESS_LIMIT=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-query-robustness --model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --corpus-root "$(CORPUS)" --split "$(SPLIT)" \
		--top-k $(RAG_K) --typo-rate $(QUERY_ROBUSTNESS_TYPO_RATE) \
		--max-tokens $(QUERY_ROBUSTNESS_MAX_TOKENS) \
		$(if $(QUERY_ROBUSTNESS_LIMIT),--limit $(QUERY_ROBUSTNESS_LIMIT),)

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
