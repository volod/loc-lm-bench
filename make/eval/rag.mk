## RAG stores, retrieval evaluation, scored runs, probes, and miss analysis.

.PHONY: build-rag-store build-index build-graph validate-retrieval compare-retrieval \
	compare-embeddings run-eval probe-context-position analyze-misses

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

validate-retrieval: ## RAG core: recall@k / MRR of the pinned embedding over the gold set; QUERY_PREP=normalize,typos,glossary QUERY_PREP_TYPO_GUARD=1 QUERY_PREP_AB=1 QUERY_GLOSSARY= for the query-side A/B (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main validate-retrieval --goldset "$(GOLDSET)" --k $(RAG_K) \
		$(if $(QUERY_PREP),--query-prep "$(QUERY_PREP)",) \
		$(if $(QUERY_GLOSSARY),--query-glossary "$(QUERY_GLOSSARY)",) \
		$(if $(QUERY_PREP_TYPO_GUARD),--query-prep-typo-guard,) \
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
		$(if $(QUERY_PREP_TYPO_GUARD),--query-prep-typo-guard,) \
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
