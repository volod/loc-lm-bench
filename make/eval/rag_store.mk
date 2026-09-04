## RAG stores, retrieval evaluation, scored runs, probes, and miss analysis.

.PHONY: build-rag-store build-index build-graph resolve-graph-entities refresh-index validate-retrieval \
	check-generation \
	measure-duplicate-residue \
	compare-retrieval compare-graph-fusion compare-answer-quality compare-embeddings \
	compare-answer-validation check-answer-gate \
	venv-encoders-legacy compare-embeddings-legacy compare-rerankers-legacy \
	compare-embedder-adoption compare-adoption-models compare-adoption-roster \
	compare-adoption-screen compare-vector-stores run-eval \
	calibrate-fusion-routing compare-context-strategies bench-query-robustness \
	sweep-restoration-constraints \
	probe-context-position probe-multihop-hops analyze-misses

build-rag-store: ## Chunk a corpus with all strategies into DATA_DIR/llb/rag (CORPUS_DIR=...)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.rag.chunking --corpus-root "$(CORPUS_DIR)" \
		--out-dir "$(DATA_DIR)/llb/rag" --strategy all --size 800 --overlap 120

# With CONFIG= the YAML owns corpus_root (and the store's DATA_DIR), so the default CORPUS is
# forwarded only when the caller actually set it on the command line or in the environment --
# otherwise a config-targeted build would silently chunk the DEFAULT corpus into the config's
# store. Without CONFIG the default corpus is the documented behavior and is always forwarded.
BUILD_INDEX_CORPUS = $(if $(CONFIG),$(if $(filter-out file default,$(origin CORPUS)),--corpus-root "$(CORPUS)"),--corpus-root "$(CORPUS)")

build-index: ## RAG core: chunk + embed CORPUS into the FAISS store (CONFIG= CHUNK_STRATEGY= CHUNK_SIZE= CHUNK_OVERLAP= EMBEDDING_MODEL= RETRIEVAL_MODE=hybrid LEMMATIZE=1 KEEP_DUPLICATE_CHUNKS=1 DUPLICATE_TIER=exact|normalized|masked; needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main build-index $(if $(CONFIG),--config "$(CONFIG)",) \
		$(BUILD_INDEX_CORPUS) \
		$(if $(CHUNK_STRATEGY),--strategy "$(CHUNK_STRATEGY)",) \
		$(if $(CHUNK_SIZE),--size "$(CHUNK_SIZE)",) \
		$(if $(CHUNK_OVERLAP),--overlap "$(CHUNK_OVERLAP)",) \
		$(if $(EMBEDDING_MODEL),--embedding-model "$(EMBEDDING_MODEL)",) \
		$(if $(RETRIEVAL_MODE),--retrieval-mode "$(RETRIEVAL_MODE)",) \
		$(if $(KEEP_DUPLICATE_CHUNKS),--keep-duplicate-chunks,) \
		$(if $(DUPLICATE_TIER),--duplicate-tier "$(DUPLICATE_TIER)",) \
		$(if $(LEMMATIZE),--lemmatize,)

check-generation: ## Validate every registered member of GENERATION (GENERATION_KIND=store|graph|prompt-system)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main check-generation "$(GENERATION)" --kind "$(or $(GENERATION_KIND),store)"

measure-duplicate-residue: ## What repetition a built store still holds after collapse (STORE= or CONFIG=; RESIDUE_THRESHOLDS= RESIDUE_EXAMPLES= RESIDUE_OUT=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main measure-duplicate-residue $(if $(CONFIG),--config "$(CONFIG)",) \
		$(if $(STORE),--store "$(STORE)",) \
		$(if $(RESIDUE_THRESHOLDS),--thresholds "$(RESIDUE_THRESHOLDS)",) \
		$(if $(RESIDUE_EXAMPLES),--examples $(RESIDUE_EXAMPLES),) \
		$(if $(RESIDUE_OUT),--out "$(RESIDUE_OUT)",)

build-graph: ## GraphRAG backend: build the GraphRAG store from an ontology-assisted draft bundle (BUNDLE=... CONFIG=...; needs ".[graph]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<prepare-goldset dir> (extraction.jsonl + corpus/)"; exit 1; }
	$(PY) -m llb.main build-graph $(if $(CONFIG),--config "$(CONFIG)",) --bundle "$(BUNDLE)"

resolve-graph-entities: ## Entity resolution: propose a graph node-cluster overlay and price it on the graph lane (GOLDSET= SPLIT= RAG_K= RESOLVE_THRESHOLDS= RESOLVE_STRATEGIES= RESOLVE_NO_EMBEDDINGS=1 RESOLVE_WITH_VECTOR=1 CORPUS= RESOLVE_OUT_DIR=; needs ".[linkage]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main resolve-graph-entities $(if $(CONFIG),--config "$(CONFIG)",) \
		--goldset "$(GOLDSET)" --k $(RAG_K) $(if $(SPLIT),--split "$(SPLIT)",) \
		$(if $(RESOLVE_THRESHOLDS),--thresholds "$(RESOLVE_THRESHOLDS)",) \
		$(if $(RESOLVE_STRATEGIES),--strategies "$(RESOLVE_STRATEGIES)",) \
		$(if $(RESOLVE_NO_EMBEDDINGS),--no-mention-embeddings,) \
		$(if $(RESOLVE_WITH_VECTOR),--with-vector $(if $(CORPUS),--corpus-root "$(CORPUS)",),) \
		$(if $(FUSION_BOOTSTRAP_RESAMPLES),--resamples $(FUSION_BOOTSTRAP_RESAMPLES),) \
		$(if $(RESOLVE_OUT_DIR),--out-dir "$(RESOLVE_OUT_DIR)",)

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
		$(BUILD_INDEX_CORPUS) \
		--goldset "$(GOLDSET)" --k $(RAG_K) $(if $(SPLIT),--split "$(SPLIT)",) \
		$(if $(RETRIEVAL_BACKEND),--retrieval-backend "$(RETRIEVAL_BACKEND)",) \
		$(if $(RETRIEVAL_STRATEGY),--retrieval-strategy "$(RETRIEVAL_STRATEGY)",) \
		$(if $(GRAPH_WEIGHT),--graph-weight $(GRAPH_WEIGHT),) \
		$(if $(QUERY_PREP),--query-prep "$(QUERY_PREP)",) \
		$(if $(QUERY_GLOSSARY),--query-glossary "$(QUERY_GLOSSARY)",) \
		$(if $(QUERY_PREP_TYPO_GUARD),--query-prep-typo-guard,) \
		$(if $(QUERY_PREP_DENSE_CASE),--query-prep-dense-case,) \
		$(if $(QUERY_PREP_MODEL),--query-prep-model "$(QUERY_PREP_MODEL)",) \
		$(if $(QUERY_PREP_BACKEND),--query-prep-backend "$(QUERY_PREP_BACKEND)",) \
		$(if $(QUERY_PREP_AB),--query-prep-ab,) \
		$(if $(QUERY_PREP_OUT),--out "$(QUERY_PREP_OUT)",)
