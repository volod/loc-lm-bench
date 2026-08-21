COMPARE_RETRIEVAL_GOLDSET_ARG = $(if $(CONFIG),$(if $(filter command line environment override,$(origin GOLDSET)),$(if $(GOLDSET),--goldset "$(GOLDSET)",)),--goldset "$(GOLDSET)")
COMPARE_RETRIEVAL_SPLIT_ARG = $(if $(CONFIG),$(if $(filter command line environment override,$(origin SPLIT)),$(if $(SPLIT),--split "$(SPLIT)",)),$(if $(SPLIT),--split "$(SPLIT)",))

compare-retrieval: ## Compare retrieval with paired evidence; RETRIEVAL_BASELINE= RETRIEVAL_RESAMPLES= RETRIEVAL_CONFIDENCE= control uncertainty; CHUNK_STRATEGIES=..., HYBRID=1, RERANKER=, NOISE_FLOOR=1 are optional lanes
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main compare-retrieval $(if $(CONFIG),--config "$(CONFIG)",) \
		$(COMPARE_RETRIEVAL_GOLDSET_ARG) --k $(RAG_K) $(COMPARE_RETRIEVAL_SPLIT_ARG) \
		$(if $(CHUNK_STRATEGIES),--strategies "$(CHUNK_STRATEGIES)",) \
		$(if $(HYBRID),--hybrid,) \
		$(if $(FUSION_WEIGHT),--fusion-weight $(FUSION_WEIGHT),) \
		$(if $(GRAPH_WEIGHT),--graph-weight $(GRAPH_WEIGHT),) \
		$(if $(RERANKER),--reranker "$(RERANKER)",) \
		$(if $(RERANK_CANDIDATES),--rerank-candidates $(RERANK_CANDIDATES),) \
		$(if $(DUPLICATE_TIER),--duplicate-tier "$(DUPLICATE_TIER)",) \
		$(if $(NOISE_FLOOR),--noise-floor,) \
		$(if $(NOISE_FLOOR_REPLICATES),--noise-floor-replicates $(NOISE_FLOOR_REPLICATES),) \
		$(if $(RETRIEVAL_BASELINE),--baseline "$(RETRIEVAL_BASELINE)",) \
		$(if $(RETRIEVAL_RESAMPLES),--resamples $(RETRIEVAL_RESAMPLES),) \
		$(if $(RETRIEVAL_CONFIDENCE),--confidence $(RETRIEVAL_CONFIDENCE),) \
		$(if $(RETRIEVAL_SEED),--seed $(RETRIEVAL_SEED),) \
		$(if $(COMPARE_RETRIEVAL_OUT),--out "$(COMPARE_RETRIEVAL_OUT)",)

compare-graph-fusion: ## Sweep graph fusion with paired evidence (GOLDSET= GRAPH_WEIGHTS= ROUTED_GRAPH_WEIGHT= GRAPH_FUSION_CANDIDATES= GRAPH_FUSION_SPAN_IDENTITY=exact,overlap GRAPH_FUSION_SPAN_MERGE_RATIO=0.25,0.5,1.0 GRAPH_STRATEGIES= FUSION_FOCUS_SLICE= FUSION_POWER_REFERENCE= FUSION_POWER_ROW= FUSION_MDE= FUSION_POWER_METRIC= FUSION_TARGET_POWER= FUSION_HIDE_ROUTING_SIDECAR=1 NOISE_FLOOR=1 FUSION_OUT_DIR=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main compare-graph-fusion $(if $(CONFIG),--config "$(CONFIG)",) \
		--goldset "$(GOLDSET)" --k $(RAG_K) $(if $(SPLIT),--split "$(SPLIT)",) \
		$(if $(GRAPH_WEIGHTS),--graph-weights "$(GRAPH_WEIGHTS)",) \
		$(if $(ROUTED_GRAPH_WEIGHT),--routed-graph-weight "$(ROUTED_GRAPH_WEIGHT)",) \
		$(if $(FUSION_HIDE_ROUTING_SIDECAR),--no-routing-sidecar,) \
		$(if $(FUSION_HEURISTIC_LONG_QUESTION_WORDS),--heuristic-long-question-words $(FUSION_HEURISTIC_LONG_QUESTION_WORDS),) \
		$(if $(FUSION_HEURISTIC_MIN_LINKED_ENTITIES),--heuristic-min-linked-entities $(FUSION_HEURISTIC_MIN_LINKED_ENTITIES),) \
		$(if $(FUSION_POWER_REFERENCE),--power-reference "$(FUSION_POWER_REFERENCE)",) \
		$(if $(FUSION_POWER_ROW),--power-row "$(FUSION_POWER_ROW)",) \
		$(if $(FUSION_POWER_METRIC),--power-metric "$(FUSION_POWER_METRIC)",) \
		$(if $(FUSION_MDE),--minimum-detectable-delta "$(FUSION_MDE)",) \
		$(if $(FUSION_TARGET_POWER),--target-power "$(FUSION_TARGET_POWER)",) \
		$(if $(GRAPH_FUSION_CANDIDATES),--graph-fusion-candidates "$(GRAPH_FUSION_CANDIDATES)",) \
		$(if $(GRAPH_FUSION_SPAN_IDENTITY),--graph-fusion-span-identity "$(GRAPH_FUSION_SPAN_IDENTITY)",) \
		$(if $(GRAPH_FUSION_SPAN_MERGE_RATIO),--graph-fusion-span-merge-ratio "$(GRAPH_FUSION_SPAN_MERGE_RATIO)",) \
		$(if $(GRAPH_STRATEGIES),--graph-strategies "$(GRAPH_STRATEGIES)",) \
		$(if $(FUSION_FOCUS_SLICE),--focus-slice "$(FUSION_FOCUS_SLICE)",) \
		$(if $(NOISE_FLOOR),--noise-floor,) \
		$(if $(NOISE_FLOOR_REPLICATES),--noise-floor-replicates $(NOISE_FLOOR_REPLICATES),) \
		$(if $(FUSION_BOOTSTRAP_RESAMPLES),--resamples $(FUSION_BOOTSTRAP_RESAMPLES),) \
		$(if $(FUSION_OUT_DIR),--out-dir "$(FUSION_OUT_DIR)",)

probe-multihop-hops: ## Diagnose/convert a stuck all-spans@k (GOLDSET= SPLIT= HOP_PROBE_BUDGETS=10,25,50 QUERY_PREP= QUERY_PREP_MODEL= QUERY_PREP_BACKEND=ollama HOP_PROBE_OUT_DIR=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main probe-multihop-hops $(if $(CONFIG),--config "$(CONFIG)",) \
		--goldset "$(GOLDSET)" $(if $(SPLIT),--split "$(SPLIT)",) \
		$(if $(HOP_PROBE_BUDGETS),--budgets "$(HOP_PROBE_BUDGETS)",) \
		$(if $(HOP_PROBE_DEPTH),--probe-depth $(HOP_PROBE_DEPTH),) \
		$(if $(HOP_PROBE_BACKEND),--retrieval-backend "$(HOP_PROBE_BACKEND)",) \
		$(if $(HOP_PROBE_STRATEGY),--retrieval-strategy "$(HOP_PROBE_STRATEGY)",) \
		$(if $(QUERY_PREP),--query-prep "$(QUERY_PREP)",) \
		$(if $(QUERY_PREP_MODEL),--query-prep-model "$(QUERY_PREP_MODEL)",) \
		$(if $(QUERY_PREP_BACKEND),--query-prep-backend "$(QUERY_PREP_BACKEND)",) \
		$(if $(FUSION_FOCUS_SLICE),--focus-slice "$(FUSION_FOCUS_SLICE)",) \
		$(if $(FUSION_BOOTSTRAP_RESAMPLES),--resamples $(FUSION_BOOTSTRAP_RESAMPLES),) \
		$(if $(HOP_PROBE_OUT_DIR),--out-dir "$(HOP_PROBE_OUT_DIR)",)

calibrate-fusion-routing: ## Tune sidecar-free routing thresholds, freeze on tuning, and score held-out final (GOLDSET= ROUTING_LONG_WORD_GRID= ROUTING_ENTITY_GRID= ROUTING_TUNING_SPLIT= ROUTING_FINAL_SPLIT= ROUTING_OUT_DIR=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main calibrate-fusion-routing $(if $(CONFIG),--config "$(CONFIG)",) \
		--goldset "$(GOLDSET)" --k $(RAG_K) \
		--tuning-split "$(ROUTING_TUNING_SPLIT)" --final-split "$(ROUTING_FINAL_SPLIT)" \
		--long-question-words "$(ROUTING_LONG_WORD_GRID)" \
		--min-linked-entities "$(ROUTING_ENTITY_GRID)" \
		--graph-strategy "$(ROUTING_GRAPH_STRATEGY)" \
		--graph-weight $(ROUTING_GRAPH_WEIGHT) \
		--candidates $(ROUTING_CANDIDATES) --span-identity "$(ROUTING_SPAN_IDENTITY)" \
		$(if $(FUSION_BOOTSTRAP_RESAMPLES),--resamples $(FUSION_BOOTSTRAP_RESAMPLES),) \
		$(if $(ROUTING_OUT_DIR),--out-dir "$(ROUTING_OUT_DIR)",)

compare-answer-quality: ## Score the multi-hop slice end to end under two retrieval lanes and compare ANSWERS (MODEL= BACKEND= GOLDSET= SPLIT=a,b ANSWER_QUALITY_LANES= FUSION_COMPARISON= ANSWER_QUALITY_BUDGETS=10,50 FUSION_FOCUS_SLICE= INCLUDE_DRAFTED=1 ANSWER_QUALITY_OUT_DIR=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main compare-answer-quality $(if $(CONFIG),--config "$(CONFIG)",) \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --split "$(SPLIT)" \
		$(if $(ANSWER_QUALITY_LANES),--lanes "$(ANSWER_QUALITY_LANES)",) \
		$(if $(FUSION_COMPARISON),--from-comparison "$(FUSION_COMPARISON)",) \
		$(if $(ANSWER_QUALITY_BUDGETS),--budgets "$(ANSWER_QUALITY_BUDGETS)",) \
		$(if $(FUSION_FOCUS_SLICE),--focus-slice "$(FUSION_FOCUS_SLICE)",) \
		$(if $(FUSION_BOOTSTRAP_RESAMPLES),--resamples $(FUSION_BOOTSTRAP_RESAMPLES),) \
		$(if $(ANSWER_QUALITY_LIMIT),--limit $(ANSWER_QUALITY_LIMIT),) \
		$(if $(INCLUDE_DRAFTED),--include-drafted,) \
		$(if $(ANSWER_QUALITY_OUT_DIR),--out-dir "$(ANSWER_QUALITY_OUT_DIR)",)

compare-context-strategies: ## Does RAG pay for itself? Score one item set closed-book vs rag vs long-context (MODEL= BACKEND= GOLDSET= CORPUS= SPLIT=a,b CONTEXT_LANES= CONTEXT_ABLATION_LIMIT= CONTEXT_POWER_REFERENCE= CONTEXT_MDE= CONTEXT_TARGET_POWER= INCLUDE_DRAFTED=1 CONTEXT_ABLATION_OUT_DIR=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main compare-context-strategies $(if $(CONFIG),--config "$(CONFIG)",) \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --split "$(SPLIT)" \
		$(if $(CORPUS),--corpus "$(CORPUS)",) \
		$(if $(CONTEXT_LANES),--lanes "$(CONTEXT_LANES)",) \
		$(if $(FUSION_BOOTSTRAP_RESAMPLES),--resamples $(FUSION_BOOTSTRAP_RESAMPLES),) \
		$(if $(CONTEXT_ABLATION_LIMIT),--limit $(CONTEXT_ABLATION_LIMIT),) \
		$(if $(CONTEXT_POWER_REFERENCE),--power-reference "$(CONTEXT_POWER_REFERENCE)",) \
		$(if $(CONTEXT_MDE),--minimum-detectable-delta "$(CONTEXT_MDE)",) \
		$(if $(CONTEXT_TARGET_POWER),--target-power "$(CONTEXT_TARGET_POWER)",) \
		$(if $(INCLUDE_DRAFTED),--include-drafted,) \
		$(if $(CONTEXT_ABLATION_OUT_DIR),--out-dir "$(CONTEXT_ABLATION_OUT_DIR)",)

# A config owns its goldset and split unless the operator explicitly overrides either on the make
# command line. Without this distinction the repository-wide fixture/default-final values silently
# replace a config's recorded corpus and item family.
COMPARE_EMBEDDINGS_GOLDSET_ARG = $(if $(CONFIG),$(if $(filter command line environment override,$(origin GOLDSET)),$(if $(GOLDSET),--goldset "$(GOLDSET)",)),--goldset "$(GOLDSET)")
COMPARE_EMBEDDINGS_SPLIT_ARG = $(if $(CONFIG),$(if $(filter command line environment override,$(origin SPLIT)),$(if $(SPLIT),--split "$(SPLIT)",)),$(if $(SPLIT),--split "$(SPLIT)",))

# The interpreter the bake-off lanes run in. `compare-*-legacy` overrides it per target rather than
# re-entering make: `SPLIT` is exported, so in a sub-make its origin becomes `environment` and the
# config-owns-its-split rule above would silently flip. One recipe, one origin, two interpreters.
BAKEOFF_PY ?= $(PY)

compare-embeddings: ## Rank UA embedders with paired evidence (CONFIG= or GOLDSET=; MODELS= EMBED_BASELINE= EMBED_POWER_REFERENCE= EMBED_POWER_CANDIDATE= EMBED_MDE= EMBED_POWER_METRIC= EMBED_TARGET_POWER= EMBED_API_MODEL= EMBED_ADOPTION_BARS=recall_at_k[,mrr] EMBED_ALLOW_REMOTE_CODE=1 EMBED_DTYPE=float32 NOISE_FLOOR=1 EMBED_RESAMPLES= EMBED_ENCODER_THROUGHPUT=1; needs ".[rag]")
	@test -x "$(BAKEOFF_PY)" || { echo "ERROR: $(BAKEOFF_PY) missing -- run 'make venv' first"; exit 1; }
	$(BAKEOFF_PY) -m llb.main compare-embeddings $(if $(CONFIG),--config "$(CONFIG)",) \
		$(COMPARE_EMBEDDINGS_GOLDSET_ARG) --k $(RAG_K) $(COMPARE_EMBEDDINGS_SPLIT_ARG) \
		$(if $(MODELS),--models "$(MODELS)",) \
		$(if $(EMBED_BASELINE),--baseline "$(EMBED_BASELINE)",) \
		$(if $(EMBED_ADOPTION_BARS),--adoption-bars "$(EMBED_ADOPTION_BARS)",) \
		$(if $(filter 1,$(EMBED_ALLOW_REMOTE_CODE)),--allow-remote-code,) \
		$(if $(EMBED_DTYPE),--encoder-dtype "$(EMBED_DTYPE)",) \
		$(if $(EMBED_RESAMPLES),--resamples $(EMBED_RESAMPLES),) \
		$(if $(EMBED_CONFIDENCE),--confidence $(EMBED_CONFIDENCE),) \
		$(if $(EMBED_POWER_REFERENCE),--power-reference "$(EMBED_POWER_REFERENCE)",) \
		$(if $(EMBED_POWER_CANDIDATE),--power-candidate "$(EMBED_POWER_CANDIDATE)",) \
		$(if $(EMBED_POWER_METRIC),--power-metric "$(EMBED_POWER_METRIC)",) \
		$(if $(EMBED_MDE),--minimum-detectable-delta "$(EMBED_MDE)",) \
		$(if $(EMBED_TARGET_POWER),--target-power "$(EMBED_TARGET_POWER)",) \
		$(if $(NOISE_FLOOR),--noise-floor,) \
		$(if $(NOISE_FLOOR_REPLICATES),--noise-floor-replicates $(NOISE_FLOOR_REPLICATES),) \
		$(if $(COMPARE_EMBEDDINGS_OUT),--out "$(COMPARE_EMBEDDINGS_OUT)",) \
		$(if $(EMBED_ENCODER_THROUGHPUT),--encoder-throughput \
			--encoder-precision "$(EMBED_ENCODER_PRECISION)" \
			--encoder-min-warm "$(EMBED_ENCODER_MIN_WARM)" \
			--encoder-max-warm "$(EMBED_ENCODER_MAX_WARM)" \
			--encoder-max-warm-seconds "$(EMBED_ENCODER_MAX_WARM_SECONDS)" \
			$(if $(filter 1,$(EMBED_ENCODER_COMPARE_CPU)),--encoder-compare-cpu,),) \
		$(if $(EMBED_API_MODEL),--api-model "$(EMBED_API_MODEL)" --data-classification "$(EMBED_DATA_CLASSIFICATION)" $(if $(EMBED_MAX_USD),--max-usd $(EMBED_MAX_USD),),)

COMPARE_RERANKERS_GOLDSET_ARG = $(if $(CONFIG),$(if $(filter command line environment override,$(origin GOLDSET)),$(if $(GOLDSET),--goldset "$(GOLDSET)",)),--goldset "$(GOLDSET)")
COMPARE_RERANKERS_SPLIT_ARG = $(if $(CONFIG),$(if $(filter command line environment override,$(origin SPLIT)),$(if $(SPLIT),--split "$(SPLIT)",)),$(if $(SPLIT),--split "$(SPLIT)",))

compare-rerankers: ## Rank cross-encoder rerankers on one pool with paired evidence + cost (CONFIG= or GOLDSET=; CORPUS= RERANK_MODELS= RERANK_BASELINE= RERANK_CANDIDATES= RERANK_ADOPTION_BARS=recall_at_k[,mrr] RERANK_ALLOW_REMOTE_CODE=1 RERANK_GENERATOR_VRAM_MB= RERANK_BATCH_SIZE= RERANK_DTYPE= NOISE_FLOOR=1; needs ".[rag]")
	@test -x "$(BAKEOFF_PY)" || { echo "ERROR: $(BAKEOFF_PY) missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(BAKEOFF_PY) -m llb.main compare-rerankers $(if $(CONFIG),--config "$(CONFIG)",) \
		$(COMPARE_RERANKERS_GOLDSET_ARG) --k $(RAG_K) $(COMPARE_RERANKERS_SPLIT_ARG) \
		$(if $(CORPUS),--corpus-root "$(CORPUS)",) \
		$(if $(RERANK_MODELS),--models "$(RERANK_MODELS)",) \
		$(if $(RERANK_CANDIDATES),--rerank-candidates $(RERANK_CANDIDATES),) \
		$(if $(RERANK_BASELINE),--baseline "$(RERANK_BASELINE)",) \
		$(if $(RERANK_ADOPTION_BARS),--adoption-bars "$(RERANK_ADOPTION_BARS)",) \
		$(if $(filter 1,$(RERANK_ALLOW_REMOTE_CODE)),--allow-remote-code,) \
		$(if $(RERANK_GENERATOR_VRAM_MB),--generator-vram-mb $(RERANK_GENERATOR_VRAM_MB),) \
		$(if $(RERANK_BATCH_SIZE),--batch-size $(RERANK_BATCH_SIZE),) \
		$(if $(RERANK_DTYPE),--dtype "$(RERANK_DTYPE)",) \
		$(if $(RERANK_RESAMPLES),--resamples $(RERANK_RESAMPLES),) \
		$(if $(RERANK_CONFIDENCE),--confidence $(RERANK_CONFIDENCE),) \
		$(if $(NOISE_FLOOR),--noise-floor,) \
		$(if $(NOISE_FLOOR_REPLICATES),--noise-floor-replicates $(NOISE_FLOOR_REPLICATES),) \
		$(if $(COMPARE_RERANKERS_OUT),--out "$(COMPARE_RERANKERS_OUT)",)

# The LEGACY scoring pass. Four roster candidates ship repository code written against the
# transformers 4.x API, so on the pinned 5.x stack they are screened out with the pin they need
# (src/llb/rag/encoders/model_stack.py). These two targets run the SAME recipe as their siblings above with
# `BAKEOFF_PY` pointed at the `[encoders-legacy]` virtualenv under $DATA_DIR, which is where those
# rows can be scored. Every other variable behaves exactly as it does on the pinned target.
ENCODERS_LEGACY_PY = $(DATA_DIR)/venvs/encoders-legacy/bin/python
LEGACY_ENCODER_ROSTER = $(EMBED_BASELINE),Alibaba-NLP/gte-multilingual-base,jinaai/jina-embeddings-v3
LEGACY_RERANKER_ROSTER = $(RERANK_BASELINE),jinaai/jina-reranker-v2-base-multilingual,Alibaba-NLP/gte-multilingual-reranker-base

venv-encoders-legacy: ## Create/refresh the legacy encoder venv (transformers<5) under $DATA_DIR
	bash "$(PROJECT_ROOT)/scripts/setup_encoders_legacy_venv.sh"

# The roster defaults name the incumbent BESIDE the legacy candidates on purpose: a pass that
# scored only the unrunnable rows would have no baseline to pair them against, and the incumbent
# reproducing its pinned-stack numbers here is what says the two passes are comparable at all.
# `LEGACY_MODELS=` overrides that roster; `MODELS=` / `RERANK_MODELS=` are set by the target itself,
# so an operator naming them on the command line would be overridden without noticing.
compare-embeddings-legacy: BAKEOFF_PY = $(ENCODERS_LEGACY_PY)
compare-embeddings-legacy: EMBED_ALLOW_REMOTE_CODE = 1
compare-embeddings-legacy: MODELS = $(if $(LEGACY_MODELS),$(LEGACY_MODELS),$(LEGACY_ENCODER_ROSTER))
compare-embeddings-legacy: venv-encoders-legacy compare-embeddings ## compare-embeddings for the transformers 4.x remote-code encoders (same vars, but LEGACY_MODELS= overrides the roster; defaults to the incumbent + those candidates)
	@:

compare-rerankers-legacy: BAKEOFF_PY = $(ENCODERS_LEGACY_PY)
compare-rerankers-legacy: RERANK_ALLOW_REMOTE_CODE = 1
compare-rerankers-legacy: RERANK_MODELS = $(if $(LEGACY_MODELS),$(LEGACY_MODELS),$(LEGACY_RERANKER_ROSTER))
compare-rerankers-legacy: venv-encoders-legacy compare-rerankers ## compare-rerankers for the transformers 4.x remote-code rerankers (same vars, but LEGACY_MODELS= overrides the roster; defaults to the incumbent + those candidates)
	@:

compare-embedder-adoption: ## Does an embedder's FIRST-HIT-RANK gain reach the answer? Sweep top_k x reranker end to end on two encoders (MODEL= BACKEND= GOLDSET= SPLIT=a,b EMBED_BASELINE= EMBED_BASELINE_DATA_DIR= EMBED_CANDIDATE= EMBED_CANDIDATE_DATA_DIR= ADOPTION_TOP_KS=10,3 ADOPTION_RERANKERS=off,on ADOPTION_LIMIT= INCLUDE_DRAFTED=1 ADOPTION_OUT_DIR=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(EMBED_BASELINE_DATA_DIR)" || { echo "ERROR: set EMBED_BASELINE_DATA_DIR=<root whose llb/rag store was built with EMBED_BASELINE>"; exit 1; }
	@test -n "$(EMBED_CANDIDATE_DATA_DIR)" || { echo "ERROR: set EMBED_CANDIDATE_DATA_DIR=<root whose llb/rag store was built with EMBED_CANDIDATE>"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main compare-embedder-adoption $(if $(CONFIG),--config "$(CONFIG)",) \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --split "$(SPLIT)" \
		$(if $(CORPUS),--corpus "$(CORPUS)",) \
		--baseline-embedder "$(EMBED_BASELINE)" --baseline-data-dir "$(EMBED_BASELINE_DATA_DIR)" \
		--candidate-embedder "$(EMBED_CANDIDATE)" --candidate-data-dir "$(EMBED_CANDIDATE_DATA_DIR)" \
		$(if $(ADOPTION_TOP_KS),--top-ks "$(ADOPTION_TOP_KS)",) \
		$(if $(ADOPTION_RERANKERS),--rerankers "$(ADOPTION_RERANKERS)",) \
		$(if $(FUSION_BOOTSTRAP_RESAMPLES),--resamples $(FUSION_BOOTSTRAP_RESAMPLES),) \
		$(if $(ADOPTION_LIMIT),--limit $(ADOPTION_LIMIT),) \
		$(if $(INCLUDE_DRAFTED),--include-drafted,) \
		$(if $(ADOPTION_OUT_DIR),--out-dir "$(ADOPTION_OUT_DIR)",)

compare-adoption-models: ## Do two models agree on the first-hit-rank reading? Compare two finished sweeps cell by cell (ADOPTION_REPORT_A= ADOPTION_REPORT_B= ADOPTION_CROSS_OUT=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(ADOPTION_REPORT_A)" || { echo "ERROR: set ADOPTION_REPORT_A=<first sweep comparison.json or dir>"; exit 1; }
	@test -n "$(ADOPTION_REPORT_B)" || { echo "ERROR: set ADOPTION_REPORT_B=<second sweep comparison.json or dir>"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main compare-adoption-models "$(ADOPTION_REPORT_A)" "$(ADOPTION_REPORT_B)" \
		$(if $(ADOPTION_CROSS_OUT),--out-dir "$(ADOPTION_CROSS_OUT)",)

compare-adoption-roster: ## Is the reranker gain predictable in advance? Test whether a declared model property separates the models that capture it (ADOPTION_REPORTS="<dir> <dir> ..." ADOPTION_PROFILES= ADOPTION_FOCUS_CELL= ADOPTION_ROSTER_OUT=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(ADOPTION_REPORTS)" || { echo "ERROR: set ADOPTION_REPORTS=\"<sweep-dir> <sweep-dir> <sweep-dir> ...\" (3+)"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main compare-adoption-roster $(ADOPTION_REPORTS) \
		$(if $(ADOPTION_PROFILES),--profiles "$(ADOPTION_PROFILES)",) \
		$(if $(ADOPTION_FOCUS_CELL),--focus-cell "$(ADOPTION_FOCUS_CELL)",) \
		$(if $(ADOPTION_ROSTER_OUT),--out-dir "$(ADOPTION_ROSTER_OUT)",)

compare-adoption-screen: ## What does deciding the reranker question for ONE model cost? Resample recorded sweeps for the cheapest screen (ADOPTION_REPORTS="<dir> ..." ADOPTION_FOCUS_CELL= ADOPTION_SCREEN_SIZES= ADOPTION_SCREEN_DRAWS= ADOPTION_SCREEN_TARGET= ADOPTION_SCREEN_OUT=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(ADOPTION_REPORTS)" || { echo "ERROR: set ADOPTION_REPORTS=\"<sweep-dir> ...\""; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main compare-adoption-screen $(ADOPTION_REPORTS) \
		$(if $(ADOPTION_FOCUS_CELL),--focus-cell "$(ADOPTION_FOCUS_CELL)",) \
		$(if $(ADOPTION_SCREEN_SIZES),--sizes "$(ADOPTION_SCREEN_SIZES)",) \
		$(if $(ADOPTION_SCREEN_DRAWS),--draws $(ADOPTION_SCREEN_DRAWS),) \
		$(if $(ADOPTION_SCREEN_TARGET),--target $(ADOPTION_SCREEN_TARGET),) \
		$(if $(ADOPTION_SCREEN_OUT),--out-dir "$(ADOPTION_SCREEN_OUT)",)

# With CONFIG= the YAML owns the gold set and the split, so the defaults are forwarded only when
# the caller actually set them -- otherwise a config-targeted backend comparison would silently
# score the DEFAULT goldset against the config's corpus. Same rule as compare-embeddings.
COMPARE_STORES_GOLDSET_ARG = $(if $(CONFIG),$(if $(filter command line environment override,$(origin GOLDSET)),$(if $(GOLDSET),--goldset "$(GOLDSET)",)),--goldset "$(GOLDSET)")
COMPARE_STORES_SPLIT_ARG = $(if $(CONFIG),$(if $(filter command line environment override,$(origin SPLIT)),$(if $(SPLIT),--split "$(SPLIT)",)),$(if $(SPLIT),--split "$(SPLIT)",))

compare-vector-stores: ## platform matrix: rank vector backends (FAISS/Chroma/Qdrant/LanceDB) with paired evidence (CONFIG= or GOLDSET=; VECTOR_BACKENDS= VECTOR_BASELINE= VECTOR_RESAMPLES= VECTOR_CONFIDENCE= VECTOR_SEED= NOISE_FLOOR=1 NOISE_FLOOR_REPLICATES= COMPARE_STORES_OUT=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main compare-vector-stores $(if $(CONFIG),--config "$(CONFIG)",) \
		$(COMPARE_STORES_GOLDSET_ARG) --k $(RAG_K) $(COMPARE_STORES_SPLIT_ARG) \
		$(if $(VECTOR_BACKENDS),--backends "$(VECTOR_BACKENDS)",) \
		$(if $(VECTOR_BASELINE),--baseline "$(VECTOR_BASELINE)",) \
		$(if $(VECTOR_RESAMPLES),--resamples $(VECTOR_RESAMPLES),) \
		$(if $(VECTOR_CONFIDENCE),--confidence $(VECTOR_CONFIDENCE),) \
		$(if $(VECTOR_SEED),--seed $(VECTOR_SEED),) \
		$(if $(NOISE_FLOOR),--noise-floor,) \
		$(if $(NOISE_FLOOR_REPLICATES),--noise-floor-replicates $(NOISE_FLOOR_REPLICATES),) \
		$(if $(COMPARE_STORES_OUT),--out "$(COMPARE_STORES_OUT)",)
PAIRED_READING_AUDIT_OUT ?=

.PHONY: audit-paired-readings
audit-paired-readings: ## Re-read selected grid verdicts with calibrated and family-wise inference
	$(PY) -m llb.main audit-paired-readings \
		$(if $(PAIRED_READING_AUDIT_OUT),--out-dir "$(PAIRED_READING_AUDIT_OUT)",)
