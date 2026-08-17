run-eval: ## Run the eval; MODEL= BACKEND= GOLDSET= SPLIT= RETRIEVAL_BACKEND=fused GRAPH_WEIGHT=0.3 RETRIEVAL_MODE=hybrid ACL_LABEL=tag RERANKER= CONTEXT_ORDER= CONTEXT_STRATEGY= QUERY_PREP=... RESUME=<run-dir>
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main run-eval $(if $(CONFIG),--config "$(CONFIG)",) \
		$(BUILD_INDEX_CORPUS) \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --split "$(SPLIT)" \
		$(if $(RETRIEVAL_BACKEND),--retrieval-backend "$(RETRIEVAL_BACKEND)",) \
		$(if $(RETRIEVAL_STRATEGY),--retrieval-strategy "$(RETRIEVAL_STRATEGY)",) \
		$(if $(RETRIEVAL_MODE),--retrieval-mode "$(RETRIEVAL_MODE)",) \
		$(if $(ACL_LABEL),--acl "$(ACL_LABEL)",) \
		$(if $(FUSION_WEIGHT),--fusion-weight $(FUSION_WEIGHT),) \
		$(if $(GRAPH_WEIGHT),--graph-weight $(GRAPH_WEIGHT),) \
		$(if $(GRAPH_FUSION_CANDIDATES),--graph-fusion-candidates $(GRAPH_FUSION_CANDIDATES),) \
		$(if $(RERANKER),--reranker "$(RERANKER)",) \
		$(if $(RERANK_CANDIDATES),--rerank-candidates $(RERANK_CANDIDATES),) \
		$(if $(CONTEXT_ORDER),--context-order "$(CONTEXT_ORDER)",) \
		$(if $(CONTEXT_STRATEGY),--context-strategy "$(CONTEXT_STRATEGY)",) \
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

analyze-verbosity: ## Compare fixed-item RAG bundles under F1, recall, found-rate, and declared format policy (RUN_DIRS="...")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(RUN_DIRS)" || { echo "ERROR: set RUN_DIRS='<bundle> <bundle> ...'"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main analyze-verbosity \
		$(foreach dir,$(RUN_DIRS),--run-dir "$(dir)") \
		$(if $(VERBOSITY_OUT),--out-dir "$(VERBOSITY_OUT)",)

bench-query-robustness: ## Noisy/language queries vs clean RAG and mitigation lanes (MODEL= BACKEND= GOLDSET= CORPUS= SPLIT= QUERY_ROBUSTNESS_LIMIT= QUERY_ROBUSTNESS_CLASSES= LANGUAGE_FIXTURE=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-query-robustness --model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --corpus-root "$(CORPUS)" --split "$(SPLIT)" \
		--top-k $(RAG_K) --typo-rate $(QUERY_ROBUSTNESS_TYPO_RATE) \
		--max-tokens $(QUERY_ROBUSTNESS_MAX_TOKENS) \
		$(if $(QUERY_ROBUSTNESS_CLASSES),--variant-classes "$(QUERY_ROBUSTNESS_CLASSES)",) \
		$(if $(LANGUAGE_FIXTURE),--language-fixture "$(LANGUAGE_FIXTURE)",) \
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
