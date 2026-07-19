## External scoring, isolated sweeps, and end-to-end evaluation orchestration.

.PHONY: score-external-rag sweep pipeline joint-search auto-rag

auto-rag: ## Autonomous corpus -> verified goldset -> joint tune -> RAG recommendation (CORPUS= SCORER_POLICY=auto|human)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -d "$(CORPUS)" || { echo "ERROR: set CORPUS=<mixed-corpus-dir>"; exit 1; }
	@set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main auto-rag --corpus "$(CORPUS)" \
		--draft-model "$(AUTO_RAG_DRAFT_MODEL)" --candidates "$(AUTO_RAG_CANDIDATES)" \
		--scorer-policy "$(SCORER_POLICY)" --max-items "$(AUTO_RAG_MAX_ITEMS)" \
		--trials "$(AUTO_RAG_TRIALS)" --screen-limit "$(AUTO_RAG_SCREEN_LIMIT)" \
		--min-finalists "$(AUTO_RAG_MIN_FINALISTS)" \
		--max-model-len "$(AUTO_RAG_MAX_MODEL_LEN)" \
		$(if $(AUTO_RAG_RUN_ID),--run-id "$(AUTO_RAG_RUN_ID)",) \
		$(if $(AUTO_RAG_CANDIDATE_MODELS),--candidate-models "$(AUTO_RAG_CANDIDATE_MODELS)",) \
		$(if $(AUTO_RAG_DOC_LIMIT),--doc-limit "$(AUTO_RAG_DOC_LIMIT)",) \
		$(if $(AUTO_RAG_EVAL_LIMIT),--eval-limit "$(AUTO_RAG_EVAL_LIMIT)",) \
		$(if $(AUTO_RAG_JUDGE_MODEL),--judge-model "$(AUTO_RAG_JUDGE_MODEL)",) \
		$(if $(AUTO_RAG_JUDGE_BASE_URL),--judge-base-url "$(AUTO_RAG_JUDGE_BASE_URL)",) \
		$(if $(SCORER_EGRESS_CONSENT),--egress-consent,) \
		$(if $(SCORER_MAX_USD),--max-usd "$(SCORER_MAX_USD)",) \
		$(if $(SCORER_MAX_CALLS),--max-calls "$(SCORER_MAX_CALLS)",) \
		$(if $(AUTO_RAG_PARITY_CHECK),--parity-check,)

score-external-rag: ## Human-score answered external RAG JSONL; final CSV/report after all rows are scored (EXTERNAL_RAG_ANSWERS=, EXTERNAL_RAG_SOURCE_MAP=<provider-to-doc_id sidecar> for the source-span audit)
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
		$(if $(EXTERNAL_RAG_KEEP_SOURCE_FOOTER),--keep-source-footer,) \
		$(if $(EXTERNAL_RAG_SOURCE_MAP),--source-map "$(EXTERNAL_RAG_SOURCE_MAP)",)

sweep: ## Run isolated candidate sweep (SWEEP_ID= MODELS_MANIFEST= SPLIT= GOLDSET= SWEEP_LIMIT= SWEEP_RAG_GRID=top_k=3,5,8; RERANKER= for positive rerank_candidates points)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main sweep --manifest "$(MODELS_MANIFEST)" --split "$(SPLIT)" \
		--goldset "$(GOLDSET)" --sweep-id "$(SWEEP_ID)" \
		--max-model-len "$(SWEEP_MAX_MODEL_LEN)" $(if $(SWEEP_OFFLINE),--offline,) \
		$(if $(SWEEP_LIMIT),--limit "$(SWEEP_LIMIT)",) \
		$(if $(SWEEP_RAG_GRID),--rag-grid "$(SWEEP_RAG_GRID)",) \
		$(if $(RERANKER),--reranker "$(RERANKER)",)

pipeline: ## Select public-screen finalists, tune, and print the final board
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main pipeline --manifest "$(MODELS_MANIFEST)" --goldset "$(GOLDSET)" \
		--top-n "$(PIPELINE_TOP_N)" --trials "$(PIPELINE_TRIALS)" \
		$(if $(PIPELINE_OFFLINE),--offline,)

joint-search: ## Successive-halving model+RAG search (JOINT_SEARCH_CANDIDATES= JOINT_SEARCH_TRIALS= JOINT_SEARCH_SCREEN_LIMIT=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main joint-search --candidates "$(JOINT_SEARCH_CANDIDATES)" \
		--goldset "$(GOLDSET)" --trials "$(JOINT_SEARCH_TRIALS)" \
		--screen-limit "$(JOINT_SEARCH_SCREEN_LIMIT)" \
		--min-finalists "$(JOINT_SEARCH_MIN_FINALISTS)" \
		--objectives "$(JOINT_SEARCH_OBJECTIVES)" \
		$(if $(JOINT_SEARCH_RUN_ID),--run-id "$(JOINT_SEARCH_RUN_ID)",) \
		$(if $(JOINT_SEARCH_OFFLINE),--offline,) \
		$(if $(JOINT_SEARCH_CORPUS),--corpus "$(JOINT_SEARCH_CORPUS)",) \
		$(if $(JOINT_SEARCH_LIMIT),--limit "$(JOINT_SEARCH_LIMIT)",) \
		$(if $(JOINT_SEARCH_NO_ISOLATE),--no-isolate,)
