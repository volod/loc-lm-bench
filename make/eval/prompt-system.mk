## Prompt-system candidate preparation, review, and comparison.

.PHONY: prompt-system-prepare prompt-system-review prompt-system-compare

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
		$(if $(PROMPT_SYSTEM_INSTRUCTION),--instruction "$(PROMPT_SYSTEM_INSTRUCTION)",) \
		$(if $(PROMPT_SYSTEM_ONTOLOGY_BUNDLE),--ontology-bundle "$(PROMPT_SYSTEM_ONTOLOGY_BUNDLE)",) \
		$(if $(PROMPT_SYSTEM_GRAPH_DIR),--graph-dir "$(PROMPT_SYSTEM_GRAPH_DIR)",) \
		--tree-depths "$(PROMPT_SYSTEM_TREE_DEPTHS)" \
		--tree-budgets "$(PROMPT_SYSTEM_TREE_BUDGETS)"

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
