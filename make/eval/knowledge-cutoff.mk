## Knowledge-cutoff evaluation and Ukrainian bilingual calibration.

.PHONY: bench-knowledge-cutoff knowledge-cutoff-ua-draft knowledge-cutoff-ua-review \
	knowledge-cutoff-ua-revise knowledge-cutoff-ua-confirm-accepted \
	knowledge-cutoff-ua-validate knowledge-cutoff-ua-freeze \
	bench-knowledge-cutoff-bilingual knowledge-cutoff-bilingual

bench-knowledge-cutoff: ## Estimate a local model's real knowledge cutoff and write JSON/Markdown reports (MODEL= BACKEND=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-knowledge-cutoff --model "$(MODEL)" --backend "$(BACKEND)" \
		--dataset-id "$(KNOWLEDGE_CUTOFF_DATASET)" \
		--dataset-revision "$(KNOWLEDGE_CUTOFF_REVISION)" \
		--threshold "$(KNOWLEDGE_CUTOFF_THRESHOLD)" \
		--optuna-trials "$(KNOWLEDGE_CUTOFF_TRIALS)" --seed "$(KNOWLEDGE_CUTOFF_SEED)" \
		$(if $(KNOWLEDGE_CUTOFF_EVENTS),--events "$(KNOWLEDGE_CUTOFF_EVENTS)",) \
		$(if $(KNOWLEDGE_CUTOFF_LIMIT),--limit "$(KNOWLEDGE_CUTOFF_LIMIT)",) \
		$(if $(KNOWLEDGE_CUTOFF_BASE_URL),--base-url "$(KNOWLEDGE_CUTOFF_BASE_URL)",) \
		$(if $(KNOWLEDGE_CUTOFF_MAX_MODEL_LEN),--max-model-len "$(KNOWLEDGE_CUTOFF_MAX_MODEL_LEN)",)

knowledge-cutoff-ua-draft: ## Draft/resume the pinned Ukrainian translation bundle (KNOWLEDGE_CUTOFF_REVISION=<commit> KNOWLEDGE_CUTOFF_UA_TRANSLATOR_MODEL=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main knowledge-cutoff-ua-draft \
		--translator-model "$(KNOWLEDGE_CUTOFF_UA_TRANSLATOR_MODEL)" \
		--backend "$(KNOWLEDGE_CUTOFF_UA_TRANSLATOR_BACKEND)" \
		--dataset-id "$(KNOWLEDGE_CUTOFF_DATASET)" \
		--dataset-revision "$(KNOWLEDGE_CUTOFF_REVISION)" \
		--out-dir "$(KNOWLEDGE_CUTOFF_UA_BUNDLE)" \
		--gpu-memory-utilization "$(KNOWLEDGE_CUTOFF_UA_TRANSLATOR_GPU_MEMORY)" \
		$(if $(KNOWLEDGE_CUTOFF_EVENTS),--events "$(KNOWLEDGE_CUTOFF_EVENTS)",) \
		$(if $(KNOWLEDGE_CUTOFF_LIMIT),--limit "$(KNOWLEDGE_CUTOFF_LIMIT)",) \
		$(if $(KNOWLEDGE_CUTOFF_UA_TRANSLATOR_BASE_URL),--base-url "$(KNOWLEDGE_CUTOFF_UA_TRANSLATOR_BASE_URL)",) \
		$(if $(KNOWLEDGE_CUTOFF_MAX_MODEL_LEN),--max-model-len "$(KNOWLEDGE_CUTOFF_MAX_MODEL_LEN)",)

knowledge-cutoff-ua-review: ## Resume bilingual review of every translation (KNOWLEDGE_CUTOFF_UA_BUNDLE= START=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main knowledge-cutoff-ua-review --bundle "$(KNOWLEDGE_CUTOFF_UA_BUNDLE)" \
		$(if $(START),--start "$(START)",)

knowledge-cutoff-ua-revise: ## Apply JSONL translation corrections and rerun automatic gates (KNOWLEDGE_CUTOFF_UA_REVISIONS=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(KNOWLEDGE_CUTOFF_UA_REVISIONS)" || { echo "ERROR: set KNOWLEDGE_CUTOFF_UA_REVISIONS=<revisions.jsonl>"; exit 1; }
	$(PY) -m llb.main knowledge-cutoff-ua-revise --bundle "$(KNOWLEDGE_CUTOFF_UA_BUNDLE)" \
		--revisions "$(KNOWLEDGE_CUTOFF_UA_REVISIONS)"

knowledge-cutoff-ua-confirm-accepted: ## Confirm aggregate accepts imply all four translation checks (KNOWLEDGE_CUTOFF_UA_BUNDLE=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main knowledge-cutoff-ua-confirm-accepted \
		--bundle "$(KNOWLEDGE_CUTOFF_UA_BUNDLE)"

knowledge-cutoff-ua-validate: ## Validate translation alignment and report review progress (KNOWLEDGE_CUTOFF_UA_BUNDLE=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main knowledge-cutoff-ua-validate --bundle "$(KNOWLEDGE_CUTOFF_UA_BUNDLE)"

knowledge-cutoff-ua-freeze: ## Freeze accepted translations after complete bilingual sign-off (KNOWLEDGE_CUTOFF_UA_BUNDLE= KNOWLEDGE_CUTOFF_UA_REVIEWER=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(KNOWLEDGE_CUTOFF_UA_REVIEWER)" || { echo "ERROR: set KNOWLEDGE_CUTOFF_UA_REVIEWER=<name-or-id>"; exit 1; }
	$(PY) -m llb.main knowledge-cutoff-ua-freeze --bundle "$(KNOWLEDGE_CUTOFF_UA_BUNDLE)" \
		--reviewer "$(KNOWLEDGE_CUTOFF_UA_REVIEWER)"

bench-knowledge-cutoff-bilingual: ## Run one model on a frozen paired EN/UK translation bundle (MODEL= BACKEND= KNOWLEDGE_CUTOFF_UA_BUNDLE=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-knowledge-cutoff-bilingual \
		--bundle "$(KNOWLEDGE_CUTOFF_UA_BUNDLE)" --model "$(MODEL)" --backend "$(BACKEND)" \
		--threshold "$(KNOWLEDGE_CUTOFF_THRESHOLD)" \
		--optuna-trials "$(KNOWLEDGE_CUTOFF_TRIALS)" --seed "$(KNOWLEDGE_CUTOFF_SEED)" \
		--gpu-memory-utilization "$(KNOWLEDGE_CUTOFF_BILINGUAL_GPU_MEMORY)" \
		$(if $(KNOWLEDGE_CUTOFF_BASE_URL),--base-url "$(KNOWLEDGE_CUTOFF_BASE_URL)",) \
		$(if $(KNOWLEDGE_CUTOFF_MAX_MODEL_LEN),--max-model-len "$(KNOWLEDGE_CUTOFF_MAX_MODEL_LEN)",)

knowledge-cutoff-bilingual: ## Full resumable draft -> bilingual review -> freeze -> paired run workflow
	$(MAKE) --no-print-directory knowledge-cutoff-ua-draft
	$(MAKE) --no-print-directory knowledge-cutoff-ua-review
	$(MAKE) --no-print-directory knowledge-cutoff-ua-validate
	$(MAKE) --no-print-directory knowledge-cutoff-ua-freeze
	$(MAKE) --no-print-directory bench-knowledge-cutoff-bilingual
