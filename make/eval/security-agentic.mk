## Security and agentic benchmark entrypoints.

.PHONY: bench-security bench-security-derived bench-agentic

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

bench-security-derived: ## One-command human-gated derived flow: scaffold worksheet -> interactive review -> VERIFIED bench-security (SECURITY_DERIVE_CASES=, SECURITY_MODEL=, SECURITY_BACKEND=, SECURITY_BASE_URL=, SECURITY_DERIVE_WORKSHEET=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	ws="$(SECURITY_DERIVE_WORKSHEET)"; [ -n "$$ws" ] || ws="$(DATA_DIR)/security-derive/verify_sample.csv"; \
	if [ -f "$$ws" ]; then \
		echo "[bench-security-derived] reusing worksheet $$ws (review resumes at first undecided row)"; \
	else \
		$(PY) -m llb.main derive-security-worksheet --cases "$(SECURITY_DERIVE_CASES)" --out "$$ws" || exit 1; \
	fi; \
	echo "[bench-security-derived] opening the shared review UI (y=accept, x=reject, q=save+quit) ..."; \
	$(PY) -m llb.goldset.verify review --worksheet "$$ws" || exit 1; \
	echo "[bench-security-derived] review done -- running the VERIFIED scored bench ..."; \
	$(PY) -m llb.main bench-security --cases "$(SECURITY_DERIVE_CASES)" \
		--model "$(SECURITY_MODEL)" --backend "$(SECURITY_BACKEND)" \
		$(if $(SECURITY_BASE_URL),--base-url "$(SECURITY_BASE_URL)",) \
		$(if $(SECURITY_MAX_MODEL_LEN),--max-model-len "$(SECURITY_MAX_MODEL_LEN)",) \
		--data-verified --verification-ref "$$ws"

bench-agentic: ## Run one agentic harness cell (AGENTIC_HARNESS=loop|langgraph|crewai)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic --tasks "$(AGENTIC_TASKS)" \
		--model "$(MODEL)" --backend "$(BACKEND)" --max-steps "$(AGENTIC_MAX_STEPS)" \
		--harness "$(AGENTIC_HARNESS)" \
		$(if $(AGENTIC_BASE_URL),--base-url "$(AGENTIC_BASE_URL)",) \
		$(if $(JUDGE_RHO),--judge-rho "$(JUDGE_RHO)" --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",),)
