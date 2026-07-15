## Cross-harness, category-composite, and platform-matrix evaluation.

.PHONY: agentic-harness-compare bench-chain-context composite-headline platform-matrix

agentic-harness-compare: ## Run loop/langgraph/crewai agentic cells, then compare harnesses
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@for harness in $(AGENTIC_HARNESSES); do \
		$(MAKE) --no-print-directory bench-agentic AGENTIC_HARNESS="$$harness" || exit 1; \
	done
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-compare --model "$(MODEL)"

bench-chain-context: ## Context-policy benchmark: rank fresh/history/summary/roles for one model over a verified chain set (CHAIN_CONTEXT_MODEL= CHAIN_CONTEXT_BACKEND= CHAIN_CONTEXT_CHAINS= CHAIN_CONTEXT_CORPUS= CHAIN_CONTEXT_POLICIES=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-chain-context --chains "$(CHAIN_CONTEXT_CHAINS)" \
		--model "$(CHAIN_CONTEXT_MODEL)" --backend "$(CHAIN_CONTEXT_BACKEND)" \
		--corpus "$(CHAIN_CONTEXT_CORPUS)" --policies "$(CHAIN_CONTEXT_POLICIES)" \
		--top-k "$(CHAIN_CONTEXT_TOP_K)" \
		$(if $(CHAIN_CONTEXT_INDEX_DIR),--index-dir "$(CHAIN_CONTEXT_INDEX_DIR)",) \
		$(if $(CHAIN_CONTEXT_BASE_URL),--base-url "$(CHAIN_CONTEXT_BASE_URL)",) \
		$(if $(CHAIN_CONTEXT_MAX_MODEL_LEN),--max-model-len "$(CHAIN_CONTEXT_MAX_MODEL_LEN)",) \
		$(if $(filter 1 true yes,$(CHAIN_CONTEXT_DATA_VERIFIED)),--data-verified,) \
		$(if $(CHAIN_CONTEXT_VERIFICATION_REF),--verification-ref "$(CHAIN_CONTEXT_VERIFICATION_REF)",)

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
