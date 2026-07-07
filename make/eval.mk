# RAG evaluation, prompt-system, benchmark, and pipeline targets.
##@ Evaluation and Pipelines

.PHONY: \
	build-rag-store build-index build-graph validate-retrieval compare-retrieval \
	compare-embeddings run-eval sweep pipeline prompt-system-prepare prompt-system-review \
	prompt-system-compare bench-security bench-agentic agentic-harness-compare \
	composite-headline platform-matrix

build-rag-store: ## Chunk a corpus with all strategies into DATA_DIR/llb/rag (CORPUS_DIR=...)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.rag.chunking --corpus-root "$(CORPUS_DIR)" \
		--out-dir "$(DATA_DIR)/llb/rag" --strategy all --size 800 --overlap 120

build-index: ## RAG core: chunk + embed CORPUS into the FAISS store (EMBEDDING_MODEL= to override; needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main build-index --corpus-root "$(CORPUS)" \
		$(if $(EMBEDDING_MODEL),--embedding-model "$(EMBEDDING_MODEL)",)

build-graph: ## GraphRAG backend: build the GraphRAG store from an ontology-assisted draft bundle (BUNDLE=...; needs ".[graph]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<prepare-goldset dir> (extraction.jsonl + corpus/)"; exit 1; }
	$(PY) -m llb.main build-graph --bundle "$(BUNDLE)"

validate-retrieval: ## RAG core: recall@k / MRR of the pinned embedding over the gold set (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main validate-retrieval --goldset "$(GOLDSET)" --k $(RAG_K)

compare-retrieval: ## GraphRAG backend: compare faiss vs both graph strategies' recall@k/MRR on the gold set
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main compare-retrieval --goldset "$(GOLDSET)" --k $(RAG_K)

compare-embeddings: ## embedding-bakeoff-uk: rank UA embedders (recall@k/MRR + throughput) on GOLDSET; MODELS= EMBED_API_MODEL= (needs ".[rag]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main compare-embeddings --goldset "$(GOLDSET)" --k $(RAG_K) \
		$(if $(MODELS),--models "$(MODELS)",) \
		$(if $(EMBED_API_MODEL),--api-model "$(EMBED_API_MODEL)" --data-classification "$(EMBED_DATA_CLASSIFICATION)" $(if $(EMBED_MAX_USD),--max-usd $(EMBED_MAX_USD),),)

run-eval: ## Run the eval; MODEL= BACKEND= GOLDSET= SPLIT= PROMPT_SYSTEM_ID= PROMPT_PACKAGE= RESUME=<run-dir>
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main run-eval --model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --split "$(SPLIT)" \
		--limit $(LIMIT) $(if $(TELEMETRY),--telemetry) \
		$(if $(RESUME),--resume "$(RESUME)",) \
		$(if $(PROMPT_SYSTEM_ID),--prompt-system "$(PROMPT_SYSTEM_ID)",) \
		$(if $(PROMPT_PACKAGE),--prompt-package "$(PROMPT_PACKAGE)",) \
		$(if $(JUDGE_RHO),--judge-rho $(JUDGE_RHO) --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)"))

sweep: ## Run isolated candidate sweep (SWEEP_ID= MODELS_MANIFEST= SPLIT= GOLDSET= SWEEP_LIMIT= SWEEP_RAG_GRID=top_k=3,5,8)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main sweep --manifest "$(MODELS_MANIFEST)" --split "$(SPLIT)" \
		--goldset "$(GOLDSET)" --sweep-id "$(SWEEP_ID)" \
		--max-model-len "$(SWEEP_MAX_MODEL_LEN)" $(if $(SWEEP_OFFLINE),--offline,) \
		$(if $(SWEEP_LIMIT),--limit "$(SWEEP_LIMIT)",) \
		$(if $(SWEEP_RAG_GRID),--rag-grid "$(SWEEP_RAG_GRID)",)

pipeline: ## Select public-screen finalists, tune, and print the final board
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main pipeline --manifest "$(MODELS_MANIFEST)" --goldset "$(GOLDSET)" \
		--top-n "$(PIPELINE_TOP_N)" --trials "$(PIPELINE_TRIALS)" \
		$(if $(PIPELINE_OFFLINE),--offline,)

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
		$(if $(PROMPT_SYSTEM_INSTRUCTION),--instruction "$(PROMPT_SYSTEM_INSTRUCTION)",)

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

bench-agentic: ## Run one agentic harness cell (AGENTIC_HARNESS=loop|langgraph|crewai)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic --tasks "$(AGENTIC_TASKS)" \
		--model "$(MODEL)" --backend "$(BACKEND)" --max-steps "$(AGENTIC_MAX_STEPS)" \
		--harness "$(AGENTIC_HARNESS)" \
		$(if $(AGENTIC_BASE_URL),--base-url "$(AGENTIC_BASE_URL)",) \
		$(if $(JUDGE_RHO),--judge-rho "$(JUDGE_RHO)" --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",),)

agentic-harness-compare: ## Run loop/langgraph/crewai agentic cells, then compare harnesses
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@for harness in $(AGENTIC_HARNESSES); do \
		$(MAKE) --no-print-directory bench-agentic AGENTIC_HARNESS="$$harness" || exit 1; \
	done
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-compare --model "$(MODEL)"

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
