## Goldset ingestion and ontology-assisted draft generation.

.PHONY: ingest-uk-squad prepare-goldset-draft

ingest-uk-squad: ## Development utility: GOLDSET_MODE=development|skeleton|draft (draft is robust backend prep)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@echo "[ingest-uk-squad] mode=$(GOLDSET_MODE)"; \
	case "$(GOLDSET_MODE)" in \
	  development) \
	    set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	    $(PY) -m llb.prep.ingest_squad --pinned-development-source \
	      --max-items $(GOLDSET_N) \
	      --out-name goldset_uk_development.jsonl ;; \
	  skeleton) \
	    $(PY) -m llb.prep.goldset_skeleton ;; \
	  draft) \
	    set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	    $(MAKE) --no-print-directory prepare-goldset-draft DRAFT_CORPUS="$(CORPUS)" ;; \
	  *) \
	    echo "ERROR: GOLDSET_MODE must be development, skeleton, or draft" >&2; exit 2 ;; \
	esac

prepare-goldset-draft: ## Ontology-assisted draft bundle; use DRAFT_DOC_LIMIT=1 for PDF probe
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	if [ "$(origin DRAFT_RESUME)" = "command line" ] && [ -z "$(strip $(DRAFT_RESUME))" ]; then \
	  echo "ERROR: DRAFT_RESUME is empty; set the shell variable or pass the bundle path" >&2; exit 2; \
	fi; \
	args=( \
	  --corpus-root "$(DRAFT_CORPUS)" \
	  --model "$(if $(DRAFT_FRONTIER_MODEL),$(DRAFT_FRONTIER_MODEL),$(DRAFT_MODEL))" \
	  --endpoint "$(DRAFT_ENDPOINT)" \
	  --frontier-stage "$(DRAFT_FRONTIER_STAGE)" \
	  --backend "$(DRAFT_BACKEND)" \
	  --max-items "$(DRAFT_MAX_ITEMS)" \
	  --extractor "$(DRAFT_EXTRACTOR)" \
	  --max-tokens "$(DRAFT_MAX_TOKENS)" \
	  --temperature "$(DRAFT_TEMPERATURE)" \
	  --timeout "$(DRAFT_TIMEOUT)" \
	  --verification-sample-size "$(DRAFT_VERIFY_N)" \
	); \
	if [ -n "$(DRAFT_BASE_URL)" ]; then args+=(--base-url "$(DRAFT_BASE_URL)"); fi; \
	if [ -n "$(DRAFT_LOCAL_MODEL)" ]; then args+=(--local-model "$(DRAFT_LOCAL_MODEL)"); fi; \
	if [ "$(DRAFT_ENDPOINT)" = "frontier" ] && [ -n "$(DRAFT_MAX_USD)" ]; then args+=(--max-usd "$(DRAFT_MAX_USD)"); fi; \
	if [ "$(DRAFT_ENDPOINT)" = "frontier" ] && [ -n "$(DRAFT_MAX_CALLS)" ]; then args+=(--max-calls "$(DRAFT_MAX_CALLS)"); fi; \
	if [ "$(DRAFT_ENDPOINT)" = "frontier" ] && [ "$(DRAFT_EGRESS_CONSENT)" = "1" ]; then args+=(--egress-consent); fi; \
	if [ -n "$(DRAFT_VLLM_PORT)" ]; then args+=(--vllm-port "$(DRAFT_VLLM_PORT)"); fi; \
	if [ -n "$(DRAFT_VLLM_GPU_MEMORY_UTILIZATION)" ]; then args+=(--vllm-gpu-memory-utilization "$(DRAFT_VLLM_GPU_MEMORY_UTILIZATION)"); fi; \
	if [ -n "$(DRAFT_VLLM_MAX_MODEL_LEN)" ]; then args+=(--vllm-max-model-len "$(DRAFT_VLLM_MAX_MODEL_LEN)"); fi; \
	if [ -n "$(DRAFT_VLLM_CPU_OFFLOAD_GB)" ]; then args+=(--vllm-cpu-offload-gb "$(DRAFT_VLLM_CPU_OFFLOAD_GB)"); fi; \
	if [ -n "$(DRAFT_VLLM_KV_OFFLOADING_SIZE_GB)" ]; then args+=(--vllm-kv-offloading-size-gb "$(DRAFT_VLLM_KV_OFFLOADING_SIZE_GB)"); fi; \
	if [ -n "$(DRAFT_VLLM_DTYPE)" ]; then args+=(--vllm-dtype "$(DRAFT_VLLM_DTYPE)"); fi; \
	if [ -n "$(DRAFT_VLLM_QUANTIZATION)" ]; then args+=(--vllm-quantization "$(DRAFT_VLLM_QUANTIZATION)"); fi; \
	if [ -n "$(DRAFT_VLLM_STARTUP_TIMEOUT)" ]; then args+=(--vllm-startup-timeout "$(DRAFT_VLLM_STARTUP_TIMEOUT)"); fi; \
	if [ -n "$(DRAFT_DOC_LIMIT)" ]; then args+=(--doc-limit "$(DRAFT_DOC_LIMIT)"); fi; \
	if [ -n "$(DRAFT_EXTRACT_MAX_CHARS)" ]; then args+=(--extract-max-chars "$(DRAFT_EXTRACT_MAX_CHARS)"); fi; \
	if [ -n "$(DRAFT_EXTRACT_CHUNK_OVERLAP)" ]; then args+=(--extract-chunk-overlap "$(DRAFT_EXTRACT_CHUNK_OVERLAP)"); fi; \
	if [ -n "$(DRAFT_CONCURRENCY)" ]; then args+=(--concurrency "$(DRAFT_CONCURRENCY)"); fi; \
	if [ -n "$(DRAFT_OUT_DIR)" ]; then args+=(--out-dir "$(DRAFT_OUT_DIR)"); fi; \
	if [ -n "$(DRAFT_RESUME)" ]; then args+=(--resume "$(DRAFT_RESUME)"); fi; \
	if [ -n "$(DRAFT_RETRIEVAL_INDEX_DIR)" ]; then args+=(--retrieval-index-dir "$(DRAFT_RETRIEVAL_INDEX_DIR)" --retrieval-k "$(DRAFT_RETRIEVAL_K)"); fi; \
	if [ "$(DRAFT_DROP_NONRETRIEVABLE_NEEDLES)" = "1" ]; then args+=(--drop-nonretrievable-needles); fi; \
	if [ "$(DRAFT_REQUIRE_PASSED_GATES)" = "1" ]; then args+=(--require-passed-gates); fi; \
	if [ -n "$(DRAFT_COVERAGE_TARGET)" ]; then args+=(--coverage-target "$(DRAFT_COVERAGE_TARGET)"); fi; \
	if [ "$(DRAFT_MULTI_HOP)" = "1" ]; then args+=(--multi-hop); fi; \
	if [ "$(DRAFT_CHAINS)" = "1" ]; then args+=(--chains); fi; \
	if [ -n "$(DRAFT_MULTI_HOP_MAX_PATHS)" ]; then args+=(--multi-hop-max-paths "$(DRAFT_MULTI_HOP_MAX_PATHS)"); fi; \
	if [ "$(DRAFT_MULTI_HOP_BRIDGE_FILL)" = "1" ]; then args+=(--multi-hop-bridge-fill); fi; \
	if [ -n "$(DRAFT_DEDUP_AGAINST)" ]; then args+=(--dedup-against "$(DRAFT_DEDUP_AGAINST)"); fi; \
	if [ -n "$(DRAFT_GRAPH_DIR)" ]; then args+=(--graph-dir "$(DRAFT_GRAPH_DIR)"); fi; \
	if [ -n "$(DRAFT_REJECTION_FEEDBACK)" ]; then args+=(--rejection-feedback "$(DRAFT_REJECTION_FEEDBACK)"); fi; \
	if [ "$(DRAFT_NO_THINK)" = "1" ]; then args+=(--no-think); fi; \
	if [ -n "$(DRAFT_NUM_CTX)" ]; then args+=(--num-ctx "$(DRAFT_NUM_CTX)"); fi; \
	$(PY) -m llb.main prepare-goldset-draft "$${args[@]}"
