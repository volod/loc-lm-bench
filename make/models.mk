# Model preparation and serving-target targets.
##@ Models and Serving

.PHONY: \
	build-vllm build-llamacpp download-model prep-models prep-serving-targets list-models \
	detect-gpu-vram gen-serving-config list-model-families sync-model-family-docs \
	lint-model-roster measure-throughput

build-vllm: ## Install prebuilt vLLM via uv; VLLM_SOURCE_DIR= builds/caches one checkout wheel
	bash "$(PROJECT_ROOT)/scripts/build_vllm.sh"

build-llamacpp: ## Build CUDA llama-server for the llama.cpp launcher; CUDA_ARCH=/LLAMACPP_REF= override
	bash "$(PROJECT_ROOT)/scripts/build_llamacpp.sh"

download-model: ## Download a huge model with resume/checksums; MODEL_DOWNLOAD_ID=/TARGET= required
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(MODEL_DOWNLOAD_ID)" || { echo "ERROR: set MODEL_DOWNLOAD_ID=<model-id>"; exit 1; }
	@test -n "$(MODEL_DOWNLOAD_TARGET)" || { echo "ERROR: set MODEL_DOWNLOAD_TARGET=<local-dir>"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; \
	$(PY) -m llb.main download-model "$(MODEL_DOWNLOAD_ID)" "$(MODEL_DOWNLOAD_TARGET)" \
		--provider "$(MODEL_DOWNLOAD_PROVIDER)" --chunk-mib "$(MODEL_DOWNLOAD_CHUNK_MIB)" \
		--session-gib "$(MODEL_DOWNLOAD_SESSION_GIB)" \
		--bandwidth-fraction "$(MODEL_DOWNLOAD_BANDWIDTH_FRACTION)" \
		--timeout-seconds "$(MODEL_DOWNLOAD_TIMEOUT_SECONDS)" \
		--retries "$(MODEL_DOWNLOAD_RETRIES)" \
		--max-rate-wait-seconds "$(MODEL_DOWNLOAD_MAX_RATE_WAIT_SECONDS)" \
		--min-free-gib "$(MODEL_DOWNLOAD_MIN_FREE_GIB)" \
		--min-free-percent "$(MODEL_DOWNLOAD_MIN_FREE_PERCENT)" \
		$(if $(MODEL_DOWNLOAD_REVISION),--revision "$(MODEL_DOWNLOAD_REVISION)",) \
		$(if $(MODEL_DOWNLOAD_MAX_MIBPS),--max-mib-per-second "$(MODEL_DOWNLOAD_MAX_MIBPS)",) \
		$(if $(MODEL_DOWNLOAD_VERIFY_COMPLETED),--verify-completed,) \
		$(if $(MODEL_DOWNLOAD_VERIFY_ONLY),--verify-only,) \
		$(if $(MODEL_DOWNLOAD_DRY_RUN),--dry-run,)

prep-models: ## Detect GPU, pull Ollama tags + cache vLLM HF weights (MODELS_MANIFEST=, PREP_BACKEND=, gated needs HF_TOKEN)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main prep-models --manifest "$(MODELS_MANIFEST)" --backend "$(PREP_BACKEND)"

prep-serving-targets: ## Pull/cache models referenced by generated serving tier.json (SERVING_TIER_JSON=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(SERVING_TIER_JSON)" || { echo "ERROR: set SERVING_TIER_JSON=<llb/serving/gpu-*/tier.json>"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main prep-serving-targets --tier-json "$(SERVING_TIER_JSON)" --backend "$(PREP_BACKEND)"

list-models: ## List which candidate models can run here (GPU+RAM, KV-cache-aware); CONTEXT= to target a context
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main list-models --manifest "$(MODELS_MANIFEST)" $(if $(CONTEXT),--context $(CONTEXT),)

measure-throughput: ## Measure roster entries under the committed throughput protocol; MODELS=name[,name] (default: all)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main measure-throughput --manifest "$(MODELS_MANIFEST)" \
		$(if $(MODELS),--models "$(MODELS)",) $(if $(CONTEXT),--context $(CONTEXT),) \
		$(if $(THROUGHPUT_BACKEND),--backend "$(THROUGHPUT_BACKEND)",) \
		$(if $(THROUGHPUT_SOURCE),--source "$(THROUGHPUT_SOURCE)",)

list-model-families: ## Print the family register: generations, status, licenses, models carried
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main list-model-families --manifest "$(MODELS_MANIFEST)" $(if $(MARKDOWN),--markdown,)

sync-model-family-docs: ## Republish the generated family tables in README + docs/reference
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.quality.roster_docs --manifest "$(MODELS_MANIFEST)"

lint-model-roster: ## Check the generated family tables still match the roster manifest (in ci-checks)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.quality.roster_docs --manifest "$(MODELS_MANIFEST)" --check

detect-gpu-vram: ## Print supported GPU VRAM tier (12/16/24/32 GiB) from nvidia-smi
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main detect-gpu-vram

gen-serving-config: ## Emit serve + run-eval artifacts under .data/llb/serving/; GPU_GB=12|16|24|32 overrides detect
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main gen-serving-config $(if $(GPU_GB),--gpu-gb $(GPU_GB),)
