# Model preparation and serving-target targets.
##@ Models and Serving

.PHONY: \
	build-vllm build-llamacpp prep-models prep-serving-targets list-models \
	detect-gpu-vram gen-serving-config

build-vllm: ## Install prebuilt vLLM via uv; VLLM_SOURCE_DIR= builds/caches one checkout wheel
	bash "$(PROJECT_ROOT)/scripts/build_vllm.sh"

build-llamacpp: ## Build CUDA llama-server for the llama.cpp launcher; CUDA_ARCH=/LLAMACPP_REF= override
	bash "$(PROJECT_ROOT)/scripts/build_llamacpp.sh"

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

detect-gpu-vram: ## Print supported GPU VRAM tier (12/16/24/32 GiB) from nvidia-smi
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main detect-gpu-vram

gen-serving-config: ## Emit serve + run-eval artifacts under .data/llb/serving/; GPU_GB=12|16|24|32 overrides detect
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main gen-serving-config $(if $(GPU_GB),--gpu-gb $(GPU_GB),)
