## Fine-tuning datasets, training loops, distillation, and adapter lifecycle.

.PHONY: export-finetune-set finetune-adapter finetune-hparams self-improve finetune-campaign \
	distill register-adapter list-adapters serve-adapter gc-adapters

export-finetune-set: ## Export tuning-split SFT/DPO records (RUN_DIR=<tuning-run> GOLDSET= OUT_DIR= MISSES=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(RUN_DIR)" || { echo "ERROR: set RUN_DIR=<tuning run-eval bundle dir>"; exit 1; }
	@test -n "$(OUT_DIR)" || { echo "ERROR: set OUT_DIR=<dataset dir>"; exit 1; }
	$(PY) -m llb.main export-finetune-set --run-dir "$(RUN_DIR)" --goldset "$(GOLDSET)" \
		--out "$(OUT_DIR)" $(if $(MISSES),--misses "$(MISSES)",)

finetune-adapter: ## Train a LoRA/QLoRA adapter (DATASET=<export dir> MODEL=<base> ADAPTER_OUT= TRAINER=auto|fake)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(DATASET)" || { echo "ERROR: set DATASET=<export-finetune-set dir>"; exit 1; }
	@test -n "$(MODEL)" || { echo "ERROR: set MODEL=<base model>"; exit 1; }
	$(PY) -m llb.main finetune-adapter --dataset "$(DATASET)" --model "$(MODEL)" \
		$(if $(ADAPTER_OUT),--out "$(ADAPTER_OUT)",) $(if $(TRAINER),--trainer "$(TRAINER)",)

finetune-hparams: ## Budgeted LoRA hparam search on a tuning-split dev slice (MODEL= DATASET= GOLDSET= MAX_TRIALS=8 MAX_HOURS= TRAINER=auto|fake HPARAMS_RESUME= HPARAMS_STRATIFY_RUN=<scored base run> HPARAMS_VRAM_HEADROOM=<MiB>)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(MODEL)" || { echo "ERROR: set MODEL=<base model>"; exit 1; }
	@test -n "$(DATASET)" || { echo "ERROR: set DATASET=<export-finetune-set dir>"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main finetune-hparams --model "$(MODEL)" --dataset "$(DATASET)" \
		--max-trials "$(or $(MAX_TRIALS),8)" \
		$(if $(BACKEND),--backend "$(BACKEND)",) \
		$(if $(GOLDSET),--goldset "$(GOLDSET)",) \
		$(if $(MAX_HOURS),--max-hours "$(MAX_HOURS)",) \
		$(if $(HPARAMS_SEED),--seed "$(HPARAMS_SEED)",) \
		$(if $(DEV_FRACTION),--dev-fraction "$(DEV_FRACTION)",) \
		$(if $(HPARAMS_OUT),--out-dir "$(HPARAMS_OUT)",) \
		$(if $(HPARAMS_RESUME),--resume "$(HPARAMS_RESUME)",) \
		$(if $(HPARAMS_STRATIFY_RUN),--stratify-by-base-score "$(HPARAMS_STRATIFY_RUN)",) \
		$(if $(HPARAMS_VRAM_HEADROOM),--vram-headroom-mib "$(HPARAMS_VRAM_HEADROOM)",) \
		$(if $(TRAINER),--trainer "$(TRAINER)",)

self-improve: ## Local self-improvement loop (MODEL= BACKEND= GOLDSET= ROUNDS=2 LIMIT= TRAINER=auto|fake)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main self-improve --model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --rounds "$(ROUNDS)" \
		$(if $(LIMIT),--limit "$(LIMIT)",) \
		$(if $(SELF_IMPROVE_OUT),--out-dir "$(SELF_IMPROVE_OUT)",) \
		$(if $(SELF_IMPROVE_RESUME),--resume "$(SELF_IMPROVE_RESUME)",) \
		$(if $(TRAINER),--trainer "$(TRAINER)",)

finetune-campaign: ## Multi-model adapter campaign (MODELS=<csv> BACKEND= GOLDSET= ROUNDS=1 TRAINER=auto|fake)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main finetune-campaign --models "$(or $(MODELS),$(FINETUNE_CAMPAIGN_MODELS))" \
		--backend "$(BACKEND)" --goldset "$(GOLDSET)" --corpus "$(CORPUS)" \
		--rounds "$(or $(ROUNDS),$(FINETUNE_CAMPAIGN_ROUNDS))" \
		$(if $(FINETUNE_CAMPAIGN_LIMIT),--limit "$(FINETUNE_CAMPAIGN_LIMIT)",) \
		$(if $(FINETUNE_CAMPAIGN_OUT),--out-dir "$(FINETUNE_CAMPAIGN_OUT)",) \
		$(if $(FINETUNE_CAMPAIGN_RESUME),--resume "$(FINETUNE_CAMPAIGN_RESUME)",) \
		$(if $(FINETUNE_CAMPAIGN_MANIFEST),--manifest "$(FINETUNE_CAMPAIGN_MANIFEST)",) \
		$(if $(TRAINER),--trainer "$(TRAINER)",)

distill: ## Local teacher -> student adapter distillation (TEACHER= STUDENT= BACKEND= GOLDSET= GATE=0.8 TRAINER=auto|fake)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(TEACHER)" || { echo "ERROR: set TEACHER=<teacher model>"; exit 1; }
	@test -n "$(STUDENT)" || { echo "ERROR: set STUDENT=<student model>"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main distill --teacher "$(TEACHER)" --student "$(STUDENT)" \
		--backend "$(BACKEND)" --goldset "$(GOLDSET)" --corpus "$(CORPUS)" \
		--gate "$(or $(GATE),0.8)" \
		$(if $(LIMIT),--limit "$(LIMIT)",) \
		$(if $(DISTILL_COMPARE_SPLIT),--compare-split "$(DISTILL_COMPARE_SPLIT)",) \
		$(if $(DISTILL_COMPARE_LIMIT),--compare-limit "$(DISTILL_COMPARE_LIMIT)",) \
		$(if $(DISTILL_OUT),--out-dir "$(DISTILL_OUT)",) \
		$(if $(TRAINER),--trainer "$(TRAINER)",)

register-adapter: ## Register an adapter trained outside the loop (ADAPTER_DIR=<dir> GOLDSET= CORPUS= SOURCE_RUN=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(ADAPTER_DIR)" || { echo "ERROR: set ADAPTER_DIR=<adapter dir>"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main register-adapter --adapter-dir "$(ADAPTER_DIR)" \
		$(if $(GOLDSET),--goldset "$(GOLDSET)",) $(if $(CORPUS),--corpus "$(CORPUS)",) \
		$(if $(SOURCE_RUN),--source-run "$(SOURCE_RUN)",)

list-adapters: ## List registered adapters with base model, eval evidence, and staleness verdict (ADAPTERS_JSON=1)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main list-adapters $(if $(ADAPTERS_JSON),--json,)

serve-adapter: ## Serve a registered adapter (ADAPTER=<id> BACKEND=vllm|ollama|llamacpp SERVE_SMOKE=1 to probe and exit)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(ADAPTER)" || { echo "ERROR: set ADAPTER=<adapter id> (see 'make list-adapters')"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main serve-adapter --adapter "$(ADAPTER)" \
		$(if $(BACKEND),--backend "$(BACKEND)",) $(if $(SERVE_SMOKE),--smoke,)

gc-adapters: ## Delete superseded adapters no run bundle cites (GC_FORCE=1 overrides citations; GC_DRY_RUN=1 previews)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main gc-adapters $(if $(GC_FORCE),--force,) $(if $(GC_DRY_RUN),--dry-run,)
