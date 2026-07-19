# Developer environment, local UI, and test targets.
##@ Development

.PHONY: \
	demo-eval mlflow board recommend venv apt-deps test test-fast format ci ci-checks ci-github lint-md

demo-eval: ## End-to-end: venv -> committed gold set -> index -> validate -> prep-models -> run-eval+telemetry
	@source "$(PROJECT_ROOT)/scripts/shared/common.sh"; \
	llb_ensure_env || exit 0; \
	mkdir -p "$(LOG_DIR)"; LOG="$(LOG_DIR)/pipeline-$$(date +%Y%m%d-%H%M%S).log"; \
	echo "[demo-eval] end-to-end pipeline (idempotent); logging to $$LOG"; \
	( \
	  llb_load_env; \
	  echo "### [1/6] venv (idempotent; RECREATE_VENV=1 to rebuild)"; \
	  $(MAKE) --no-print-directory venv || exit 1; \
	  echo "### [2/6] validate committed published gold set"; \
	  $(PY) -m llb.goldset.validate --goldset "$(ALL_GOLDSET)" \
	    --corpus-root "$(ALL_CORPUS)" || exit 1; \
	  echo "### [3/6] build index"; \
	  $(PY) -m llb.main build-index --corpus-root "$(ALL_CORPUS)" || exit 1; \
	  echo "### [4/6] validate retrieval"; \
	  $(PY) -m llb.main validate-retrieval --goldset "$(ALL_GOLDSET)" --k $(RAG_K) \
	    || echo "  WARN: retrieval below the 0.8 gate (non-fatal; continuing)"; \
	  echo "### [5/6] prep models (backend=$(PREP_ALL_BACKEND); cached downloads are skipped)"; \
	  $(MAKE) --no-print-directory prep-models PREP_BACKEND=$(PREP_ALL_BACKEND) || exit 1; \
	  echo "### [6/6] run-eval + telemetry (model=$(MODEL) backend=$(BACKEND))"; \
	  $(PY) -m llb.main run-eval --model "$(MODEL)" --backend "$(BACKEND)" \
	    --goldset "$(ALL_GOLDSET)" --split final --limit $(LIMIT) --telemetry || exit 1; \
	  echo "### pipeline complete"; \
	) 2>&1 | tee "$$LOG"; \
	rc=$${PIPESTATUS[0]}; \
	if [ "$$rc" -eq 0 ]; then \
	  { echo "[demo-eval] OK -- full log: $$LOG"; \
	    echo "[demo-eval] review experiment results: make mlflow"; \
	    echo "[demo-eval] MLflow UI: http://$(MLFLOW_HOST):$(MLFLOW_PORT)"; \
	    echo "[demo-eval] guide: docs/guides/benchmarking/mlflow-analysis.md"; } | tee -a "$$LOG"; \
	else echo "[demo-eval] FAILED (exit $$rc) -- investigate the log: $$LOG" \
	  | tee -a "$$LOG" >&2; exit "$$rc"; fi

mlflow: ## Serve the shared MLflow experiment UI (MLFLOW_HOST=127.0.0.1 MLFLOW_PORT=5000)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main mlflow-ui --host "$(MLFLOW_HOST)" --port "$(MLFLOW_PORT)"

board: ## Serve the Streamlit leaderboard (BOARD_HOST=127.0.0.1 BOARD_PORT=8501)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main board --host "$(BOARD_HOST)" --port "$(BOARD_PORT)"

recommend: ## Summarize a sweep into host-adaptive picks + chart (RECOMMEND_MIN_CASES= RECOMMEND_GPU_GB= RECOMMEND_MIN_TOK_S=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main recommend --min-cases "$(RECOMMEND_MIN_CASES)" \
		$(if $(RECOMMEND_GPU_GB),--gpu-gb $(RECOMMEND_GPU_GB),) \
		$(if $(RECOMMEND_MIN_TOK_S),--min-tokens-per-s $(RECOMMEND_MIN_TOK_S),) \
		$(if $(RECOMMEND_JSON_OUT),--json-out "$(RECOMMEND_JSON_OUT)",) \
		$(if $(filter 1 true yes,$(RECOMMEND_NO_CHART)),--no-chart,)

venv: ## Create/update .venv + extras + vLLM on CUDA hosts; VENV_INSTALL_VLLM=0 to skip
	@command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found -- install from https://docs.astral.sh/uv/"; exit 1; }
	@SKIP_APT="$(SKIP_APT)" bash "$(PROJECT_ROOT)/scripts/install_apt_deps.sh" production
	@case ",$(EXTRAS)," in *,dev,*|*,dev) SKIP_APT="$(SKIP_APT)" bash "$(PROJECT_ROOT)/scripts/install_apt_deps.sh" dev ;; esac
	@if [ -n "$(RECREATE_VENV)" ] && [ -d "$(VENV)" ]; then echo "[venv] RECREATE_VENV set -- removing $(VENV)"; rm -rf "$(VENV)"; fi
	@if [ ! -x "$(PY)" ]; then \
		echo "[venv] creating $(VENV) (py$(PYTHON_VERSION))"; uv venv --python $(PYTHON_VERSION) "$(VENV)"; \
	else \
		echo "[venv] reusing $(VENV) -- updating deps (RECREATE_VENV=1 to rebuild)"; \
	fi
	@UV_LINK_MODE="$(UV_LINK_MODE)" bash -c 'source "$(PROJECT_ROOT)/scripts/shared/common.sh"; llb_export_uv_link_mode; echo "[venv] uv link mode: $${UV_LINK_MODE:-default (cache + checkout share a device)}"; uv pip install --python "$(PY)" -e ".[$(EXTRAS)]"'
	@echo "[venv] ready: $(VENV) (extras: $(EXTRAS))"
	@case "$(VENV_INSTALL_VLLM)" in \
	  0|false|no) echo "[venv] vLLM install skipped (VENV_INSTALL_VLLM=$(VENV_INSTALL_VLLM))" ;; \
	  1|true|yes) echo "[venv] installing vLLM binary wheels (forced)"; bash "$(PROJECT_ROOT)/scripts/build_vllm.sh" ;; \
	  auto) \
	    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then \
	      echo "[venv] CUDA host detected; installing vLLM binary wheels"; \
	      bash "$(PROJECT_ROOT)/scripts/build_vllm.sh"; \
	    else \
	      echo "[venv] vLLM install skipped (no CUDA GPU detected; set VENV_INSTALL_VLLM=1 to force)"; \
	    fi ;; \
	  *) echo "ERROR: VENV_INSTALL_VLLM must be auto, 1, or 0 (got $(VENV_INSTALL_VLLM))" >&2; exit 2 ;; \
	esac
	@bash -c 'source "$(PROJECT_ROOT)/scripts/shared/common.sh"; llb_ensure_env || true'

apt-deps: ## Install apt packages (APT_PROFILE=production|dev|all; SKIP_APT=1 to skip; APT_DRY_RUN=1 to list only)
	@SKIP_APT="$(SKIP_APT)" APT_DRY_RUN="$(APT_DRY_RUN)" bash "$(PROJECT_ROOT)/scripts/install_apt_deps.sh" "$(APT_PROFILE)"

# Three test groups (markers registered in pyproject.toml):
#   `make test`      -- FULL local suite: every test, including the `slow` ones (real Optuna
#                       sweeps, embedder/model loads, deepeval, subprocess builds).
#   `make ci` / `test-fast` -- LIGHTWEIGHT suite (`-m "not slow"`) for the full local install.
#   `make ci-github` -- GitHub CI suite (`-m "not slow and not heavy_env"`): also deselects the
#                       quick tests that need optional extras (faiss/duckdb/adapter stores), so
#                       the base `[dev]`-only GitHub install runs with no dependency skips.
NOT_SLOW := -m "not slow"
GITHUB_SUITE := -m "not slow and not heavy_env"

# Markdown docs linted by `make lint-md` (override to narrow scope, e.g. MD_PATHS=docs/design).
MD_PATHS ?= README.md AGENTS.md CLAUDE.md GEMINI.md docs

test: ## FULL local precommit flow: pytest (incl. slow) + markdown lint (NOT run in GitHub CI)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m pytest $(PYTEST_CACHE_OPT)
	$(MAKE) lint-md

test-fast: ## Run the lightweight test suite (skips slow tests; mirrors CI)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m pytest $(PYTEST_CACHE_OPT) $(NOT_SLOW)

format: ## Format Python sources and tests with Ruff
	@test -x "$(VENV)/bin/ruff" || { echo "ERROR: ruff missing -- run 'make venv' first"; exit 1; }
	$(VENV)/bin/ruff format src tests

ci-checks:
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- create one + install '.[dev]' first"; exit 1; }
	$(VENV)/bin/ruff format --check src tests
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/mypy --python-version $(PYTHON_VERSION)

ci: ci-checks ## Format check + lint + type check + LIGHTWEIGHT unit tests (full local install)
	$(PY) -m pytest $(PYTEST_CACHE_OPT) $(NOT_SLOW)

ci-github: ci-checks ## `ci` for the base [dev]-only env: also deselects heavy_env tests -- used by GitHub CI
	$(PY) -m pytest $(PYTEST_CACHE_OPT) $(GITHUB_SUITE)

# Fix findings BY HAND. Do NOT run `pymarkdown fix` -- it corrupts prose on this version (AGENTS.md).
lint-md: ## Lint Markdown docs with pymarkdown (config in pyproject; MD_PATHS overrides scope)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m pymarkdown scan -r --respect-gitignore $(MD_PATHS)
