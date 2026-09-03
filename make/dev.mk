# Developer environment, local UI, and test targets.
##@ Development

.PHONY: \
	demo-eval mlflow board recommend recommend-agent-profile acceptance-gate-audit venv venv-restamp \
	install-extras lock-drift \
	apt-deps test test-fast format \
	ci ci-checks ci-github complexity-gate shell-lint-gate lint-md lint-doc-links lint-spec-plan \
	lint-toolchain check-artifact-contracts generate-artifact-contracts

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

recommend-agent-profile: ## Compose ONE agent operating profile from the lanes that have run here
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main recommend --agent-profile --no-chart \
		--min-cases "$(RECOMMEND_MIN_CASES)" \
		$(if $(RECOMMEND_GPU_GB),--gpu-gb $(RECOMMEND_GPU_GB),) \
		$(if $(RECOMMEND_MIN_TOK_S),--min-tokens-per-s $(RECOMMEND_MIN_TOK_S),)

acceptance-gate-audit: ## Classify experiment counts and write $DATA_DIR/acceptance-gate-audit/<run>/
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.quality.acceptance_gates

# `uv sync --inexact` installs the uv.lock versions -- the same set GitHub CI syncs -- WITHOUT
# pruning packages the lock does not name. The exclusion matters: vLLM/torch/flash-attn (added by
# scripts/build_vllm.sh) and hand-installed extras like `.[pdf-quality]` / `.[review]` live in this
# venv too, and plain `uv sync` would uninstall them on every re-run. It stops holding the moment
# the SYSTEM python is patched under the venv, because uv then REPLACES the environment instead of
# updating it -- so scripts/setup_venv.sh decides reuse-or-rebuild before syncing (llb.build.venv_state).
venv: ## Create/update .venv from uv.lock + extras + vLLM on CUDA hosts; VENV_INSTALL_VLLM=0 to skip
	@command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found -- install from https://docs.astral.sh/uv/"; exit 1; }
	@SKIP_APT="$(SKIP_APT)" bash "$(PROJECT_ROOT)/scripts/install_apt_deps.sh" production
	@case ",$(EXTRAS)," in *,dev,*|*,dev) SKIP_APT="$(SKIP_APT)" bash "$(PROJECT_ROOT)/scripts/install_apt_deps.sh" dev ;; esac
	@VENV="$(VENV)" PYTHON_VERSION="$(PYTHON_VERSION)" EXTRAS="$(EXTRAS)" \
	  RECREATE_VENV="$(RECREATE_VENV)" VENV_INSTALL_VLLM="$(VENV_INSTALL_VLLM)" \
	  UV_LINK_MODE="$(UV_LINK_MODE)" \
	  UV_SYNC_ARGS="--inexact $(UV_SYNC_LOCK_FLAG) --python $(PYTHON_VERSION) $(UV_SYNC_EXTRAS)" \
	  bash "$(PROJECT_ROOT)/scripts/setup_venv.sh"
	@bash -c 'source "$(PROJECT_ROOT)/scripts/shared/common.sh"; llb_ensure_env || true'

# The cheap half of the answer above: a CPython PATCH release keeps the ABI tag and the
# site-packages layout, so a venv whose system python moved 3.13.14 -> 3.13.15 already runs the new
# one and only its recorded version is stale. Recording it keeps a hardware-matched CUDA stack that
# a rebuild would spend gigabytes reinstalling. A minor move is refused (llb.build.venv_interpreter).
venv-restamp: ## Record the patched system python in .venv/pyvenv.cfg so `make venv` reuses the venv
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@$(PY) -m llb.build.venv_state --venv "$(VENV)" --restamp

# A bare `uv pip install -e ".[review]"` re-resolves the WHOLE requirement set (uv's pip interface
# has no lockfile), so it upgrades packages `uv sync` had just pinned -- ruff and mypy included,
# which is how a local `make ci` starts disagreeing with GitHub CI. These two run that install under
# a uv.lock-derived constraint and name whatever is still off the lock (llb.build.extras).
install-extras: ## Install optional extras under the uv.lock constraint (EXTRAS=review,pdf-quality)
	@command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found -- install from https://docs.astral.sh/uv/"; exit 1; }
	@UV_LINK_MODE="$(UV_LINK_MODE)" bash "$(PROJECT_ROOT)/scripts/install_extras.sh" --extras "$(EXTRAS)"

lock-drift: ## Name every declared package whose installed version is off uv.lock
	@UV_LINK_MODE="$(UV_LINK_MODE)" bash "$(PROJECT_ROOT)/scripts/install_extras.sh" --check

apt-deps: ## Install apt packages (APT_PROFILE=production|dev|all; SKIP_APT=1 to skip; APT_DRY_RUN=1 to list only)
	@SKIP_APT="$(SKIP_APT)" APT_DRY_RUN="$(APT_DRY_RUN)" bash "$(PROJECT_ROOT)/scripts/install_apt_deps.sh" "$(APT_PROFILE)"

# Test groups (markers registered in pyproject.toml):
#   `make test`      -- FULL local suite: every test, including the `slow` ones (real Optuna
#                       sweeps, embedder/model loads, deepeval, subprocess builds).
#   `make ci` / `test-fast` -- LIGHTWEIGHT suite for the default full local install; deselects
#                       intrinsically slow tests and opt-in dependency lanes.
#   `make ci-github` -- GitHub CI suite: also deselects quick tests that need optional extras
#                       (faiss/duckdb/adapter stores), so the base `[dev]`-only install runs with
#                       no dependency skips.
NOT_SLOW := -m "not slow and not opt_in_env"
GITHUB_SUITE := -m "not slow and not heavy_env and not opt_in_env"

# Markdown docs linted by `make lint-md` (override to narrow scope, e.g. MD_PATHS=docs/design).
MD_PATHS ?= README.md AGENTS.md CLAUDE.md GEMINI.md docs

test: ## FULL local precommit flow: pytest (incl. slow) + markdown lint (NOT run in GitHub CI)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m pytest $(PYTEST_CACHE_OPT)
	$(MAKE) lint-md

test-fast: ## Run the lightweight test suite (deselects slow and opt-in dependency tests)
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
	$(PY) -m llb.quality.acceptance_gates --check
	@$(MAKE) --no-print-directory complexity-gate
	@$(MAKE) --no-print-directory shell-lint-gate
	@$(MAKE) --no-print-directory lint-spec-plan
	@$(MAKE) --no-print-directory lint-toolchain
	@$(MAKE) --no-print-directory lint-model-roster
	@$(MAKE) --no-print-directory check-artifact-contracts

# Both also run inside ci-checks -- these aliases are for running one gate alone after a change.
complexity-gate: ## Fail on any Radon D-or-worse or cognitive-complexity finding
	@bash "$(PROJECT_ROOT)/scripts/complexity_gate.sh"

shell-lint-gate: ## Fail on any bash -n syntax error or shellcheck warning in a repo *.sh
	@bash "$(PROJECT_ROOT)/scripts/shell_lint_gate.sh"

ci: ci-checks ## Format check + lint + type check + LIGHTWEIGHT unit tests (full local install)
	$(PY) -m pytest $(PYTEST_CACHE_OPT) $(NOT_SLOW)

ci-github: ci-checks ## `ci` for the base [dev]-only env: also deselects heavy_env tests -- used by GitHub CI
	$(PY) -m pytest $(PYTEST_CACHE_OPT) $(GITHUB_SUITE)

# Fix findings BY HAND. Do NOT run `pymarkdown fix` -- it corrupts prose on this version (AGENTS.md).
lint-md: ## Lint Markdown docs with pymarkdown (config in pyproject; MD_PATHS overrides scope)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m pymarkdown scan -r --respect-gitignore $(MD_PATHS)
	$(MAKE) lint-doc-links

lint-doc-links: ## Check every relative docs link resolves and no result is cited by run path
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.quality.doc_links
	$(PY) -m llb.quality.doc_citations

lint-spec-plan: ## Check the spec's capability registry and plan.md agree (also runs in ci-checks)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.quality.spec_plan_integrity

lint-toolchain: ## Check mypy/ruff/basedpyright restatements match [tool.llb.toolchain]
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.build.toolchain check

check-artifact-contracts: ## Check registry, generated schemas/catalog, ODCS projection, and external validation
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.artifacts.check
	$(PY) samples/artifact_contracts/external_validate.py

generate-artifact-contracts: ## Refresh generated artifact schemas and catalog, then validate them
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.artifacts.check --write
	$(PY) samples/artifact_contracts/external_validate.py
