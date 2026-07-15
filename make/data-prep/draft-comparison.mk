## Local/frontier draft comparison, review, finalization, and coverage export.

.PHONY: draft-compare draft-compare-report draft-compare-review draft-compare-finalize \
	frontier-ua-draft-probe draft-compare-local local-ua-draft-probe draft-compare-analyze \
	local-ua-draft-review local-ua-draft-finalize local-ua-draft-analyze \
	local-ua-draft-complete coverage-plan-text

draft-compare: ## Compare exact shared seeds locally vs frontier (DRAFT_COMPARE_FRONTIER_MODEL=, DRAFT_COMPARE_SEEDS=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(DRAFT_COMPARE_FRONTIER_MODEL)" || { echo "ERROR: set DRAFT_COMPARE_FRONTIER_MODEL=<litellm-id>"; exit 2; }
	@set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	args=( \
	  --corpus-root "$(DRAFT_COMPARE_CORPUS)" \
	  --seeds "$(DRAFT_COMPARE_SEEDS)" \
	  --frontier-model "$(DRAFT_COMPARE_FRONTIER_MODEL)" \
	  --local-model "$(DRAFT_COMPARE_LOCAL_MODEL)" \
	  --local-backend "$(DRAFT_COMPARE_LOCAL_BACKEND)" \
	  --max-calls "$(DRAFT_COMPARE_MAX_CALLS)" \
	); \
	if [ -n "$(DRAFT_COMPARE_LOCAL_BASE_URL)" ]; then args+=(--local-base-url "$(DRAFT_COMPARE_LOCAL_BASE_URL)"); fi; \
	if [ -n "$(DRAFT_COMPARE_MAX_USD)" ]; then args+=(--max-usd "$(DRAFT_COMPARE_MAX_USD)"); fi; \
	if [ -n "$(DRAFT_COMPARE_OUT_DIR)" ]; then args+=(--out-dir "$(DRAFT_COMPARE_OUT_DIR)"); fi; \
	if [ -n "$(DRAFT_COMPARE_LOCAL_VERIFICATION)" ]; then args+=(--local-verification "$(DRAFT_COMPARE_LOCAL_VERIFICATION)"); fi; \
	if [ -n "$(DRAFT_COMPARE_FRONTIER_VERIFICATION)" ]; then args+=(--frontier-verification "$(DRAFT_COMPARE_FRONTIER_VERIFICATION)"); fi; \
	$(PY) -m llb.main draft-compare "$${args[@]}"

draft-compare-report: ## Refresh comparison accept rates without model calls (DRAFT_COMPARE_OUT_DIR=, verification worksheets required)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(DRAFT_COMPARE_OUT_DIR)" || { echo "ERROR: set DRAFT_COMPARE_OUT_DIR=<comparison-root>"; exit 2; }
	@test -n "$(DRAFT_COMPARE_LOCAL_VERIFICATION)" || { echo "ERROR: set DRAFT_COMPARE_LOCAL_VERIFICATION=<reviewed-local-csv>"; exit 2; }
	@test -n "$(DRAFT_COMPARE_FRONTIER_VERIFICATION)" || { echo "ERROR: set DRAFT_COMPARE_FRONTIER_VERIFICATION=<reviewed-frontier-csv>"; exit 2; }
	$(PY) -m llb.main draft-compare-report \
	  --report "$(DRAFT_COMPARE_OUT_DIR)/comparison.json" \
	  --local-verification "$(DRAFT_COMPARE_LOCAL_VERIFICATION)" \
	  --frontier-verification "$(DRAFT_COMPARE_FRONTIER_VERIFICATION)"

draft-compare-review: ## Interactively review both comparison worksheets (DRAFT_COMPARE_OUT_DIR=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(DRAFT_COMPARE_OUT_DIR)" || { echo "ERROR: set DRAFT_COMPARE_OUT_DIR=<comparison-root>"; exit 2; }
	$(PY) -m llb.main draft-compare-review --comparison-root "$(DRAFT_COMPARE_OUT_DIR)" $(if $(VERIFY_ORDER),--order "$(VERIFY_ORDER)")

draft-compare-finalize: ## Refresh metrics and check every comparison gate (DRAFT_COMPARE_OUT_DIR=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(DRAFT_COMPARE_OUT_DIR)" || { echo "ERROR: set DRAFT_COMPARE_OUT_DIR=<comparison-root>"; exit 2; }
	$(PY) -m llb.main draft-compare-finalize --comparison-root "$(DRAFT_COMPARE_OUT_DIR)"

frontier-ua-draft-probe: ## Run the bounded committed two-document UA probe (FRONTIER_UA_PROBE_FRONTIER_MODEL=, MAX_USD=, OUT_DIR=)
	@test -n "$(FRONTIER_UA_PROBE_FRONTIER_MODEL)" || { echo "ERROR: set FRONTIER_UA_PROBE_FRONTIER_MODEL=<litellm-id>"; exit 2; }
	@test -n "$(FRONTIER_UA_PROBE_MAX_USD)" || { echo "ERROR: set FRONTIER_UA_PROBE_MAX_USD=<authorized-usd-cap>"; exit 2; }
	@test -n "$(FRONTIER_UA_PROBE_OUT_DIR)" || { echo "ERROR: set FRONTIER_UA_PROBE_OUT_DIR=<comparison-root>"; exit 2; }
	$(MAKE) --no-print-directory draft-compare \
	  DRAFT_COMPARE_CORPUS="$(FRONTIER_UA_PROBE_CORPUS)" \
	  DRAFT_COMPARE_SEEDS="$(FRONTIER_UA_PROBE_SEEDS)" \
	  DRAFT_COMPARE_LOCAL_MODEL="$(FRONTIER_UA_PROBE_LOCAL_MODEL)" \
	  DRAFT_COMPARE_FRONTIER_MODEL="$(FRONTIER_UA_PROBE_FRONTIER_MODEL)" \
	  DRAFT_COMPARE_MAX_USD="$(FRONTIER_UA_PROBE_MAX_USD)" \
	  DRAFT_COMPARE_MAX_CALLS="$(FRONTIER_UA_PROBE_MAX_CALLS)" \
	  DRAFT_COMPARE_OUT_DIR="$(FRONTIER_UA_PROBE_OUT_DIR)"

draft-compare-local: ## Sequential local Qwen/Gemma exact-seed comparison (LOCAL_DRAFT_COMPARE_OUT_DIR=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	args=(--corpus-root "$(LOCAL_DRAFT_COMPARE_CORPUS)" --seeds "$(LOCAL_DRAFT_COMPARE_SEEDS)"); \
	if [ -n "$(LOCAL_DRAFT_COMPARE_BASELINE_MODEL)" ]; then args+=(--baseline-model "$(LOCAL_DRAFT_COMPARE_BASELINE_MODEL)"); fi; \
	if [ -n "$(LOCAL_DRAFT_COMPARE_PROBE_MODEL)" ]; then args+=(--probe-model "$(LOCAL_DRAFT_COMPARE_PROBE_MODEL)"); fi; \
	if [ -n "$(LOCAL_DRAFT_COMPARE_OUT_DIR)" ]; then args+=(--out-dir "$(LOCAL_DRAFT_COMPARE_OUT_DIR)"); fi; \
	$(PY) -m llb.main draft-compare-local "$${args[@]}"

local-ua-draft-probe: ## Adaptive two-document Qwen/Gemma run (LOCAL_DRAFT_COMPARE_OUT_DIR= required)
	@test -n "$(LOCAL_DRAFT_COMPARE_OUT_DIR)" || { echo "ERROR: set LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root>"; exit 2; }
	$(MAKE) --no-print-directory draft-compare-local

draft-compare-analyze: ## Print comparison.json metrics and deltas (DRAFT_COMPARE_OUT_DIR=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(DRAFT_COMPARE_OUT_DIR)" || { echo "ERROR: set DRAFT_COMPARE_OUT_DIR=<comparison-root>"; exit 2; }
	$(PY) -m llb.main draft-compare-analyze --report "$(DRAFT_COMPARE_OUT_DIR)/comparison.json" $(if $(COMPARE_ANALYZE_JSON),--json) $(if $(COMPARE_REQUIRE_GATES),--require-passed-gates)

local-ua-draft-review: ## Review both adaptive local comparison lanes (LOCAL_DRAFT_COMPARE_OUT_DIR=)
	@test -n "$(LOCAL_DRAFT_COMPARE_OUT_DIR)" || { echo "ERROR: set LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root>"; exit 2; }
	$(MAKE) --no-print-directory draft-compare-review DRAFT_COMPARE_OUT_DIR="$(LOCAL_DRAFT_COMPARE_OUT_DIR)"

local-ua-draft-finalize: ## Finalize reviewed local comparison gates (LOCAL_DRAFT_COMPARE_OUT_DIR=)
	@test -n "$(LOCAL_DRAFT_COMPARE_OUT_DIR)" || { echo "ERROR: set LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root>"; exit 2; }
	$(MAKE) --no-print-directory draft-compare-finalize DRAFT_COMPARE_OUT_DIR="$(LOCAL_DRAFT_COMPARE_OUT_DIR)"

local-ua-draft-analyze: ## Analyze local comparison metrics (LOCAL_DRAFT_COMPARE_OUT_DIR=, COMPARE_ANALYZE_JSON=1)
	@test -n "$(LOCAL_DRAFT_COMPARE_OUT_DIR)" || { echo "ERROR: set LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root>"; exit 2; }
	$(MAKE) --no-print-directory draft-compare-analyze \
	  DRAFT_COMPARE_OUT_DIR="$(LOCAL_DRAFT_COMPARE_OUT_DIR)" \
	  COMPARE_ANALYZE_JSON="$(COMPARE_ANALYZE_JSON)" \
	  COMPARE_REQUIRE_GATES="$(COMPARE_REQUIRE_GATES)"

local-ua-draft-complete: ## Human review -> finalize -> analyze for an existing local comparison
	@test -n "$(LOCAL_DRAFT_COMPARE_OUT_DIR)" || { echo "ERROR: set LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root>"; exit 2; }
	$(MAKE) --no-print-directory local-ua-draft-review
	$(MAKE) --no-print-directory local-ua-draft-finalize
	$(MAKE) --no-print-directory local-ua-draft-analyze

coverage-plan-text: ## Convert a coverage JSON slice into a NotebookLM-friendly text source
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(COVERAGE_JSON)" || { echo "ERROR: set COVERAGE_JSON=<coverage-slice.json>" >&2; exit 2; }
	@set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	args=(--input "$(COVERAGE_JSON)"); \
	if [ -n "$(COVERAGE_TEXT)" ]; then args+=(--out "$(COVERAGE_TEXT)"); fi; \
	$(PY) -m llb.main coverage-plan-text "$${args[@]}"
