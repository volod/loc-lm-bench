## Scanned-book recovery: the resumable per-page, per-stage `book-ocr` run.

.PHONY: book-ocr book-ocr-status book-ocr-engines book-ocr-score book-ocr-pilot-select \
	book-survey-list book-survey-install book-survey-smoke book-survey-rescore

book-ocr: ## Run (or BOOK_DRY_RUN=1 validate) a scanned-book recovery over BOOK_PDF (BOOK_PAGES=1-, BOOK_BATCH_PAGES=25, BOOK_STAGES=render, BOOK_CONFIG=yaml, BOOK_SOURCE_POLICY=symlink|copy|reference, BOOK_REUSE_FROM=run, BOOK_OUT_DIR=); BOOK_RESUME=<run-dir> continues a run
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@args=(); \
	if [ -n "$(BOOK_PDF)" ]; then args+=("$(BOOK_PDF)"); fi; \
	if [ -n "$(BOOK_RESUME)" ]; then args+=(--resume "$(BOOK_RESUME)"); fi; \
	if [ -n "$(BOOK_PAGES)" ]; then args+=(--pages "$(BOOK_PAGES)"); fi; \
	if [ -n "$(BOOK_BATCH_PAGES)" ]; then args+=(--batch-pages "$(BOOK_BATCH_PAGES)"); fi; \
	if [ -n "$(BOOK_MAX_ATTEMPTS)" ]; then args+=(--max-attempts "$(BOOK_MAX_ATTEMPTS)"); fi; \
	if [ -n "$(BOOK_STAGES)" ]; then args+=(--stages "$(BOOK_STAGES)"); fi; \
	if [ -n "$(BOOK_CONFIG)" ]; then args+=(--config "$(BOOK_CONFIG)"); fi; \
	if [ -n "$(BOOK_SOURCE_POLICY)" ]; then args+=(--source-policy "$(BOOK_SOURCE_POLICY)"); fi; \
	if [ -n "$(BOOK_REUSE_FROM)" ]; then args+=(--reuse-from "$(BOOK_REUSE_FROM)"); fi; \
	if [ -n "$(BOOK_OUT_DIR)" ]; then args+=(--out-dir "$(BOOK_OUT_DIR)"); fi; \
	if [ -n "$(BOOK_DRY_RUN)" ]; then args+=(--dry-run); fi; \
	$(PY) -m llb.main book-ocr run "$${args[@]}"

book-ocr-status: ## Report per-page, per-stage status of BOOK_RUN (BOOK_SHOW=all|open|none, BOOK_JSON=1 prints the page manifest)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@args=("$(BOOK_RUN)" --show "$(or $(BOOK_SHOW),all)"); \
	if [ -n "$(BOOK_JSON)" ]; then args+=(--json); fi; \
	$(PY) -m llb.main book-ocr status "$${args[@]}"

book-ocr-score: ## Score BOOK_RUN against reference dir BOOK_REFERENCE (BOOK_CITATIONS=existing-lane sidecar, BOOK_ENGINE=, BOOK_SCORE_OUT=, BOOK_JSON=1)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BOOK_REFERENCE)" || { echo "ERROR: set BOOK_REFERENCE=<reference-dir>"; exit 2; }
	@args=(--reference "$(BOOK_REFERENCE)"); \
	if [ -n "$(BOOK_RUN)" ]; then args+=("$(BOOK_RUN)"); fi; \
	if [ -n "$(BOOK_CITATIONS)" ]; then args+=(--citations "$(BOOK_CITATIONS)"); fi; \
	if [ -n "$(BOOK_ENGINE)" ]; then args+=(--engine "$(BOOK_ENGINE)"); fi; \
	if [ -n "$(BOOK_SCORE_OUT)" ]; then args+=(--out "$(BOOK_SCORE_OUT)"); fi; \
	if [ -n "$(BOOK_JSON)" ]; then args+=(--json); fi; \
	$(PY) -m llb.main book-ocr score "$${args[@]}"

book-ocr-pilot-select: ## Select a seeded pilot page set from BOOK_RUN (BOOK_PILOT_SIZE=25, BOOK_PILOT_SEED=0, BOOK_PILOT_OUT=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BOOK_RUN)" || { echo "ERROR: set BOOK_RUN=<run-dir>"; exit 2; }
	@args=("$(BOOK_RUN)" --size "$(or $(BOOK_PILOT_SIZE),25)" --seed "$(or $(BOOK_PILOT_SEED),0)"); \
	if [ -n "$(BOOK_PILOT_OUT)" ]; then args+=(--out "$(BOOK_PILOT_OUT)"); fi; \
	$(PY) -m llb.main book-ocr pilot-select "$${args[@]}"

book-ocr-engines: ## List ledger engines and whether each one can run on this host
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@$(PY) -m llb.main book-ocr engines

# The recovery-tool survey. Each candidate stack lives in its own uv virtualenv under
# $DATA_DIR/venvs/book-<env>/ (never .venv), and the install's uv cache defaults to
# $DATA_DIR/cache/uv so tens of GB of CUDA wheels land on the data volume, not the home directory.
# MAX_JOBS caps any source build a candidate triggers (AGENTS.md: llb_max_jobs).
book-survey-list: ## List the surveyed recovery-tool candidates per stage (BOOK_SURVEY_STAGE=ocr|formula|...)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@$(PY) -m llb.main book-ocr survey list $(if $(BOOK_SURVEY_STAGE),--stage "$(BOOK_SURVEY_STAGE)",)

book-survey-install: ## Install the isolated survey environments under DATA_DIR/venvs/book-<env> (BOOK_SURVEY_ENVS=all|light,mineru,docling,marker,paddle,paddle-fw)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@bash -c 'export UV_CACHE_DIR="$${UV_CACHE_DIR:-$(DATA_DIR)/cache/uv}"; \
		source "$(PROJECT_ROOT)/scripts/shared/common.sh"; llb_load_env; \
		MAX_JOBS="$$(llb_max_jobs)" CMAKE_BUILD_PARALLEL_LEVEL="$$(llb_max_jobs)" \
		"$(PY)" -m llb.main book-ocr survey install --envs "$(or $(BOOK_SURVEY_ENVS),all)"'

book-survey-smoke: ## One-page smoke of each candidate, one at a time (BOOK_SURVEY_CANDIDATES=all|ids, BOOK_SURVEY_PAGES="synthetic <pdf>:<page>")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@args=(--candidates "$(or $(BOOK_SURVEY_CANDIDATES),all)"); \
	for p in $(or $(BOOK_SURVEY_PAGES),synthetic); do args+=(--page "$$p"); done; \
	$(PY) -m llb.main book-ocr survey smoke "$${args[@]}"

book-survey-rescore: ## Recompute a survey's synthetic-page smoke scores from its saved readings, no GPU (BOOK_SURVEY_DIR=<run-dir>/survey)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BOOK_SURVEY_DIR)" || { echo "ERROR: set BOOK_SURVEY_DIR=<run-dir>/survey"; exit 2; }
	@$(PY) -m llb.main book-ocr survey rescore "$(BOOK_SURVEY_DIR)"
