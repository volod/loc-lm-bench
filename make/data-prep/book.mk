## Scanned-book recovery: the resumable per-page, per-stage `book-ocr` run.

.PHONY: book-ocr book-ocr-status

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
