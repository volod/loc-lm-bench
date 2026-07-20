## Corpus conversion, ingestion, validation, and SQuAD preparation.

.PHONY: gen-rag-items pdf-to-markdown ingest-corpus validate-goldset ingest-squad \
	external-squad-rag audit-corpus-conflicts

gen-rag-items: ## Generate sample canonical UA RAG gold items into .data/llb/
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	bash "$(PROJECT_ROOT)/scripts/gen_rag_items.sh"

pdf-to-markdown: ## Convert PDF_DIR to markdown corpus (default DATA_DIR/quickstart-pdf-corpus; PDF_OUT_DIR=, PDF_MIN_CHARS=, PDF_PARSER=auto, PDF_REFRESH=1 reconverts unchanged PDFs)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@args=(); \
	if [ -n "$(PDF_OUT_DIR)" ]; then args+=("$(PDF_OUT_DIR)"); fi; \
	if [ -n "$(PDF_MIN_CHARS)" ]; then args+=(--min-chars "$(PDF_MIN_CHARS)"); fi; \
	if [ -n "$(PDF_PARSER)" ]; then args+=(--parser "$(PDF_PARSER)"); fi; \
	if [ -n "$(PDF_REFRESH)" ]; then args+=(--refresh); fi; \
	$(PY) -m llb.main pdf-to-markdown "$(PDF_DIR)" "$${args[@]}"

ingest-corpus: ## Ingest a mixed txt/md/pdf CORPUS_ROOT into one .md/.txt corpus (CORPUS_OUT_DIR=, CORPUS_MIN_CHARS=, CORPUS_PARSER=auto, CORPUS_REFRESH=1, CORPUS_DEFAULT_LANGUAGE=uk, CORPUS_ACL_LABEL=tag)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@args=(--root "$(CORPUS_ROOT)" --min-chars "$(CORPUS_MIN_CHARS)" --parser "$(CORPUS_PARSER)"); \
	if [ -n "$(CORPUS_OUT_DIR)" ]; then args+=(--out-dir "$(CORPUS_OUT_DIR)"); fi; \
	if [ -n "$(CORPUS_REFRESH)" ]; then args+=(--refresh); fi; \
	if [ -n "$(CORPUS_DEFAULT_LANGUAGE)" ]; then args+=(--default-language "$(CORPUS_DEFAULT_LANGUAGE)"); fi; \
	if [ -n "$(CORPUS_SOURCE_SYSTEM)" ]; then args+=(--source-system "$(CORPUS_SOURCE_SYSTEM)"); fi; \
	if [ -n "$(CORPUS_ACL_LABEL)" ]; then args+=(--acl-label "$(CORPUS_ACL_LABEL)"); fi; \
	$(PY) -m llb.main ingest-corpus "$${args[@]}"

audit-corpus-conflicts: ## Report duplicate/stale/contradictory knowledge in CORPUS (EFFORT=hash|lexical|semantic|claim, STORE=, PROJECT_DIMS=32 exact PCA blocking, GOLDSET=, CONFLICT_MODEL=); never edits the corpus
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@args=(--corpus "$(CORPUS)" --effort "$(or $(EFFORT),hash)"); \
	if [ -n "$(STORE)" ]; then args+=(--store "$(STORE)"); fi; \
	if [ -n "$(GOLDSET)" ]; then args+=(--goldset "$(GOLDSET)"); fi; \
	if [ -n "$(CONFLICTS_OUT)" ]; then args+=(--out "$(CONFLICTS_OUT)"); fi; \
	if [ -n "$(CONFLICT_MODEL)" ]; then args+=(--conflict-model "$(CONFLICT_MODEL)"); fi; \
	if [ -n "$(CONFLICT_BACKEND)" ]; then args+=(--conflict-backend "$(CONFLICT_BACKEND)"); fi; \
	if [ -n "$(CONFLICT_BASE_URL)" ]; then args+=(--conflict-base-url "$(CONFLICT_BASE_URL)"); fi; \
	if [ -n "$(COS_THRESHOLD)" ]; then args+=(--cos-threshold "$(COS_THRESHOLD)"); fi; \
	if [ -n "$(COS_QUANTILE)" ]; then args+=(--cos-quantile "$(COS_QUANTILE)"); fi; \
	if [ -n "$(MAX_CANDIDATE_PAIRS)" ]; then args+=(--max-candidate-pairs "$(MAX_CANDIDATE_PAIRS)"); fi; \
	if [ -n "$(NULL_SAMPLE_PAIRS)" ]; then args+=(--null-sample-pairs "$(NULL_SAMPLE_PAIRS)"); fi; \
	if [ -n "$(NULL_SEED)" ]; then args+=(--null-seed "$(NULL_SEED)"); fi; \
	if [ -n "$(MAX_CLAIM_PAIRS)" ]; then args+=(--max-claim-pairs "$(MAX_CLAIM_PAIRS)"); fi; \
	if [ -n "$(MIN_CLAIM_TOKENS)" ]; then args+=(--min-claim-tokens "$(MIN_CLAIM_TOKENS)"); fi; \
	if [ -n "$(PROJECT_DIMS)" ]; then args+=(--project-dims "$(PROJECT_DIMS)"); fi; \
	if [ -n "$(NO_CENTER_VECTORS)" ]; then args+=(--no-center-vectors); fi; \
	$(PY) -m llb.main audit-corpus-conflicts "$${args[@]}"

validate-goldset: ## Validate GOLDSET and/or CHAINS against CORPUS (defaults to the committed fixture)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@args=(--corpus-root "$(CORPUS)"); \
	if [ -n "$(GOLDSET)" ] && { [ -z "$(CHAINS)" ] || [ "$(origin GOLDSET)" != "file" ]; }; then args+=(--goldset "$(GOLDSET)"); fi; \
	if [ -n "$(CHAINS)" ]; then args+=(--chains "$(CHAINS)"); fi; \
	$(PY) -m llb.goldset.validate "$${args[@]}"

ingest-squad: ## Ingest local SQuAD QA; matching reviewed ids are verified (SQUAD_JSON=path)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.prep.ingest_squad --squad-json "$(SQUAD_JSON)"

external-squad-rag: ## Curate prompt-02 SQuAD exports, import a canonical goldset, validate, and build RAG
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(SQUAD_DRAFT_CORPUS)" || { echo "ERROR: set SQUAD_DRAFT_CORPUS=<staged-corpus-dir>"; exit 1; }
	@test -n "$(SQUAD_DRAFT_INPUT_DIR)$(SQUAD_DRAFT_INPUTS)" || { echo "ERROR: set SQUAD_DRAFT_INPUT_DIR=<exports-dir> or SQUAD_DRAFT_INPUTS=\"<file> [<file> ...]\""; exit 1; }
	@set -euo pipefail; \
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; \
	mkdir -p "$(SQUAD_DRAFT_OUT_DIR)"; \
	inputs=(); \
	if [ -n "$(SQUAD_DRAFT_INPUTS)" ]; then \
	  read -r -a inputs <<< "$(SQUAD_DRAFT_INPUTS)"; \
	else \
	  while IFS= read -r path; do inputs+=("$$path"); done < <(find "$(SQUAD_DRAFT_INPUT_DIR)" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.txt' -o -name '*.md' \) | sort); \
	fi; \
	if [ "$${#inputs[@]}" -eq 0 ]; then echo "ERROR: no draft export files found" >&2; exit 1; fi; \
	echo "[external-squad-rag] curate $${#inputs[@]} export files -> $(SQUAD_DRAFT_CURATED)"; \
	$(PY) -m llb.main curate-drafts "$${inputs[@]}" --kind squad --out "$(SQUAD_DRAFT_CURATED)" \
	  --corpus-root "$(SQUAD_DRAFT_CORPUS)" $(if $(filter 0,$(SQUAD_DRAFT_SEMANTIC)),--no-semantic-dedup,); \
	echo "[external-squad-rag] ingest -> $(SQUAD_DRAFT_OUT_DIR)/llb/goldset/$(SQUAD_DRAFT_GOLDSET_NAME)"; \
	$(PY) -m llb.prep.ingest_squad --squad-json "$(SQUAD_DRAFT_CURATED)" \
	  --out-dir "$(SQUAD_DRAFT_OUT_DIR)/llb" --out-name "$(SQUAD_DRAFT_GOLDSET_NAME)"; \
	echo "[external-squad-rag] validate"; \
	$(PY) -m llb.goldset.validate --goldset "$(SQUAD_DRAFT_OUT_DIR)/llb/goldset/$(SQUAD_DRAFT_GOLDSET_NAME)" \
	  --corpus-root "$(SQUAD_DRAFT_OUT_DIR)/llb/corpus"; \
	echo "[external-squad-rag] build RAG index"; \
	export DATA_DIR="$(SQUAD_DRAFT_OUT_DIR)"; \
	$(PY) -m llb.main build-index --corpus-root "$(SQUAD_DRAFT_OUT_DIR)/llb/corpus" \
	  $(if $(EMBEDDING_MODEL),--embedding-model "$(EMBEDDING_MODEL)",); \
	echo "[external-squad-rag] goldset: $(SQUAD_DRAFT_OUT_DIR)/llb/goldset/$(SQUAD_DRAFT_GOLDSET_NAME)"; \
	echo "[external-squad-rag] corpus:  $(SQUAD_DRAFT_OUT_DIR)/llb/corpus"; \
	echo "[external-squad-rag] rag:     $(SQUAD_DRAFT_OUT_DIR)/llb/rag"
