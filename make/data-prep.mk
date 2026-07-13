# Data preparation, external-draft curation, and human verification targets.
##@ Data Preparation and Verification

.PHONY: \
	gen-rag-items pdf-to-markdown ingest-corpus validate-goldset ingest-squad \
	external-squad-rag curate-drafts import-external-draft coverage-plan-text \
	calibration-worksheet calibration-run calibration-rate calibration-score cross-check-goldset \
	verify-sample verify-review verify-adjudicate verify-accept judge-experiment ingest-uk-squad \
	prepare-goldset-draft draft-compare draft-compare-review draft-compare-report \
	draft-compare-finalize frontier-ua-draft-probe build-query-glossary derive-security-cases \
	draft-compare-local draft-compare-analyze local-ua-draft-probe local-ua-draft-review \
	local-ua-draft-finalize local-ua-draft-analyze local-ua-draft-complete \
	derive-security-worksheet chain-goldset-pipeline chain-goldset-finalize

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

curate-drafts: ## Merge/dedup/filter external drafts; CURATE_KIND= CURATE_INPUTS="a b" CURATE_OUT= CURATE_CORPUS= CURATE_DEDUP_AGAINST= CURATE_SEMANTIC=0|1
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(CURATE_INPUTS)" || { echo "ERROR: set CURATE_INPUTS=\"<file> [<file> ...]\""; exit 1; }
	@test -n "$(CURATE_OUT)" || { echo "ERROR: set CURATE_OUT=<merged-artifact-path>"; exit 1; }
	$(PY) -m llb.main curate-drafts $(CURATE_INPUTS) --kind "$(CURATE_KIND)" --out "$(CURATE_OUT)" \
		$(if $(CURATE_CORPUS),--corpus-root "$(CURATE_CORPUS)",) \
		$(foreach b,$(CURATE_DEDUP_AGAINST),--dedup-against "$(b)") \
		$(if $(filter 0,$(CURATE_SEMANTIC)),--no-semantic-dedup,)

import-external-draft: ## Import an external-service grounded goldset (Artifact B); ARTIFACT= CORPUS= SIDECAR= [OUT_DIR= RETRIEVAL_INDEX_DIR= RETRIEVAL_K= DROP_NONRETRIEVABLE_NEEDLES=1]
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(ARTIFACT)" || { echo "ERROR: set ARTIFACT=<grounded-jsonl export>"; exit 1; }
	@test -n "$(CORPUS)" || { echo "ERROR: set CORPUS=<local corpus dir the quotes ground against>"; exit 1; }
	@test -n "$(SIDECAR)" || { echo "ERROR: set SIDECAR=<external_provenance.json (data_classification: open)>"; exit 1; }
	$(PY) -m llb.main import-external-draft --artifact "$(ARTIFACT)" --corpus-root "$(CORPUS)" \
		--sidecar "$(SIDECAR)" $(if $(OUT_DIR),--out-dir "$(OUT_DIR)",) \
		$(if $(RETRIEVAL_INDEX_DIR),--retrieval-index-dir "$(RETRIEVAL_INDEX_DIR)",) \
		$(if $(RETRIEVAL_K),--retrieval-k "$(RETRIEVAL_K)",) \
		$(if $(DROP_NONRETRIEVABLE_NEEDLES),--drop-nonretrievable-needles,)

calibration-worksheet: ## Emit a blank judge-calibration worksheet from GOLDSET
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.judge.calibration worksheet --goldset "$(GOLDSET)" \
		--out "$(CAL_WS)"

calibration-run: ## Run MODEL on the calibration split -> filled worksheet (model_answer + judge_rating if JUDGE_MODEL set)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main run-eval --model "$(MODEL)" --backend "$(BACKEND)" \
		--goldset "$(GOLDSET)" --split calibration --worksheet "$(CAL_WS)" \
		$(if $(JUDGE_MODEL),--judge-model "$(JUDGE_MODEL)",) \
		$(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",)

calibration-rate: ## Interactively fill human ratings/answers in CAL_WS (judge_rating hidden; SHOW_JUDGE=1 to reveal, START=N, CLEAR=1 to reset)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.judge.calibration rate --worksheet "$(CAL_WS)" $(if $(START),--start $(START)) $(if $(SHOW_JUDGE),--show-judge) $(if $(CLEAR),--clear)

calibration-score: ## Score a filled worksheet: rho + bootstrap CI + trust decision (RATINGS=path, gate rho>=0.6)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.judge.calibration score --ratings "$(RATINGS)"

build-query-glossary: ## uk-query-processing: build query_glossary.json from a draft BUNDLE's dictionary candidates (QUERY_GLOSSARY_OUT=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<draft dir with prompt_dictionary_candidates.jsonl>"; exit 1; }
	$(PY) -m llb.main build-query-glossary --bundle "$(BUNDLE)" \
		--out "$(if $(QUERY_GLOSSARY_OUT),$(QUERY_GLOSSARY_OUT),$(BUNDLE)/query_glossary.json)"

derive-security-cases: ## Security benchmark: derive corpus-specific content-safety cases from a draft BUNDLE (SECURITY_DERIVE_OUT=, SECURITY_DERIVE_MAX_DENIAL=, SECURITY_DERIVE_MAX_PAIRS=, SECURITY_DERIVE_MERGE_SEED=1)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<draft dir with ontology.json + extraction.jsonl>"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main derive-security-cases --bundle "$(BUNDLE)" \
		$(if $(SECURITY_DERIVE_OUT),--out "$(SECURITY_DERIVE_OUT)",) \
		$(if $(SECURITY_DERIVE_MAX_DENIAL),--max-denial-per-vector "$(SECURITY_DERIVE_MAX_DENIAL)",) \
		$(if $(SECURITY_DERIVE_MAX_PAIRS),--max-bias-pairs "$(SECURITY_DERIVE_MAX_PAIRS)",) \
		$(if $(filter 1 true yes,$(SECURITY_DERIVE_MERGE_SEED)),--merge-seed,)

derive-security-worksheet: ## human verification gate: scaffold a review worksheet from derived security cases (SECURITY_DERIVE_CASES=, SECURITY_DERIVE_WORKSHEET=out.csv)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main derive-security-worksheet --cases "$(SECURITY_DERIVE_CASES)" \
		$(if $(SECURITY_DERIVE_WORKSHEET),--out "$(SECURITY_DERIVE_WORKSHEET)",)

cross-check-goldset: ## Data gate: a SECOND frontier re-confirms grounding/support on a draft BUNDLE (CROSS_CHECK_MODEL=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<draft dir with goldset.jsonl + corpus/>"; exit 1; }
	@test -n "$(CROSS_CHECK_MODEL)" || { echo "ERROR: set CROSS_CHECK_MODEL=<second-frontier id, != the drafter>"; exit 1; }
	$(PY) -m llb.main cross-check-goldset --goldset "$(BUNDLE)/goldset.jsonl" --corpus "$(BUNDLE)/corpus" --model "$(CROSS_CHECK_MODEL)"

verify-sample: ## human verification gate: draw a stratified sample from a draft BUNDLE -> verification worksheet (VERIFY_KIND=auto|goldset|chains, VERIFY_N=, VERIFY_SEED=, VERIFY_MERGE=1, VERIFY_ANNOTATORS=k)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<draft dir with goldset.jsonl + corpus/>"; exit 1; }
	$(PY) -m llb.goldset.verify sample --bundle "$(BUNDLE)" --out "$(VERIFY_WS)" -n $(VERIFY_N) --seed $(VERIFY_SEED) --kind "$(VERIFY_KIND)" $(if $(VERIFY_MERGE),--merge) $(if $(VERIFY_ANNOTATORS),--annotators $(VERIFY_ANNOTATORS))

verify-review: ## human verification gate: interactively verify the sampled items (VERIFY_WS=path, VERIFY_ORDER=confidence, SHOW_CROSSCHECK=1 to reveal, START=N, CLEAR=1 to reset)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.goldset.verify review --worksheet "$(VERIFY_WS)" $(if $(START),--start $(START)) $(if $(SHOW_CROSSCHECK),--show-crosscheck) $(if $(CLEAR),--clear) $(if $(VERIFY_ORDER),--order $(VERIFY_ORDER))

verify-adjudicate: ## human verification gate: agreement report (kappa) + adjudication worksheet from multi-reviewer disagreements (BUNDLE=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<the draft dir the samples came from>"; exit 1; }
	$(PY) -m llb.goldset.verify adjudicate --bundle "$(BUNDLE)"

verify-accept: ## human verification gate: acceptance report + emit the accepted-ledger bundle (VERIFY_WS=, BUNDLE=, VERIFY_TOLERANCE=, VERIFY_ACCEPT_POLICY=global|per-stratum|weighted, VERIFY_STRATUM_TOLERANCES="key=tol ...")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<the draft dir the sample came from>"; exit 1; }
	$(PY) -m llb.goldset.verify accept --worksheet "$(VERIFY_WS)" --bundle "$(BUNDLE)" --tolerance $(VERIFY_TOLERANCE) $(if $(VERIFY_ACCEPT_POLICY),--policy $(VERIFY_ACCEPT_POLICY)) $(foreach t,$(VERIFY_STRATUM_TOLERANCES),--stratum-tolerance "$(t)")

chain-goldset-pipeline: ## Draft, validate, gate, and sample chains (CHAIN_CORPUS=, CHAIN_BUNDLE=, CHAIN_WS=, CHAIN_MIN_ACCEPTED=10)
	@test -n "$(CHAIN_CORPUS)" || { echo "ERROR: set CHAIN_CORPUS=<converted-corpus-dir>"; exit 1; }
	@test -n "$(CHAIN_BUNDLE)" || { echo "ERROR: set CHAIN_BUNDLE=<new-draft-bundle-dir>"; exit 1; }
	$(MAKE) --no-print-directory prepare-goldset-draft DRAFT_CORPUS="$(CHAIN_CORPUS)" \
		DRAFT_OUT_DIR="$(CHAIN_BUNDLE)" DRAFT_CHAINS=1 \
		DRAFT_MULTI_HOP_MAX_PATHS="$(CHAIN_MAX_PATHS)" DRAFT_REQUIRE_PASSED_GATES=1
	$(MAKE) --no-print-directory validate-goldset CHAINS="$(CHAIN_BUNDLE)/chains.jsonl" \
		CORPUS="$(CHAIN_BUNDLE)/corpus"
	@chain_count="$$(wc -l < "$(CHAIN_BUNDLE)/chains.jsonl")"; \
	if [ "$$chain_count" -lt "$(CHAIN_MIN_ACCEPTED)" ]; then \
		echo "ERROR: generated chains $$chain_count is below review minimum $(CHAIN_MIN_ACCEPTED)" >&2; \
		exit 1; \
	fi; \
	echo "[chain-goldset] candidate gate: $$chain_count >= $(CHAIN_MIN_ACCEPTED)"
	$(MAKE) --no-print-directory verify-sample BUNDLE="$(CHAIN_BUNDLE)" VERIFY_KIND=chains \
		VERIFY_N="$(CHAIN_VERIFY_N)" VERIFY_WS="$(CHAIN_WS)"
	@echo "[chain-goldset] review: make verify-review VERIFY_WS=$(CHAIN_WS)"

chain-goldset-finalize: ## Gate accepted chains and promote a compact fixture (CHAIN_BUNDLE=, CHAIN_FIXTURE=, CHAIN_MIN_ACCEPTED=10)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(CHAIN_BUNDLE)" || { echo "ERROR: set CHAIN_BUNDLE=<reviewed-draft-bundle-dir>"; exit 1; }
	@test -n "$(CHAIN_FIXTURE)" || { echo "ERROR: set CHAIN_FIXTURE=<new-samples-fixture-dir>"; exit 1; }
	$(PY) -m llb.goldset.promote_chains --bundle "$(CHAIN_BUNDLE)" \
		--out "$(CHAIN_FIXTURE)" --min-chains "$(CHAIN_MIN_ACCEPTED)"
	$(MAKE) --no-print-directory validate-goldset CHAINS="$(CHAIN_FIXTURE)/chains.jsonl" \
		CORPUS="$(CHAIN_FIXTURE)/corpus"

judge-experiment: ## Run fixed UA judge cases against a local OpenAI-compatible endpoint
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(PY) -m llb.main judge-experiment --judge-model "$(JUDGE_MODEL)" \
		$(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",)

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
	if [ -n "$(DRAFT_DEDUP_AGAINST)" ]; then args+=(--dedup-against "$(DRAFT_DEDUP_AGAINST)"); fi; \
	if [ -n "$(DRAFT_GRAPH_DIR)" ]; then args+=(--graph-dir "$(DRAFT_GRAPH_DIR)"); fi; \
	if [ -n "$(DRAFT_REJECTION_FEEDBACK)" ]; then args+=(--rejection-feedback "$(DRAFT_REJECTION_FEEDBACK)"); fi; \
	if [ "$(DRAFT_NO_THINK)" = "1" ]; then args+=(--no-think); fi; \
	if [ -n "$(DRAFT_NUM_CTX)" ]; then args+=(--num-ctx "$(DRAFT_NUM_CTX)"); fi; \
	$(PY) -m llb.main prepare-goldset-draft "$${args[@]}"

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
