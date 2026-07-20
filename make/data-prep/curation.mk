## External-draft curation, judge calibration, and query glossary preparation.

.PHONY: curate-drafts import-external-draft calibration-worksheet calibration-run \
	calibration-rate calibration-score frontier-judge-agreement build-query-glossary

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

frontier-judge-agreement: ## Frontier judge authorization: rho vs human + vs local judge and cost/item per provider (FRONTIER_JUDGE_MODELS= FRONTIER_EGRESS_CONSENT=1 FRONTIER_MAX_USD= ; SPENDS MONEY + SENDS ANSWERS OFF-HOST)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(FRONTIER_JUDGE_MODELS)" || { echo "ERROR: set FRONTIER_JUDGE_MODELS=<litellm id>[,<id>...]"; exit 1; }
	@test -n "$(FRONTIER_EGRESS_CONSENT)" || { echo "ERROR: set FRONTIER_EGRESS_CONSENT=1 to approve sending answers to the providers"; exit 1; }
	@test -n "$(FRONTIER_MAX_USD)$(FRONTIER_MAX_CALLS)" || { echo "ERROR: set FRONTIER_MAX_USD and/or FRONTIER_MAX_CALLS"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main frontier-judge-agreement --worksheet "$(CAL_WS)" \
		--models "$(FRONTIER_JUDGE_MODELS)" --goldset "$(GOLDSET)" \
		--scorer-egress-consent --threshold $(FRONTIER_JUDGE_THRESHOLD) \
		$(if $(FRONTIER_MAX_USD),--frontier-max-usd $(FRONTIER_MAX_USD),) \
		$(if $(FRONTIER_MAX_CALLS),--frontier-max-calls $(FRONTIER_MAX_CALLS),) \
		$(if $(FRONTIER_JUDGE_LIMIT),--limit $(FRONTIER_JUDGE_LIMIT),) \
		$(if $(FRONTIER_JUDGE_OUT),--out-dir "$(FRONTIER_JUDGE_OUT)",)

build-query-glossary: ## uk-query-processing: build query_glossary.json from a draft BUNDLE's dictionary candidates (QUERY_GLOSSARY_OUT=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(BUNDLE)" || { echo "ERROR: set BUNDLE=<draft dir with prompt_dictionary_candidates.jsonl>"; exit 1; }
	$(PY) -m llb.main build-query-glossary --bundle "$(BUNDLE)" \
		--out "$(if $(QUERY_GLOSSARY_OUT),$(QUERY_GLOSSARY_OUT),$(BUNDLE)/query_glossary.json)"
