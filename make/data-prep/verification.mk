## Derived security data, human verification gates, and chain promotion.


.PHONY: review-workbench derive-security-cases derive-security-worksheet cross-check-goldset verify-sample \
	verify-review verify-adjudicate verify-accept chain-goldset-pipeline \
	chain-goldset-finalize judge-experiment

review-workbench: ## Open any supported review ledger or run directory (REVIEW_PATH=, START=N)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(REVIEW_PATH)" || { echo "ERROR: set REVIEW_PATH=<ledger-or-run-dir>"; exit 2; }
	$(PY) -m llb.main review "$(REVIEW_PATH)" $(if $(START),--start $(START))

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
