## Ontology axiom layer: check an extraction ledger for logical inconsistencies.

.PHONY: validate-ontology-axioms

validate-ontology-axioms: ## Check EXTRACTION (bundle dir or extraction.jsonl; space-separated for several) against AXIOMS (AXIOM_CROSSCHECK=1 adds the OWL reasoner; needs ".[ontology]")
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(EXTRACTION)" || { echo "ERROR: set EXTRACTION=<bundle dir or extraction.jsonl>"; exit 1; }
	@args=(); \
	for ledger in $(EXTRACTION); do args+=(--extraction "$$ledger"); done; \
	args+=(--axioms "$(or $(AXIOMS),$(PROJECT_ROOT)/samples/ontology/axioms_uk_v1.ttl)"); \
	if [ -n "$(AXIOM_CROSSCHECK)" ]; then args+=(--crosscheck); fi; \
	if [ -n "$(AXIOM_RUN)" ]; then args+=(--run "$(AXIOM_RUN)"); fi; \
	if [ -n "$(AXIOM_FAIL_ON_VIOLATIONS)" ]; then args+=(--fail-on-violations); fi; \
	$(PY) -m llb.main validate-ontology-axioms "$${args[@]}"
