## Probabilistic record linkage: identity clusters over any record table the project holds.

.PHONY: link-records replay-linkage

link-records: ## Fit a linkage model over RECORDS with LINK_SPEC and publish identity clusters (LINK_LABELS= reviewer labels, LINK_METHOD=, LINK_RUN=, LINK_EXAMPLES=); needs ".[linkage]"
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@args=(--records "$(or $(RECORDS),$(PROJECT_ROOT)/samples/linkage/entity_records_uk.jsonl)" \
	  --spec "$(or $(LINK_SPEC),$(PROJECT_ROOT)/samples/linkage/entity_spec_uk.json)"); \
	if [ -n "$(LINK_LABELS)" ]; then args+=(--labels "$(LINK_LABELS)"); fi; \
	if [ -n "$(LINK_METHOD)" ]; then args+=(--method "$(LINK_METHOD)"); fi; \
	if [ -n "$(LINK_RUN)" ]; then args+=(--run "$(LINK_RUN)"); fi; \
	if [ -n "$(LINK_EXAMPLES)" ]; then args+=(--examples "$(LINK_EXAMPLES)"); fi; \
	$(PY) -m llb.main link-records "$${args[@]}"

replay-linkage: ## Re-score RECORDS from LINK_BUNDLE's saved model without re-fitting (LINK_METHOD=, LINK_RUN=, LINK_EXAMPLES=); needs ".[linkage]"
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(LINK_BUNDLE)" || { echo "ERROR: set LINK_BUNDLE=<previous run bundle>"; exit 1; }
	@args=(--records "$(or $(RECORDS),$(PROJECT_ROOT)/samples/linkage/entity_records_uk.jsonl)" \
	  --replay-from "$(LINK_BUNDLE)"); \
	if [ -n "$(LINK_METHOD)" ]; then args+=(--method "$(LINK_METHOD)"); fi; \
	if [ -n "$(LINK_RUN)" ]; then args+=(--run "$(LINK_RUN)"); fi; \
	if [ -n "$(LINK_EXAMPLES)" ]; then args+=(--examples "$(LINK_EXAMPLES)"); fi; \
	$(PY) -m llb.main link-records "$${args[@]}"
