## Cross-harness, category-composite, and platform-matrix evaluation.

.PHONY: agentic-harness-compare bench-agentic-loop bench-agentic-loop-repeat-power \
	bench-agentic-loop-repeat-feedback bench-agentic-loop-repeat-feedback-generalization \
	bench-agentic-loop-repeat-feedback-family-adaptation \
	bench-agentic-loop-repeat-feedback-task-family-transfer \
	bench-agentic-loop-repeat-feedback-controller-authority-transfer \
	bench-agentic-loop-controller-channel-authority \
	bench-agentic-loop-controller-channel-cross-model \
	bench-agentic-loop-controller-preamble-placement \
	bench-agentic-context \
	bench-agentic-context-sweep \
	prepare-agentic-long-transcript bench-agentic-context-keep-long \
	bench-agentic-context-compact-long prepare-agentic-memory-transcript \
	bench-agentic-context-compact-memory bench-agentic-context-compact-memory-transfer \
	bench-agentic-context-compact-memory-replication \
	bench-agentic-context-compact-memory-boundary-surface \
	bench-agentic-context-compact-trigger-collapse \
	bench-agentic-context-compact-fold-step \
	bench-agentic-context-compact-repeated-fold \
	bench-agentic-context-compact-summary-input-cap \
	bench-agentic-context-compact-window-elision \
	bench-agentic-context-compact-crossover-restatement \
	bench-agentic-published-provenance \
	bench-agentic-policy-change-audit \
	bench-chain-context composite-headline platform-matrix

agentic-harness-compare: ## Run loop/langgraph/crewai agentic cells, then compare harnesses
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@for harness in $(AGENTIC_HARNESSES); do \
		$(MAKE) --no-print-directory bench-agentic AGENTIC_HARNESS="$$harness" || exit 1; \
	done
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-compare --model "$(MODEL)"

bench-agentic-loop: ## Sweep max steps, malformed-call, and repeated-call policy; recommend one cell per model (AGENT_MAX_STEPS= AGENT_MALFORMED_POLICY= AGENT_REPEATED_CALL_POLICY= MODEL= BACKEND=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-loop --tasks "$(AGENT_LOOP_TASKS)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		--agent-max-steps "$(AGENT_MAX_STEPS)" \
		--agent-malformed-policy "$(AGENT_MALFORMED_POLICY)" \
		--agent-repeated-call-policy "$(AGENT_REPEATED_CALL_POLICY)" \
		--agent-repeat-feedback "$(AGENT_REPEAT_FEEDBACK)" \
		$(if $(AGENT_LOOP_POWER_DESIGN),--repeat-power-design "$(AGENT_LOOP_POWER_DESIGN)",) \
		$(if $(AGENT_LOOP_FEEDBACK_DESIGN),--repeat-feedback-design "$(AGENT_LOOP_FEEDBACK_DESIGN)",) \
		$(if $(AGENT_LOOP_MODEL_FAMILY),--model-family "$(AGENT_LOOP_MODEL_FAMILY)",) \
		$(if $(AGENT_LOOP_MAX_PROMPT_CHARS),--max-prompt-chars "$(AGENT_LOOP_MAX_PROMPT_CHARS)",) \
		$(if $(AGENT_LOOP_BASE_URL),--base-url "$(AGENT_LOOP_BASE_URL)",) \
		$(if $(AGENT_LOOP_MAX_MODEL_LEN),--max-model-len "$(AGENT_LOOP_MAX_MODEL_LEN)",)

bench-agentic-loop-repeat-power: ## Run the predeclared allow-vs-noop power study over two local model families
	@for entry in $(AGENT_LOOP_REPEAT_POWER_ROSTER); do \
		family="$${entry%%=*}"; model="$${entry#*=}"; \
		$(MAKE) --no-print-directory bench-agentic-loop \
			AGENT_LOOP_TASKS="$(AGENT_LOOP_REPEAT_POWER_TASKS)" \
			AGENT_LOOP_POWER_DESIGN="$(AGENT_LOOP_REPEAT_POWER_DESIGN)" \
			AGENT_LOOP_MODEL_FAMILY="$$family" MODEL="$$model" BACKEND=ollama \
			AGENT_MAX_STEPS=6 AGENT_MALFORMED_POLICY=answer \
			AGENT_REPEATED_CALL_POLICY=allow,noop || exit 1; \
	done

bench-agentic-loop-repeat-feedback: ## Compare current, Ukrainian, and bilingual repeat feedback over the powered roster
	@for entry in $(AGENT_LOOP_REPEAT_POWER_ROSTER); do \
		family="$${entry%%=*}"; model="$${entry#*=}"; \
		$(MAKE) --no-print-directory bench-agentic-loop \
			AGENT_LOOP_TASKS="$(AGENT_LOOP_REPEAT_POWER_TASKS)" \
			AGENT_LOOP_FEEDBACK_DESIGN="$(AGENT_LOOP_FEEDBACK_DESIGN_FILE)" \
			AGENT_LOOP_MODEL_FAMILY="$$family" MODEL="$$model" BACKEND=ollama \
			AGENT_MAX_STEPS=6 AGENT_MALFORMED_POLICY=answer \
			AGENT_REPEATED_CALL_POLICY=allow,noop \
			AGENT_REPEAT_FEEDBACK=current,uk,bilingual || exit 1; \
	done

bench-agentic-loop-repeat-feedback-generalization: ## Run the predeclared cross-family, multi-seed bilingual-feedback study
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-loop-repeat-feedback-generalization \
		--design "$(AGENT_LOOP_FEEDBACK_GENERALIZATION_DESIGN)" \
		--tasks "$(AGENT_LOOP_FEEDBACK_GENERALIZATION_TASKS)"

bench-agentic-loop-repeat-feedback-family-adaptation: ## Run predeclared two-seed family-adapted feedback candidates
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-loop-repeat-feedback-family-adaptation \
		--design "$(AGENT_LOOP_FEEDBACK_ADAPTATION_DESIGN)" \
		--tasks "$(AGENT_LOOP_FEEDBACK_ADAPTATION_TASKS)"

bench-agentic-loop-repeat-feedback-task-family-transfer: ## Run the predeclared two-seed Gemma task-family transfer study
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-loop-repeat-feedback-task-family-transfer \
		--design "$(AGENT_LOOP_FEEDBACK_TRANSFER_DESIGN)" \
		--tasks "$(AGENT_LOOP_FEEDBACK_TRANSFER_TASKS)"

bench-agentic-loop-repeat-feedback-controller-authority-transfer: ## Run the predeclared two-seed Gemma authority-transfer study
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-loop-repeat-feedback-controller-authority-transfer \
		--design "$(AGENT_LOOP_FEEDBACK_AUTHORITY_DESIGN)" \
		--tasks "$(AGENT_LOOP_FEEDBACK_AUTHORITY_TASKS)"

bench-agentic-loop-controller-channel-authority: ## Compare identical authority text as observation versus controller role
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-loop-controller-channel-authority \
		--design "$(AGENT_LOOP_CONTROLLER_CHANNEL_DESIGN)" \
		--tasks "$(AGENT_LOOP_CONTROLLER_CHANNEL_TASKS)"

bench-agentic-loop-controller-channel-cross-model: ## Transfer the controller-channel comparison to the predeclared non-Gemma family
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-loop-controller-channel-authority \
		--design "$(AGENT_LOOP_CONTROLLER_CHANNEL_CROSS_MODEL_DESIGN)" \
		--tasks "$(AGENT_LOOP_CONTROLLER_CHANNEL_CROSS_MODEL_TASKS)"

bench-agentic-loop-controller-preamble-placement: ## Compare observation against native system preamble placement on Gemma and Qwen
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-loop-controller-channel-authority \
		--design "$(AGENT_LOOP_CONTROLLER_PREAMBLE_DESIGN)" \
		--tasks "$(AGENT_LOOP_CONTROLLER_PREAMBLE_TASKS)"

bench-agentic-context: ## Agent context-policy benchmark: rank full/observation_cap/keep_last_n/compact for one model over one agentic task set (AGENT_CONTEXT_POLICIES= MODEL= BACKEND= AGENT_CONTEXT_MAX_PROMPT_CHARS=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-context --tasks "$(AGENT_CONTEXT_TASKS)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		--policies "$(AGENT_CONTEXT_POLICIES)" --max-steps "$(AGENT_CONTEXT_MAX_STEPS)" \
		--observation-cap-chars "$(AGENT_CONTEXT_OBSERVATION_CAP_CHARS)" \
		--observation-head-share "$(AGENT_CONTEXT_OBSERVATION_HEAD_SHARE)" \
		--keep-last-n "$(AGENT_CONTEXT_KEEP_LAST_N)" \
		--compact-share "$(AGENT_CONTEXT_COMPACT_SHARE)" \
		$(if $(AGENT_CONTEXT_MAX_PROMPT_CHARS),--max-prompt-chars "$(AGENT_CONTEXT_MAX_PROMPT_CHARS)",) \
		$(if $(AGENT_CONTEXT_BASE_URL),--base-url "$(AGENT_CONTEXT_BASE_URL)",) \
		$(if $(AGENT_CONTEXT_MAX_MODEL_LEN),--max-model-len "$(AGENT_CONTEXT_MAX_MODEL_LEN)",)

bench-agentic-context-sweep: ## Sweep observation_cap_chars / head_share / keep_last_n; pin or expose each (MODEL= BACKEND= AGENT_CONTEXT_SWEEP_TASKS= AGENT_CONTEXT_SWEEP_AXES= AGENT_CONTEXT_SWEEP_MAX_MODEL_LEN=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-context-sweep --tasks "$(AGENT_CONTEXT_SWEEP_TASKS)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		--max-steps "$(AGENT_CONTEXT_SWEEP_MAX_STEPS)" \
		--axes "$(AGENT_CONTEXT_SWEEP_AXES)" \
		$(if $(AGENT_CONTEXT_SWEEP_MAX_PROMPT_CHARS),--max-prompt-chars "$(AGENT_CONTEXT_SWEEP_MAX_PROMPT_CHARS)",) \
		$(if $(AGENT_CONTEXT_SWEEP_BASE_URL),--base-url "$(AGENT_CONTEXT_SWEEP_BASE_URL)",) \
		$(if $(AGENT_CONTEXT_SWEEP_MAX_MODEL_LEN),--max-model-len "$(AGENT_CONTEXT_SWEEP_MAX_MODEL_LEN)",)

prepare-agentic-long-transcript: ## Build medium-obs keep_last_n tasks from fat search set (AGENT_CONTEXT_KEEP_LONG_FROM_SEARCH= AGENT_CONTEXT_KEEP_LONG_TASKS=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main prepare-agentic-long-transcript --out "$(AGENT_CONTEXT_KEEP_LONG_TASKS)" \
		--from-search-tasks "$(AGENT_CONTEXT_KEEP_LONG_FROM_SEARCH)" \
		--max-match-docs "$(AGENT_CONTEXT_KEEP_LONG_MAX_MATCH_DOCS)" \
		--max-other-docs "$(AGENT_CONTEXT_KEEP_LONG_MAX_OTHER_DOCS)" \
		--max-doc-chars "$(AGENT_CONTEXT_KEEP_LONG_MAX_DOC_CHARS)"

bench-agentic-context-keep-long: prepare-agentic-long-transcript ## keep=1/2/3 on long-transcript tasks at raised max_steps (MODEL= BACKEND= AGENT_CONTEXT_KEEP_LONG_MAX_STEPS=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	$(MAKE) --no-print-directory bench-agentic-context-sweep \
		AGENT_CONTEXT_SWEEP_TASKS="$(AGENT_CONTEXT_KEEP_LONG_TASKS)" \
		AGENT_CONTEXT_SWEEP_MAX_STEPS="$(AGENT_CONTEXT_KEEP_LONG_MAX_STEPS)" \
		AGENT_CONTEXT_SWEEP_AXES="$(AGENT_CONTEXT_KEEP_LONG_AXES)" \
		AGENT_CONTEXT_SWEEP_MAX_MODEL_LEN="$(AGENT_CONTEXT_SWEEP_MAX_MODEL_LEN)" \
		AGENT_CONTEXT_SWEEP_MAX_PROMPT_CHARS="$(AGENT_CONTEXT_SWEEP_MAX_PROMPT_CHARS)" \
		AGENT_CONTEXT_SWEEP_BASE_URL="$(AGENT_CONTEXT_SWEEP_BASE_URL)"

bench-agentic-context-compact-long: prepare-agentic-long-transcript ## Active compact vs observation_cap on long transcripts, paired incl summarizer input cost (MODEL= BACKEND=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-compact-vs-cap \
		--tasks "$(AGENT_CONTEXT_COMPACT_LONG_TASKS)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		--max-steps "$(AGENT_CONTEXT_COMPACT_LONG_MAX_STEPS)" \
		--compact-share "$(AGENT_CONTEXT_COMPACT_LONG_COMPACT_SHARE)" \
		--max-prompt-chars "$(AGENT_CONTEXT_COMPACT_LONG_MAX_PROMPT_CHARS)" \
		$(if $(AGENT_CONTEXT_COMPACT_LONG_BASE_URL),--base-url "$(AGENT_CONTEXT_COMPACT_LONG_BASE_URL)",) \
		$(if $(AGENT_CONTEXT_COMPACT_LONG_MAX_MODEL_LEN),--max-model-len "$(AGENT_CONTEXT_COMPACT_LONG_MAX_MODEL_LEN)",)

prepare-agentic-memory-transcript: ## Build read-once memory tasks with externally checked later progress
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main prepare-agentic-memory-transcript \
		--out "$(AGENT_CONTEXT_COMPACT_MEMORY_TASKS)" \
		--n-tasks "$(AGENT_CONTEXT_COMPACT_MEMORY_N_TASKS)" \
		--depth "$(AGENT_CONTEXT_COMPACT_MEMORY_DEPTH)" \
		--pad-chars "$(AGENT_CONTEXT_COMPACT_MEMORY_PAD_CHARS)"

bench-agentic-context-compact-memory: prepare-agentic-memory-transcript ## Compact vs cap when an early read-once fact must survive later tool calls (MODEL= BACKEND=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-compact-vs-cap \
		--tasks "$(AGENT_CONTEXT_COMPACT_MEMORY_TASKS)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		--max-steps "$(AGENT_CONTEXT_COMPACT_MEMORY_MAX_STEPS)" \
		--compact-share "$(AGENT_CONTEXT_COMPACT_MEMORY_COMPACT_SHARE)" \
		--min-compaction-rate "$(AGENT_CONTEXT_COMPACT_MEMORY_MIN_COMPACTION_RATE)" \
		--max-prompt-chars "$(AGENT_CONTEXT_COMPACT_MEMORY_MAX_PROMPT_CHARS)" \
		$(if $(AGENT_CONTEXT_COMPACT_MEMORY_BASE_URL),--base-url "$(AGENT_CONTEXT_COMPACT_MEMORY_BASE_URL)",) \
		$(if $(AGENT_CONTEXT_COMPACT_MEMORY_MAX_MODEL_LEN),--max-model-len "$(AGENT_CONTEXT_COMPACT_MEMORY_MAX_MODEL_LEN)",)

bench-agentic-context-compact-memory-transfer: ## Gate a non-Qwen model, then run the compact-memory depth/trigger matrix
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-context-compact-memory-transfer \
		--design "$(AGENT_CONTEXT_COMPACT_MEMORY_TRANSFER_DESIGN)"

bench-agentic-context-compact-memory-replication: ## Replicate compact memory on a second family with tighter evidence and a cap-fitting cell
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-context-compact-memory-replication \
		--design "$(AGENT_CONTEXT_COMPACT_MEMORY_REPLICATION_DESIGN)"

bench-agentic-context-compact-memory-boundary-surface: ## Map the cap-fitting compact-versus-cap cost crossover over a predeclared depth/prompt-guard grid
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-context-compact-memory-boundary-surface \
		--design "$(AGENT_CONTEXT_COMPACT_MEMORY_BOUNDARY_SURFACE_DESIGN)"

bench-agentic-context-compact-trigger-collapse: ## Test whether compact_share and prompt guard act only through their product (the compaction trigger)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-context-compact-trigger-collapse \
		--design "$(AGENT_CONTEXT_COMPACT_TRIGGER_COLLAPSE_DESIGN)"

bench-agentic-context-compact-fold-step: ## Test whether the compact cost side flips at a fold-step change rather than at an interpolated char guard
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-context-compact-fold-step \
		--design "$(AGENT_CONTEXT_COMPACT_FOLD_STEP_DESIGN)"

bench-agentic-context-compact-repeated-fold: ## Measure compact-memory completion through repeated folds and attribute survival with a typed-marker ablation
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-context-compact-repeated-fold \
		--design "$(AGENT_CONTEXT_COMPACT_REPEATED_FOLD_DESIGN)"

bench-agentic-context-compact-summary-input-cap: ## Price the compact summarize call's input cap: does pinning it to a step-aligned quantity zero the within-step residual, and what did the elided span cost
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-context-compact-summary-input-cap \
		--design "$(AGENT_CONTEXT_COMPACT_SUMMARY_INPUT_CAP_DESIGN)"

bench-agentic-context-compact-window-elision: ## Price unavoidable middle elision under the shipped window summary-input bound against a trigger-matched fitting control
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-context-compact-window-elision \
		--design "$(AGENT_CONTEXT_COMPACT_WINDOW_ELISION_DESIGN)"

bench-agentic-context-compact-crossover-restatement: ## Restate every published compact crossover under the shipped summarize-input cap, re-measuring only the cells the model-free audit calls bound-sensitive
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-context-compact-crossover-restatement \
		--design "$(AGENT_CONTEXT_COMPACT_CROSSOVER_RESTATEMENT_DESIGN)" \
		$(if $(AGENT_CONTEXT_COMPACT_CROSSOVER_RESTATEMENT_SURFACE),--surface-aggregate "$(AGENT_CONTEXT_COMPACT_CROSSOVER_RESTATEMENT_SURFACE)",) \
		$(if $(filter 1 true yes,$(AGENT_CONTEXT_COMPACT_CROSSOVER_AUDIT_ONLY)),--audit-only,)

bench-agentic-published-provenance: ## Re-commit the run aggregates EVERY registered published-value design resolves against (the union, so no study's evidence is pruned) and their content pins, from the artifacts under DATA_DIR, then name any published value that evidence no longer states (exit 3; the write still stands)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-context-compact-crossover-restatement \
		--design "$(AGENT_CONTEXT_COMPACT_CROSSOVER_RESTATEMENT_DESIGN)" \
		--refresh-provenance

bench-agentic-policy-change-audit: ## Report which published agentic numbers an agent context-policy constant change invalidates, with no GPU (POLICY_FIELD= POLICY_BASELINE= POLICY_CANDIDATE=; space-separated lists audit a compound change as ONE change)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(POLICY_FIELD)" || { echo "ERROR: set POLICY_FIELD=<context-policy constant(s)>"; exit 1; }
	@test -n "$(POLICY_BASELINE)" || { echo "ERROR: set POLICY_BASELINE=<value(s) the evidence was measured under>"; exit 1; }
	@test -n "$(POLICY_CANDIDATE)" || { echo "ERROR: set POLICY_CANDIDATE=<value(s) being considered>"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-agentic-policy-change-audit \
		$(foreach f,$(POLICY_FIELD),--field "$(f)") \
		$(foreach v,$(POLICY_BASELINE),--baseline "$(v)") \
		$(foreach v,$(POLICY_CANDIDATE),--candidate "$(v)")

bench-chain-context: ## Context-policy benchmark: rank fresh/history/summary/roles for one model over a verified chain set (CHAIN_CONTEXT_MODEL= CHAIN_CONTEXT_BACKEND= CHAIN_CONTEXT_CHAINS= CHAIN_CONTEXT_CORPUS= CHAIN_CONTEXT_POLICIES=)
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-chain-context --chains "$(CHAIN_CONTEXT_CHAINS)" \
		--model "$(CHAIN_CONTEXT_MODEL)" --backend "$(CHAIN_CONTEXT_BACKEND)" \
		--corpus "$(CHAIN_CONTEXT_CORPUS)" --policies "$(CHAIN_CONTEXT_POLICIES)" \
		--top-k "$(CHAIN_CONTEXT_TOP_K)" \
		$(if $(CHAIN_CONTEXT_INDEX_DIR),--index-dir "$(CHAIN_CONTEXT_INDEX_DIR)",) \
		$(if $(CHAIN_CONTEXT_BASE_URL),--base-url "$(CHAIN_CONTEXT_BASE_URL)",) \
		$(if $(CHAIN_CONTEXT_MAX_MODEL_LEN),--max-model-len "$(CHAIN_CONTEXT_MAX_MODEL_LEN)",) \
		$(if $(filter 1 true yes,$(CHAIN_CONTEXT_DATA_VERIFIED)),--data-verified,) \
		$(if $(CHAIN_CONTEXT_VERIFICATION_REF),--verification-ref "$(CHAIN_CONTEXT_VERIFICATION_REF)",)

composite-headline: ## Run the verified category suite for MODEL, then require a clean bench-composite preflight
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	@test -n "$(COMPOSITE_TEXT_ANALYSIS_BUNDLE)" || { echo "ERROR: set COMPOSITE_TEXT_ANALYSIS_BUNDLE=<verified text-analysis bundle>"; exit 1; }
	@test -n "$(COMPOSITE_TEXT_ANALYSIS_VERIFICATION_REF)" || { echo "ERROR: set COMPOSITE_TEXT_ANALYSIS_VERIFICATION_REF or COMPOSITE_VERIFICATION_REF"; exit 1; }
	@test -n "$(COMPOSITE_SUMMARIZATION_VERIFICATION_REF)" || { echo "ERROR: set COMPOSITE_SUMMARIZATION_VERIFICATION_REF or COMPOSITE_VERIFICATION_REF"; exit 1; }
	@test -n "$(COMPOSITE_STRUCTURED_VERIFICATION_REF)" || { echo "ERROR: set COMPOSITE_STRUCTURED_VERIFICATION_REF or COMPOSITE_VERIFICATION_REF"; exit 1; }
	@test -n "$(COMPOSITE_SECURITY_VERIFICATION_REF)" || { echo "ERROR: set COMPOSITE_SECURITY_VERIFICATION_REF or COMPOSITE_VERIFICATION_REF"; exit 1; }
	@test -n "$(COMPOSITE_AGENTIC_VERIFICATION_REF)" || { echo "ERROR: set COMPOSITE_AGENTIC_VERIFICATION_REF or COMPOSITE_VERIFICATION_REF"; exit 1; }
	@test -n "$(COMPOSITE_TOOLING_VERIFICATION_REF)" || { echo "ERROR: set COMPOSITE_TOOLING_VERIFICATION_REF or COMPOSITE_VERIFICATION_REF"; exit 1; }
	set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	$(PY) -m llb.main bench-text-analysis --bundle "$(COMPOSITE_TEXT_ANALYSIS_BUNDLE)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		$(if $(COMPOSITE_BASE_URL),--base-url "$(COMPOSITE_BASE_URL)",) \
		$(if $(COMPOSITE_REAL_CORPUS),--real-corpus,) \
		$(if $(JUDGE_RHO),--judge-rho "$(JUDGE_RHO)" --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",),) \
		--data-verified --verification-ref "$(COMPOSITE_TEXT_ANALYSIS_VERIFICATION_REF)" && \
	$(PY) -m llb.main bench-summarization --cases "$(COMPOSITE_SUMMARIZATION_CASES)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		$(if $(COMPOSITE_BASE_URL),--base-url "$(COMPOSITE_BASE_URL)",) \
		$(if $(JUDGE_RHO),--judge-rho "$(JUDGE_RHO)" --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",),) \
		--data-verified --verification-ref "$(COMPOSITE_SUMMARIZATION_VERIFICATION_REF)" && \
	$(PY) -m llb.main bench-structured --cases "$(COMPOSITE_STRUCTURED_CASES)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		$(if $(COMPOSITE_BASE_URL),--base-url "$(COMPOSITE_BASE_URL)",) \
		--data-verified --verification-ref "$(COMPOSITE_STRUCTURED_VERIFICATION_REF)" && \
	$(PY) -m llb.main bench-security --cases "$(COMPOSITE_SECURITY_CASES)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		$(if $(COMPOSITE_BASE_URL),--base-url "$(COMPOSITE_BASE_URL)",) \
		$(if $(JUDGE_RHO),--judge-rho "$(JUDGE_RHO)" --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",),) \
		--data-verified --verification-ref "$(COMPOSITE_SECURITY_VERIFICATION_REF)" && \
	$(PY) -m llb.main bench-agentic --tasks "$(COMPOSITE_AGENTIC_TASKS)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		$(if $(COMPOSITE_BASE_URL),--base-url "$(COMPOSITE_BASE_URL)",) \
		$(if $(JUDGE_RHO),--judge-rho "$(JUDGE_RHO)" --judge-model "$(JUDGE_MODEL)" $(if $(JUDGE_BASE_URL),--judge-base-url "$(JUDGE_BASE_URL)",),) \
		--data-verified --verification-ref "$(COMPOSITE_AGENTIC_VERIFICATION_REF)" && \
	$(PY) -m llb.main bench-tooling --catalog "$(COMPOSITE_TOOLING_CATALOG)" \
		--model "$(MODEL)" --backend "$(BACKEND)" \
		$(if $(COMPOSITE_BASE_URL),--base-url "$(COMPOSITE_BASE_URL)",) \
		--data-verified --verification-ref "$(COMPOSITE_TOOLING_VERIFICATION_REF)" && \
	$(PY) -m llb.main bench-composite

platform-matrix: ## Run same logical model base across Ollama, vLLM, and llama.cpp with telemetry
	@test -x "$(PY)" || { echo "ERROR: .venv missing -- run 'make venv' first"; exit 1; }
	HF_HUB_OFFLINE="$(HF_HUB_OFFLINE)" $(MAKE) --no-print-directory build-index
	@set -a; [ -f "$(PROJECT_ROOT)/.env" ] && . "$(PROJECT_ROOT)/.env"; set +a; export DATA_DIR="$(DATA_DIR)"; \
	wants_backend() { case " $(PLATFORM_MATRIX_BACKENDS) " in *" $$1 "*) return 0 ;; *) return 1 ;; esac; }; \
	record_failure() { failed=1; echo "[platform-matrix] failed $$1 (continuing; set PLATFORM_MATRIX_STRICT=1 to fail fast)"; }; \
	ran=0; failed=0; \
	if wants_backend ollama; then \
	  echo "[platform-matrix] run ollama model=$(PLATFORM_MATRIX_OLLAMA_MODEL)"; \
	  if $(PY) -m llb.main run-eval --model "$(PLATFORM_MATRIX_OLLAMA_MODEL)" --backend ollama \
	    --goldset "$(PLATFORM_MATRIX_GOLDSET)" --split "$(PLATFORM_MATRIX_SPLIT)" --limit "$(PLATFORM_MATRIX_LIMIT)" \
	    --telemetry; then ran=$$((ran + 1)); else record_failure ollama; fi; \
	fi; \
	if wants_backend vllm; then \
	  if [ -x "$(VENV)/bin/vllm" ] || command -v vllm >/dev/null 2>&1; then \
	    echo "[platform-matrix] run vllm model=$(PLATFORM_MATRIX_VLLM_MODEL)"; \
	    if $(PY) -m llb.main run-eval --model "$(PLATFORM_MATRIX_VLLM_MODEL)" --backend vllm \
	      --goldset "$(PLATFORM_MATRIX_GOLDSET)" --split "$(PLATFORM_MATRIX_SPLIT)" --limit "$(PLATFORM_MATRIX_LIMIT)" \
	      --telemetry --max-model-len "$(PLATFORM_MATRIX_MAX_MODEL_LEN)" \
	      --gpu-memory-utilization "$(PLATFORM_MATRIX_GPU_MEMORY_UTILIZATION)" --evict; then ran=$$((ran + 1)); else record_failure vllm; fi; \
	  else \
	    echo "[platform-matrix] skipped vllm: vllm executable not found (run make build-vllm)"; \
	    [ "$(PLATFORM_MATRIX_STRICT)" = "1" ] && failed=1; \
	  fi; \
	fi; \
	if wants_backend llamacpp; then \
	  llama_bin="$$DATA_DIR/llb/llamacpp/build/bin/llama-server"; \
	  if [ -x "$$llama_bin" ] || command -v llama-server >/dev/null 2>&1; then \
	    echo "[platform-matrix] run llamacpp model=$(PLATFORM_MATRIX_LLAMACPP_MODEL)"; \
	    if $(PY) -m llb.main run-eval --model "$(PLATFORM_MATRIX_LLAMACPP_MODEL)" --backend llamacpp \
	      --goldset "$(PLATFORM_MATRIX_GOLDSET)" --split "$(PLATFORM_MATRIX_SPLIT)" --limit "$(PLATFORM_MATRIX_LIMIT)" \
	      --telemetry --max-model-len "$(PLATFORM_MATRIX_MAX_MODEL_LEN)" \
	      --gpu-layers "$(PLATFORM_MATRIX_LLAMACPP_GPU_LAYERS)"; then ran=$$((ran + 1)); else record_failure llamacpp; fi; \
	  else \
	    echo "[platform-matrix] skipped llamacpp: llama-server not found (run make build-llamacpp)"; \
	    [ "$(PLATFORM_MATRIX_STRICT)" = "1" ] && failed=1; \
	  fi; \
	fi; \
	if [ "$$ran" -eq 0 ]; then echo "ERROR: platform-matrix produced no successful backend rows" >&2; exit 1; fi; \
	if [ "$(PLATFORM_MATRIX_STRICT)" = "1" ] && [ "$$failed" -ne 0 ]; then exit 1; fi; \
	echo "[platform-matrix] successful backend rows: $$ran"
