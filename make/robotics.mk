##@ Robotics

ROBOTICS_CONTRACT_FIXTURE ?= $(PROJECT_ROOT)/samples/robotics/contracts
ROBOTICS_HFLOW_FIXTURE ?= $(PROJECT_ROOT)/samples/robotics/hflow
ROBOTICS_EMULATOR_FIXTURE ?= $(PROJECT_ROOT)/samples/robotics/emulator/scenarios.json
HFLOW_PACKAGE_SPEC := hflow @ git+https://github.com/Hebbian-Robotics/hflow.git@d2e0f3700f2267cfeb0db1957743bb9f5f41256b

.PHONY: robotics-contract-check
robotics-contract-check: ## Validate pinned robotics contracts and write an offline report
	@$(PY) -m llb.main robotics-contract-check \
		--fixture-dir "$(ROBOTICS_CONTRACT_FIXTURE)" \
		--data-dir "$(DATA_DIR)"

.PHONY: robotics-evidence-replay
robotics-evidence-replay: ## Replay the pinned HFlow fixture without network access
	@$(PY) -m llb.main robotics-evidence-bridge \
		--fixture-dir "$(ROBOTICS_HFLOW_FIXTURE)" \
		--data-dir "$(DATA_DIR)"

.PHONY: robotics-evidence-fixture
robotics-evidence-fixture: ## Run the exact pinned HFlow app.test integration
	@bash -c 'source "$(PROJECT_ROOT)/scripts/shared/common.sh"; llb_load_env; \
		uv run --isolated --extra robotics --with "$(HFLOW_PACKAGE_SPEC)" \
		python -m llb.main robotics-hflow-integration --data-dir "$$DATA_DIR"'

.PHONY: test-robotics-emulator
test-robotics-emulator: ## Test and replay the deterministic robotics action gate and emulator
	@$(PY) -m pytest tests/llb/robotics/test_action_gate.py \
		tests/llb/robotics/test_device_emulator.py \
		tests/llb/robotics/test_emulator_run.py $(PYTEST_CACHE_OPT)
	@$(PY) -m llb.main robotics-emulator-check \
		--fixture "$(ROBOTICS_EMULATOR_FIXTURE)" \
		--data-dir "$(DATA_DIR)"
