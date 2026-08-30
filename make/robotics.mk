##@ Robotics

ROBOTICS_CONTRACT_FIXTURE ?= $(PROJECT_ROOT)/samples/robotics/contracts

.PHONY: robotics-contract-check
robotics-contract-check: ## Validate pinned robotics contracts and write an offline report
	@$(PY) -m llb.main robotics-contract-check \
		--fixture-dir "$(ROBOTICS_CONTRACT_FIXTURE)" \
		--data-dir "$(DATA_DIR)"
