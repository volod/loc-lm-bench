# loc-lm-bench -- developer entrypoints
SHELL := /bin/bash
PROJECT_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
VENV := $(PROJECT_ROOT)/.venv
PY := $(VENV)/bin/python
PYTHON_VERSION := 3.13
DATA_DIR ?= $(shell bash -c 'source "$(PROJECT_ROOT)/scripts/shared/common.sh"; llb_load_env; printf "%s" "$$DATA_DIR"')

.DEFAULT_GOAL := help

include $(PROJECT_ROOT)/make/config.mk
include $(PROJECT_ROOT)/make/quickstart.mk
include $(PROJECT_ROOT)/make/dev.mk
include $(PROJECT_ROOT)/make/data-prep.mk
include $(PROJECT_ROOT)/make/eval.mk
include $(PROJECT_ROOT)/make/models.mk

##@ General
.PHONY: help
help: ## List available targets
	@awk -f "$(PROJECT_ROOT)/make/help.awk" $(MAKEFILE_LIST)
