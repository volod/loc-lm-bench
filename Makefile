# loc-lm-bench -- developer entrypoints
SHELL := /bin/bash
PROJECT_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
# Defaults live in pyproject.toml [tool.llb.toolchain]. `make venv PYTHON_VERSION=` / `VENV=`
# still override (Make command-line variables). PYTHONPATH is set so this runs before `.venv`.
PYTHON_VERSION := $(shell PYTHONPATH="$(PROJECT_ROOT)/src" python3 -m llb.build.toolchain python-version)
VENV_NAME := $(shell PYTHONPATH="$(PROJECT_ROOT)/src" python3 -m llb.build.toolchain venv)
VENV := $(PROJECT_ROOT)/$(VENV_NAME)
PY := $(VENV)/bin/python
ifeq ($(strip $(PYTHON_VERSION)),)
$(error unread [tool.llb.toolchain] python-version from pyproject.toml)
endif
ifeq ($(strip $(VENV_NAME)),)
$(error unread [tool.llb.toolchain] venv from pyproject.toml)
endif
comma := ,
DATA_DIR ?= $(shell bash -c 'source "$(PROJECT_ROOT)/scripts/shared/common.sh"; llb_load_env; printf "%s" "$$DATA_DIR"')

.DEFAULT_GOAL := help

include $(PROJECT_ROOT)/make/config.mk
include $(PROJECT_ROOT)/make/quickstart.mk
include $(PROJECT_ROOT)/make/dev.mk
include $(PROJECT_ROOT)/make/data-prep.mk
include $(PROJECT_ROOT)/make/eval.mk
include $(PROJECT_ROOT)/make/models.mk
include $(PROJECT_ROOT)/make/robotics.mk

##@ General
.PHONY: help
help: ## List available targets
	@awk -f "$(PROJECT_ROOT)/make/help.awk" $(MAKEFILE_LIST)
