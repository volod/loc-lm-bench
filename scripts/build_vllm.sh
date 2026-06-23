#!/usr/bin/env bash
# Thin entrypoint for the vLLM installer/build orchestrator.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/shared/common.sh
. "$SCRIPT_DIR/shared/common.sh"
llb_load_env

export PROJECT_ROOT
MAX_JOBS="$(max_jobs)"
export MAX_JOBS
PY="$(llb_python)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PY" -m llb.build.vllm "$@"
