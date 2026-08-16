#!/usr/bin/env bash
# Thin entrypoint for the lock-constrained optional-extras installer / off-lock report.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/shared/common.sh
. "$SCRIPT_DIR/shared/common.sh"
llb_load_env
llb_export_uv_link_mode

export PROJECT_ROOT
PY="$(llb_python)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PY" -m llb.build.extras "$@"
