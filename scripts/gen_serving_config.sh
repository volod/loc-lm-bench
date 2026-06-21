#!/usr/bin/env bash
# Generate serve scripts + run-eval configs under .data/llb/serving/gpu-<tier>gb/
# Usage: scripts/gen_serving_config.sh [12|16|24|32]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/shared/common.sh
source "$ROOT/scripts/shared/common.sh"
llb_load_env
PY="$(llb_python)"
TIER="${1:-}"
if [ -n "$TIER" ]; then
  exec "$PY" -m llb.main gen-serving-config --gpu-gb "$TIER"
fi
exec "$PY" -m llb.main gen-serving-config
