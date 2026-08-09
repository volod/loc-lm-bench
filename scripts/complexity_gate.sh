#!/usr/bin/env bash
# CI gate: fail on any Radon D-or-worse or cognitive-complexity finding under src/ and tests/.
#
# Runs in `make ci-checks` (so in `make ci` and GitHub CI) beside the acceptance-gate inventory.
# scripts/code_quality.sh runs the same two scans as part of its wider sweep; the thresholds and
# the scans themselves live once, in scripts/shared/complexity.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Label the report blocks as this gate rather than as the wider code-quality sweep (common.sh
# honors a prefix set before it is sourced).
# shellcheck disable=SC2034
LLB_REPORT_PREFIX="complexity-gate"
# shellcheck source=shared/common.sh
. "$SCRIPT_DIR/shared/common.sh"

llb_load_env  # resolve + export DATA_DIR (the complexipy cache lives under it)

# shellcheck source=shared/complexity.sh
. "$SCRIPT_DIR/shared/complexity.sh"

if llb_complexity_gate; then
  echo "[complexity-gate] ok -- no function over Radon ${LLB_RADON_MIN_GRADE} or cognitive ${COGNITIVE_MAX}"
  exit 0
fi

llb_complexity_failure_hint
exit 1
