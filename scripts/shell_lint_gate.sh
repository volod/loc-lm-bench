#!/usr/bin/env bash
# CI gate: fail on any `bash -n` syntax error or shellcheck warning in a tracked-or-new *.sh.
#
# Runs in `make ci-checks` (so in `make ci` and GitHub CI) beside the complexity gate.
# scripts/code_quality.sh runs the same two scans as part of its wider sweep; the scans, the
# severity, and the missing-binary policy live once, in scripts/shared/shell_lint.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Label the report blocks as this gate rather than as the wider code-quality sweep (common.sh
# honors a prefix set before it is sourced).
# shellcheck disable=SC2034
LLB_REPORT_PREFIX="shell-lint-gate"
# shellcheck source=shared/common.sh
. "$SCRIPT_DIR/shared/common.sh"

llb_load_env

# shellcheck source=shared/shell_lint.sh
. "$SCRIPT_DIR/shared/shell_lint.sh"

if llb_shell_lint_gate; then
  llb_shell_lint_ok_line
  exit 0
fi

llb_shell_lint_failure_hint
exit 1
