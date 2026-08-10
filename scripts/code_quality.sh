#!/usr/bin/env bash
# Repo hygiene sweep: largest tracked files, Python complexity, markdown lint, shell checks.
# Everything here is a report EXCEPT the shell-lint and complexity scans, which are the same hard
# gates `make ci-checks` runs (scripts/shell_lint_gate.sh, scripts/complexity_gate.sh); this script
# exits non-zero on their findings.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=shared/common.sh
. "$SCRIPT_DIR/shared/common.sh"

llb_load_env  # resolve + export DATA_DIR (default $PROJECT_ROOT/.data)

# The gated scans (all four also run in `make ci`, via their own entrypoints).
# shellcheck source=shared/complexity.sh
. "$SCRIPT_DIR/shared/complexity.sh"
# shellcheck source=shared/shell_lint.sh
. "$SCRIPT_DIR/shared/shell_lint.sh"

TOP_K="${1:-10}"
LONGEST_TOP_K="${LONGEST_TOP_K:-20}"
PYMARKDOWN="${PROJECT_ROOT}/.venv/bin/pymarkdown"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

ROOT_MARKDOWN=(README.md AGENTS.md CLAUDE.md GEMINI.md)

llb_markdown_scan() {
  local label="$1"
  shift
  local output
  set +e
  output="$(
    cd "$PROJECT_ROOT"
    "$PYMARKDOWN" --continue-on-error --log-level WARNING scan "$@" 2>&1 || true
  )"
  set -e
  if [ -n "$output" ]; then
    llb_print_block "$label" "$output"
  fi
}

llb_largest_tracked_files() {
  local label="$1"
  local python_filter="$2"
  local output
  output="$(
    set +o pipefail
    git -C "$PROJECT_ROOT" ls-tree -r --long -z HEAD \
      | awk -v RS='\0' -v python_filter="$python_filter" '
          {
            split($0, parts, "\t")
            split(parts[1], meta, " ")
            size = meta[4]
            path = parts[2]
            is_python = path ~ /\.py$/
            if ((python_filter == "yes" && is_python) || (python_filter == "no" && !is_python)) {
              printf "%s\t%s\n", size, path
            }
          }
        ' \
      | sort -k 1 -n -r \
      | sed -n "1,${TOP_K}p" \
      | awk -F '\t' '{printf "%-10s %s\n", $1, $2}'
  )"
  llb_print_block "$label" "$output"
}

llb_files_over_line_limit() {
  # Soft limit: tracked .py/.sh files should stay at or under $LINE_SOFT_LIMIT lines (AGENTS.md).
  # Reports offenders largest-first; informational only (never fails the run).
  local label="$1"
  local limit="${LINE_SOFT_LIMIT:-250}"
  local output
  output="$(
    set +o pipefail
    git -C "$PROJECT_ROOT" ls-files -z '*.py' '*.sh' \
      | (cd "$PROJECT_ROOT" && xargs -0 wc -l 2>/dev/null) \
      | awk -v limit="$limit" '$2 != "total" && $1 > limit {printf "%-8s %s\n", $1, $2}' \
      | sort -k 1 -n -r
  )"
  if [ -n "$output" ]; then
    llb_print_block "$label" "$output"
  fi
}

llb_longest_code_files() {
  # Top-N longest tracked code files by line count (py/sh/mk/awk/Makefile), largest-first. The
  # ~$LINE_SOFT_LIMIT-line soft limit (AGENTS.md) applies to .py/.sh; make/awk rows are context.
  local label="$1"
  local top="${LONGEST_TOP_K:-20}"
  local output
  output="$(
    set +o pipefail
    git -C "$PROJECT_ROOT" ls-files -z '*.py' '*.sh' '*.mk' '*.awk' 'Makefile' \
      | (cd "$PROJECT_ROOT" && xargs -0 wc -l 2>/dev/null) \
      | awk '$2 != "total" {printf "%-8s %s\n", $1, $2}' \
      | sort -k 1 -n -r \
      | sed -n "1,${top}p"
  )"
  if [ -n "$output" ]; then
    llb_print_block "$label" "$output"
  fi
}

llb_check_root_files() {
  llb_report_if_output \
    "project root files (pyproject.toml, Makefile, root markdown)" \
    bash -c '
      cd "$1"
      "$2" -c "import tomllib; tomllib.load(open(\"pyproject.toml\", \"rb\"))"
      make -n help >/dev/null
    ' _ "$PROJECT_ROOT" "$PYTHON"
}

llb_largest_tracked_files "top ${TOP_K} largest tracked Python files (bytes, path)" yes
llb_largest_tracked_files "top ${TOP_K} largest tracked non-Python files (bytes, path)" no

llb_longest_code_files "top ${LONGEST_TOP_K} longest tracked code files (lines, path; py/sh/mk/awk/Makefile)"

llb_files_over_line_limit "tracked .py/.sh files over the ${LINE_SOFT_LIMIT:-250}-line soft limit"

if [ ! -x "$PYMARKDOWN" ] || [ ! -x "$PYTHON" ]; then
  echo "ERROR: dev tools missing in .venv -- run 'make venv EXTRAS=dev' first" >&2
  exit 1
fi
llb_complexity_tools_ready

llb_check_root_files
llb_report_if_output "experiment acceptance-gate inventory" \
  "$PYTHON" -m llb.quality.acceptance_gates --check
llb_markdown_scan "project root markdown" "${ROOT_MARKDOWN[@]}"
llb_markdown_scan "docs markdown (recursive)" -r docs

llb_maintainability_report

# From here down the sweep runs the same hard gates `make ci-checks` runs
# (scripts/shell_lint_gate.sh, scripts/complexity_gate.sh), so a finding fails this run too.
# Everything above stays informational, including the soft line-limit report (AGENTS.md keeps THAT
# limit soft on purpose). Both gates run before either exits, so one sweep shows every finding.
GATE_FAILED=0
llb_shell_lint_gate || { llb_shell_lint_failure_hint; GATE_FAILED=1; }
llb_complexity_gate || { llb_complexity_failure_hint; GATE_FAILED=1; }
exit "$GATE_FAILED"
