#!/usr/bin/env bash
# Repo hygiene: largest tracked files, Python complexity, markdown lint, shell checks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=shared/common.sh
. "$SCRIPT_DIR/shared/common.sh"

llb_load_env  # resolve + export DATA_DIR (default $PROJECT_ROOT/.data)

TOP_K="${1:-10}"
LONGEST_TOP_K="${LONGEST_TOP_K:-20}"
RADON="${PROJECT_ROOT}/.venv/bin/radon"
COMPLEXIPY="${PROJECT_ROOT}/.venv/bin/complexipy"
PYMARKDOWN="${PROJECT_ROOT}/.venv/bin/pymarkdown"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
COGNITIVE_MAX="${COGNITIVE_MAX:-15}"
# complexipy has no cache-dir flag -- it writes .complexipy_cache in its CWD. Run it from the shared
# $DATA_DIR cache tree (against absolute src/tests) so nothing lands in the project root.
COMPLEXIPY_CACHE_DIR="${DATA_DIR}/cache/complexipy"
mkdir -p "$COMPLEXIPY_CACHE_DIR"

ROOT_MARKDOWN=(README.md AGENTS.md CLAUDE.md GEMINI.md)
LLB_PRINTED_BLOCK=0

llb_print_block() {
  local label="$1"
  local output="${2:-}"
  if [ "$LLB_PRINTED_BLOCK" -eq 1 ]; then
    echo
  fi
  echo "[code-quality] ${label}"
  LLB_PRINTED_BLOCK=1
  if [ -n "$output" ]; then
    printf '%s\n' "$output"
  fi
}

llb_report_if_output() {
  local label="$1"
  shift
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  if [ -n "$output" ] || [ "$status" -ne 0 ]; then
    llb_print_block "$label" "$output"
  fi
  return "$status"
}

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

llb_print_script_failure() {
  local label="$1"
  local script="$2"
  local output="$3"
  llb_print_block "$label"
  printf '  [failed] %s\n' "${script#"$PROJECT_ROOT"/}"
  if [ -n "$output" ]; then
    printf '%s\n' "$output" | sed 's/^/    /'
  fi
}

llb_check_shell_syntax() {
  local script output status failed
  failed=0
  while IFS= read -r -d '' script; do
    set +e
    output="$(bash -n "$script" 2>&1)"
    status=$?
    set -e
    if [ "$status" -ne 0 ]; then
      llb_print_script_failure "shell syntax (bash -n) under scripts/" "$script" "$output"
      failed=1
    fi
  done < <(find "$PROJECT_ROOT/scripts" -type f -name '*.sh' -print0 | sort -z)
  return "$failed"
}

llb_check_shellcheck() {
  local script output status failed
  if ! command -v shellcheck >/dev/null 2>&1; then
    llb_print_block "shell lint (shellcheck) skipped -- run: make apt-deps APT_PROFILE=dev"
    return 0
  fi

  failed=0
  while IFS= read -r -d '' script; do
    set +e
    output="$(shellcheck -S warning "$script" 2>&1)"
    status=$?
    set -e
    if [ "$status" -ne 0 ] || [ -n "$output" ]; then
      llb_print_script_failure "shell lint (shellcheck) under scripts/" "$script" "$output"
      [ "$status" -ne 0 ] && failed=1
    fi
  done < <(find "$PROJECT_ROOT/scripts" -type f -name '*.sh' -print0 | sort -z)
  return "$failed"
}

llb_check_shell_scripts() {
  llb_check_shell_syntax
  llb_check_shellcheck
}

llb_largest_tracked_files "top ${TOP_K} largest tracked Python files (bytes, path)" yes
llb_largest_tracked_files "top ${TOP_K} largest tracked non-Python files (bytes, path)" no

llb_longest_code_files "top ${LONGEST_TOP_K} longest tracked code files (lines, path; py/sh/mk/awk/Makefile)"

llb_files_over_line_limit "tracked .py/.sh files over the ${LINE_SOFT_LIMIT:-250}-line soft limit"

if [ ! -x "$RADON" ] || [ ! -x "$COMPLEXIPY" ] || [ ! -x "$PYMARKDOWN" ] || [ ! -x "$PYTHON" ]; then
  echo "ERROR: dev tools missing in .venv -- run 'make venv EXTRAS=dev' first" >&2
  exit 1
fi

llb_check_root_files
llb_markdown_scan "project root markdown" "${ROOT_MARKDOWN[@]}"

llb_check_shell_scripts

llb_markdown_scan "docs markdown (recursive)" -r docs

llb_report_if_output "cyclomatic complexity grade D or worse (src tests only)" \
  "$RADON" cc src tests -s -n D

llb_report_if_output "maintainability index grade C only (repo root; hidden dirs skipped)" \
  bash -c 'cd "$1" && "$2" mi . -s -n C -x C' _ "$PROJECT_ROOT" "$RADON"

llb_report_if_output "cognitive complexity above ${COGNITIVE_MAX} (src tests only)" \
  bash -c 'cd "$4" && "$2" "$1/src" "$1/tests" --max-complexity-allowed "$3" --failed --ignore-complexity --color no --plain --sort desc' \
  _ "$PROJECT_ROOT" "$COMPLEXIPY" "$COGNITIVE_MAX" "$COMPLEXIPY_CACHE_DIR"
