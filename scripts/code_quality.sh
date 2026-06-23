#!/usr/bin/env bash
# Repo hygiene: largest tracked files, Python complexity, markdown lint, shell checks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=shared/common.sh
. "$SCRIPT_DIR/shared/common.sh"

TOP_K="${1:-10}"
RADON="${PROJECT_ROOT}/.venv/bin/radon"
COMPLEXIPY="${PROJECT_ROOT}/.venv/bin/complexipy"
PYMARKDOWN="${PROJECT_ROOT}/.venv/bin/pymarkdown"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
COGNITIVE_MAX="${COGNITIVE_MAX:-15}"

ROOT_MARKDOWN=(README.md AGENTS.md CLAUDE.md GEMINI.md)

llb_report() {
  local label="$1"
  shift
  echo
  echo "[code-quality] ${label}"
  "$@"
}

llb_markdown_scan() {
  local label="$1"
  shift
  echo
  echo "[code-quality] ${label}"
  (
    cd "$PROJECT_ROOT"
    "$PYMARKDOWN" --continue-on-error --log-level WARNING scan "$@" || true
  )
}

llb_check_shell_scripts() {
  local script
  llb_report "shell syntax (bash -n) under scripts/"
  while IFS= read -r -d '' script; do
    bash -n "$script"
    printf '  [ok] %s\n' "${script#"$PROJECT_ROOT"/}"
  done < <(find "$PROJECT_ROOT/scripts" -type f -name '*.sh' -print0 | sort -z)

  if command -v shellcheck >/dev/null 2>&1; then
    llb_report "shell lint (shellcheck) under scripts/"
    while IFS= read -r -d '' script; do
      shellcheck -S warning "$script"
      printf '  [ok] %s\n' "${script#"$PROJECT_ROOT"/}"
    done < <(find "$PROJECT_ROOT/scripts" -type f -name '*.sh' -print0 | sort -z)
  else
    echo
    echo "[code-quality] shell lint (shellcheck) skipped -- run: make apt-deps APT_PROFILE=dev"
  fi
}

echo "[code-quality] top ${TOP_K} largest tracked files (bytes, path)"
(
  set +o pipefail
  git -C "$PROJECT_ROOT" ls-tree -r --long HEAD \
    | sort -k 4 -n -r \
    | head -n "$TOP_K" \
    | awk '{printf "%-10s %s\n", $4, $5}'
)

if [ ! -x "$RADON" ] || [ ! -x "$COMPLEXIPY" ] || [ ! -x "$PYMARKDOWN" ] || [ ! -x "$PYTHON" ]; then
  echo "ERROR: dev tools missing in .venv -- run 'make venv EXTRAS=dev' first" >&2
  exit 1
fi

llb_report "project root files (pyproject.toml, Makefile, root markdown)"
(
  cd "$PROJECT_ROOT"
  "$PYTHON" -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
  printf '  [ok] pyproject.toml\n'
  make -n help >/dev/null
  printf '  [ok] Makefile\n'
)
llb_markdown_scan "project root markdown" "${ROOT_MARKDOWN[@]}"

llb_check_shell_scripts

llb_markdown_scan "docs markdown (recursive)" -r docs

llb_report "cyclomatic complexity grade D or worse (src tests only)" \
  "$RADON" cc src tests -s -n D

llb_report "maintainability index grade C only (repo root; hidden dirs skipped)" \
  bash -c 'cd "$1" && "$2" mi . -s -n C -x C' _ "$PROJECT_ROOT" "$RADON"

llb_report "cognitive complexity above ${COGNITIVE_MAX} (src tests only)" \
  bash -c 'cd "$1" && "$2" src tests --max-complexity-allowed "$3" --failed --ignore-complexity --color no --plain --sort desc' \
  _ "$PROJECT_ROOT" "$COMPLEXIPY" "$COGNITIVE_MAX"
