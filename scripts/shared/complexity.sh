#!/usr/bin/env bash
# The Python complexity scans. SOURCE this file after scripts/shared/common.sh + llb_load_env.
# llb-requires: common.sh
#
# Cyclomatic and cognitive complexity are HARD gates: a function over either threshold fails the
# run on the commit that introduces it (`make ci-checks` via scripts/complexity_gate.sh), not in a
# later sweep. Neither
# tool signals through its exit status here -- radon always exits 0, and complexipy is asked for a
# plain listing with --ignore-complexity -- so a printed row IS the failure (llb_fail_if_output).
# Maintainability index remains an informational code-quality report: MI also measures module
# volume, which is already governed by the deliberately soft line target. Keep its scope and
# threshold here so every Python complexity scan uses the same src/tests boundary.
#
# Keep the hard-gate thresholds where they ship. Raising one to match whatever the tree happens to
# hold turns the gate back into a report; split the function instead.

LLB_RADON_MIN_GRADE="${LLB_RADON_MIN_GRADE:-D}"
LLB_MI_MIN_GRADE="${LLB_MI_MIN_GRADE:-C}"
COGNITIVE_MAX="${COGNITIVE_MAX:-15}"

LLB_RADON="${LLB_RADON:-$PROJECT_ROOT/.venv/bin/radon}"
LLB_COMPLEXIPY="${LLB_COMPLEXIPY:-$PROJECT_ROOT/.venv/bin/complexipy}"

LLB_CYCLOMATIC_LABEL="cyclomatic complexity grade ${LLB_RADON_MIN_GRADE} or worse (src tests only)"
LLB_MI_LABEL="maintainability index grade ${LLB_MI_MIN_GRADE} only (src tests only; informational)"
LLB_COGNITIVE_LABEL="cognitive complexity above ${COGNITIVE_MAX} (src tests only)"

# complexipy has no cache-dir flag -- it writes .complexipy_cache in its CWD. Run it from the
# shared $DATA_DIR cache tree (against absolute src/tests) so nothing lands in the project root.
llb_complexity_cache_dir() {
  local dir="${DATA_DIR:-$PROJECT_ROOT/.data}/cache/complexipy"
  mkdir -p "$dir"
  printf '%s' "$dir"
}

llb_complexity_tools_ready() {
  if [ ! -x "$LLB_RADON" ] || [ ! -x "$LLB_COMPLEXIPY" ]; then
    echo "ERROR: radon/complexipy missing in .venv -- run 'make venv EXTRAS=dev' first" >&2
    return 1
  fi
}

llb_cyclomatic_scan() {
  (cd "$PROJECT_ROOT" && "$LLB_RADON" cc src tests -s -n "$LLB_RADON_MIN_GRADE")
}

llb_maintainability_scan() {
  (cd "$PROJECT_ROOT" && "$LLB_RADON" mi src tests -s -n "$LLB_MI_MIN_GRADE" -x "$LLB_MI_MIN_GRADE")
}

llb_maintainability_report() {
  llb_report_if_output "$LLB_MI_LABEL" llb_maintainability_scan
}

llb_cognitive_scan() {
  (
    cache_dir="$(llb_complexity_cache_dir)" && cd "$cache_dir" || exit 1
    "$LLB_COMPLEXIPY" "$PROJECT_ROOT/src" "$PROJECT_ROOT/tests" \
      --max-complexity-allowed "$COGNITIVE_MAX" --failed --ignore-complexity \
      --color no --plain --sort desc
  )
}

# Run BOTH scans (never short-circuit -- a reader wants every peak in one pass) and return
# non-zero when either printed a finding.
llb_complexity_gate() {
  local failed=0
  llb_complexity_tools_ready || return 1
  llb_fail_if_output "$LLB_CYCLOMATIC_LABEL" llb_cyclomatic_scan || failed=1
  llb_fail_if_output "$LLB_COGNITIVE_LABEL" llb_cognitive_scan || failed=1
  return "$failed"
}

# Both entrypoints print the same instruction, so the fix does not depend on which one caught it.
llb_complexity_failure_hint() {
  echo >&2
  echo "[${LLB_REPORT_PREFIX}] FAILED -- split the functions above; do not raise the thresholds." >&2
}
