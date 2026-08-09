#!/usr/bin/env bash
# The shell half of the lint gates. SOURCE this after scripts/shared/common.sh + llb_load_env.
#
# Two scans over every TRACKED *.sh (git ls-files, so the gate covers exactly what a commit
# carries, wherever it lives):
#   * `bash -n` syntax -- needs nothing installed, so it always runs.
#   * shellcheck at severity `warning` -- the binary ships in the `dev` extra (shellcheck-py), so
#     every host that can run `make ci` has it, GitHub CI included.
#
# A MISSING shellcheck is a failure, not a skip: a linter that reports itself as fine when it never
# ran is worse than no linter. LLB_SHELLCHECK_OPTIONAL=1 downgrades that to a printed skip, for a
# lean venv on a dev box that still wants the rest of the sweep.

LLB_SHELLCHECK="${LLB_SHELLCHECK:-$PROJECT_ROOT/.venv/bin/shellcheck}"
LLB_SHELLCHECK_SEVERITY="${LLB_SHELLCHECK_SEVERITY:-warning}"

LLB_SHELL_SYNTAX_LABEL="shell syntax (bash -n) over tracked/new *.sh"
LLB_SHELLCHECK_LABEL="shell lint (shellcheck -S ${LLB_SHELLCHECK_SEVERITY}) over tracked/new *.sh"

# Tracked plus new-but-not-ignored *.sh, anywhere in the repo: a script is linted before its first
# commit, and nothing under a gitignored tree (.venv, $DATA_DIR) is ever scanned. Index entries
# whose file is gone (a staged delete) are dropped -- there is nothing to parse.
llb_lintable_shell_scripts() {
  local script
  (
    cd "$PROJECT_ROOT" || exit 1
    while IFS= read -r -d '' script; do
      [ -f "$script" ] && printf '%s\0' "$script"
    done < <(git ls-files -z --cached --others --exclude-standard '*.sh')
  )
}

# Run from the project root so bash's own error line names the script the way the repo does.
llb_shell_syntax_scan() {
  local script output
  (
    cd "$PROJECT_ROOT" || exit 1
    while IFS= read -r -d '' script; do
      if ! output="$(bash -n "$script" 2>&1)"; then
        printf '  [failed] %s\n' "$script"
        [ -n "$output" ] && printf '%s\n' "$output" | sed 's/^/    /'
      fi
    done < <(llb_lintable_shell_scripts)
  )
}

# One invocation over the whole set: shellcheck names the file in every finding, and running from
# the project root keeps those paths repo-relative.
llb_shellcheck_scan() {
  (
    cd "$PROJECT_ROOT" || exit 1
    llb_lintable_shell_scripts | xargs -0 -r "$LLB_SHELLCHECK" -S "$LLB_SHELLCHECK_SEVERITY"
  )
}

# Resolve the venv binary, then any shellcheck on PATH (an apt install on an older host).
llb_shellcheck_step() {
  if [ -x "$LLB_SHELLCHECK" ] || LLB_SHELLCHECK="$(command -v shellcheck 2>/dev/null)"; then
    llb_fail_if_output "$LLB_SHELLCHECK_LABEL" llb_shellcheck_scan
    return
  fi
  if [ "${LLB_SHELLCHECK_OPTIONAL:-0}" = "1" ]; then
    llb_print_block "shell lint (shellcheck) skipped -- LLB_SHELLCHECK_OPTIONAL=1"
    LLB_SHELLCHECK_SKIPPED=1
    return 0
  fi
  llb_print_block "shell lint (shellcheck) MISSING -- run: make venv EXTRAS=dev"
  return 1
}

# Run BOTH scans (never short-circuit -- one pass shows every broken script) and return non-zero
# when either found something.
llb_shell_lint_gate() {
  local failed=0
  llb_fail_if_output "$LLB_SHELL_SYNTAX_LABEL" llb_shell_syntax_scan || failed=1
  llb_shellcheck_step || failed=1
  return "$failed"
}

# A pass never claims more than it checked: a skipped shellcheck says so on the ok line.
llb_shell_lint_ok_line() {
  if [ "${LLB_SHELLCHECK_SKIPPED:-0}" = "1" ]; then
    echo "[${LLB_REPORT_PREFIX}] ok -- every tracked/new *.sh parses (shellcheck NOT run)"
  else
    echo "[${LLB_REPORT_PREFIX}] ok -- every tracked/new *.sh parses and is clean at severity ${LLB_SHELLCHECK_SEVERITY}"
  fi
}

# Both entrypoints print the same instruction, so the fix does not depend on which one caught it.
llb_shell_lint_failure_hint() {
  echo >&2
  echo "[${LLB_REPORT_PREFIX}] FAILED -- fix the findings above; do not disable or skip the check." >&2
}
