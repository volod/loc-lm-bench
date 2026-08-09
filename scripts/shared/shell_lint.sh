#!/usr/bin/env bash
# The shell half of the lint gates. SOURCE this after scripts/shared/common.sh + llb_load_env.
# llb-requires: common.sh
#
# Three scans over every TRACKED *.sh (git ls-files, so the gate covers exactly what a commit
# carries, wherever it lives):
#   * `bash -n` syntax -- needs nothing installed, so it always runs.
#   * shellcheck at severity `warning` -- the PINNED binary from the `dev` extra (shellcheck-py)
#     and only that one, so every host that can run `make ci` reaches the same verdict, GitHub CI
#     included.
#   * every `# shellcheck source=` directive resolves, so the pass above is a CROSS-FILE result.
#   * every `llb_*` call has a definition in its caller's declared scope (llb.quality.shell_symbols),
#     which is the one cross-file mistake shellcheck does not model.
#
# A MISSING shellcheck is a failure, not a skip: a linter that reports itself as fine when it never
# ran is worse than no linter. LLB_SHELLCHECK_OPTIONAL=1 downgrades that to a printed skip, for a
# lean venv on a dev box that still wants the rest of the sweep.

# The pinned wheel binary, and no PATH fallback -- see llb_shellcheck_step. LLB_SHELLCHECK still
# overrides the path, for a venv that lives somewhere other than $PROJECT_ROOT/.venv.
LLB_SHELLCHECK="${LLB_SHELLCHECK:-$PROJECT_ROOT/.venv/bin/shellcheck}"
LLB_SHELLCHECK_SEVERITY="${LLB_SHELLCHECK_SEVERITY:-warning}"

LLB_SHELL_SYNTAX_LABEL="shell syntax (bash -n) over tracked/new *.sh"
LLB_SHELLCHECK_LABEL="shell lint (shellcheck -x -S ${LLB_SHELLCHECK_SEVERITY}) over tracked/new *.sh"
LLB_SHELLCHECK_SOURCES_LABEL="shellcheck source directives that do not resolve (SC1090/SC1091)"
LLB_SHELL_SYMBOLS_LABEL="llb_* calls with no definition in the caller's declared scope"

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
#
# `-x` follows a sourced file, so a caller is checked against what the shared module actually
# defines instead of in isolation. `-P SCRIPTDIR` is what makes that work here: a `# shellcheck
# source=` path resolves relative to CWD by default, so from the project root every
# `source=shared/common.sh` directive in scripts/ resolved to nothing and `-x` followed nothing.
llb_shellcheck_run() {
  (
    cd "$PROJECT_ROOT" || exit 1
    llb_lintable_shell_scripts | xargs -0 -r "$LLB_SHELLCHECK" -x -P SCRIPTDIR "$@"
  )
}

llb_shellcheck_scan() {
  llb_shellcheck_run -S "$LLB_SHELLCHECK_SEVERITY"
}

# An unresolved source is reported at INFO, below the severity floor above, so the lint pass alone
# cannot tell "followed and clean" from "never followed". Check the directives on their own: this
# is the scan that keeps `-x` honest. A path that is genuinely computed at run time (an operator's
# .env, a plugin dir) is annotated in the script with a reasoned `# shellcheck disable=SC1091`
# rather than by dropping this scan.
llb_shellcheck_sources_scan() {
  llb_shellcheck_run -S style -i SC1090,SC1091
}

# The linter resolves VARIABLES across a followed source, not FUNCTIONS, so a call to a helper that
# no longer exists passes every scan above; llb.quality.shell_symbols is that check. It carries its
# own summary line, which is worth reading from a terminal but not from a clean gate, so only a
# failure is passed on.
#
# (A comment must not OPEN with the linter's own name: that reads as a directive, and an unparsable
# one fails this very gate.)
llb_shell_symbols_scan() {
  local output status
  output="$(cd "$PROJECT_ROOT" && "$(llb_python)" -m llb.quality.shell_symbols 2>&1)"
  status=$?
  [ "$status" -eq 0 ] || printf '%s\n' "$output"
  return "$status"
}

# ONE binary decides this gate: the pinned wheel from the `dev` extra. There is deliberately no
# `command -v shellcheck` fallback -- a distro package is releases behind the pin (this dev box
# ships 0.9.0 against the pinned 0.11.0) and shellcheck ADDS checks between releases, so the
# fallback made a green run mean different things on different hosts: the same commit passes on the
# lean host and fails in CI, which is the split verdict a pinned linter exists to prevent. A venv
# without the `dev` extra cannot run `make ci` anyway, so requiring the wheel costs no workflow.
llb_shellcheck_step() {
  local failed=0
  if [ -x "$LLB_SHELLCHECK" ]; then
    llb_fail_if_output "$LLB_SHELLCHECK_LABEL" llb_shellcheck_scan || failed=1
    llb_fail_if_output "$LLB_SHELLCHECK_SOURCES_LABEL" llb_shellcheck_sources_scan || failed=1
    return "$failed"
  fi
  if [ "${LLB_SHELLCHECK_OPTIONAL:-0}" = "1" ]; then
    llb_print_block "shell lint (shellcheck) skipped -- LLB_SHELLCHECK_OPTIONAL=1"
    LLB_SHELLCHECK_SKIPPED=1
    return 0
  fi
  # Name the path: on a host that HAS a distro shellcheck, "missing" is otherwise puzzling.
  llb_print_block "shell lint (shellcheck) MISSING at ${LLB_SHELLCHECK} -- run: make venv EXTRAS=dev"
  return 1
}

# Run BOTH scans (never short-circuit -- one pass shows every broken script) and return non-zero
# when either found something.
llb_shell_lint_gate() {
  local failed=0
  llb_fail_if_output "$LLB_SHELL_SYNTAX_LABEL" llb_shell_syntax_scan || failed=1
  llb_shellcheck_step || failed=1
  llb_fail_if_output "$LLB_SHELL_SYMBOLS_LABEL" llb_shell_symbols_scan || failed=1
  return "$failed"
}

# A pass never claims more than it checked: a skipped shellcheck says so on the ok line.
llb_shell_lint_ok_line() {
  if [ "${LLB_SHELLCHECK_SKIPPED:-0}" = "1" ]; then
    echo "[${LLB_REPORT_PREFIX}] ok -- every tracked/new *.sh parses (shellcheck NOT run)"
  else
    echo "[${LLB_REPORT_PREFIX}] ok -- every tracked/new *.sh parses, resolves its sources and llb_* calls, and is clean at severity ${LLB_SHELLCHECK_SEVERITY}"
  fi
}

# Both entrypoints print the same instruction, so the fix does not depend on which one caught it.
llb_shell_lint_failure_hint() {
  echo >&2
  echo "[${LLB_REPORT_PREFIX}] FAILED -- fix the findings above; do not disable or skip the check." >&2
}
