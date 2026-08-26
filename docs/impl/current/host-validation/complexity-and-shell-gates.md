# Complexity And Shell Gates

Two gates fail the build on their own commit rather than surfacing in a later sweep: function
complexity over `src` and `tests`, and the ShellCheck-plus-symbols scan over every tracked or
new `*.sh`. The rest of `scripts/code_quality.sh` is deliberately informational.

## Function complexity

**Both function-complexity thresholds are enforced, not reported.** `scripts/complexity_gate.sh`
runs the Radon D-or-worse scan and the Complexipy scan at `COGNITIVE_MAX=15` over `src` and
`tests`, and
exits non-zero as soon as either prints a row -- so a function that crosses a threshold turns the
build red on its own commit instead of surfacing in a later sweep. `ci-checks` runs it beside
`llb.quality.acceptance_gates --check` (so it runs in `make ci`, `make ci-github`, and GitHub CI);
`make complexity-gate` runs it alone after a refactor. Neither tool signals through its exit status
in this configuration (radon always exits 0, complexipy is asked for a plain listing with
`--ignore-complexity`), so the gate treats printed output as the failure. Both scans always run --
one pass shows every peak. The thresholds are the shipped numbers: a finding is split, not
accommodated by raising the maximum.

**Both scanners are pinned exactly** (`radon==6.0.1`, `complexipy==6.0.0`), for the reason the
shell-lint gate pins its ShellCheck wheel: the tool version is half of the verdict. Complexipy
rescored this unchanged tree across its 5.6.1 -> 6.0.0 boundary in both directions --
`persist_run` 18 -> 14 and `research_conflict_nulls_cmd` 16 -> 15 (dropping to the maximum),
`fetch_manifest` 11 -> 13 (climbing) -- so a floating requirement let the same commit fail the gate
on the host that resolved 5.6.1 and pass on the host that resolved 6.0.0. The pin is what makes
`COGNITIVE_MAX=15` mean one thing. Upgrading either scanner is a deliberate edit: change the pin in
`pyproject.toml`, refresh `uv.lock`, then run `make complexity-gate` plus `make ci` and split
whatever the new algorithm surfaces.

The scans, thresholds, and labels live once in `scripts/shared/complexity.sh`, sourced by both the
gate and `scripts/code_quality.sh`, so the sweep fails on exactly what CI fails on and prints it
identically (block reporting is `llb_print_block` / `llb_report_if_output` / `llb_fail_if_output`
in `scripts/shared/common.sh`). The same shared module owns the informational maintainability-index
scan so its scope cannot drift from the two gates.

## Shell scripts

**Shell scripts are gated the same way.** `scripts/shell_lint_gate.sh` (`make shell-lint-gate`,
also inside `ci-checks`) runs three scans over every tracked-or-new `*.sh` in the repo
(`git ls-files --cached --others --exclude-standard`, so a script is linted before its first commit
and nothing under a gitignored tree is scanned; a staged delete is dropped):

- `bash -n` syntax, which needs nothing installed and therefore always runs.
- `shellcheck -x -P SCRIPTDIR -S warning`, over the whole set in one invocation so findings stay
  repo-relative.
- the same run restricted to `SC1090,SC1091` at `-S style`: every `# shellcheck source=` directive
  must resolve.
- `llb.quality.shell_symbols`: every function a tracked `*.sh` defines carries the `llb_` prefix,
  and every `llb_*` name a caller uses has a definition in that caller's declared scope.

**The lint is cross-file, and the third scan is what keeps it that way.** `-x` follows a sourced
file, so a caller is checked against what `scripts/shared/{common,complexity,shell_lint}.sh`
actually define instead of in isolation -- the axis these scripts grew along. `-P SCRIPTDIR` is
what makes `-x` work here: a `source=` path resolves relative to the CWD by default, so running
from the project root, every `source=shared/common.sh` directive in `scripts/` resolved to nothing
and `-x` followed nothing (9 such directives). An unresolved directive is reported at INFO,
*below* the severity floor, so the lint pass alone cannot tell "followed and clean" from "never
followed" -- hence the dedicated scan. The proof that following now happens is in the tree: the two
gate entrypoints set `LLB_REPORT_PREFIX` for `common.sh` to read, which un-followed shellcheck
called an unused variable (SC2034); both `# shellcheck disable` workarounds are gone and the gate
stays green.

**A path computed at run time is annotated `# shellcheck source=/dev/null` with its reason, and
that choice is the difference between one verdict and one verdict per host.** `common.sh` sources
`$PROJECT_ROOT/.env`, an untracked operator file, and `-x` FOLLOWS a sourced path where it exists
while reporting SC1091 where it does not -- so the source-directive scan was silent on every dev
box (which has a `.env`) and failed the gate in a fresh GitHub checkout (which does not). That is
the same split verdict the pinned binary exists to prevent, arriving through the tree instead of
through the linter, so the fix says once that there is nothing static to follow rather than letting
the linter decide per host; `disable=SC1091` would have hidden the message while leaving the
following itself host-dependent. `test_the_source_directive_scan_is_clean_in_a_checkout_that_has_no_env_file`
in `tests/llb/quality/test_shell_lint_resolution.py` is the regression: it copies the tracked `*.sh`
into a `git init` tmp tree with no `.env`, runs the real `llb_shellcheck_sources_scan` against it,
and requires empty output -- the side a dev box cannot otherwise see (1.5s, non-slow, since a
CI-only failure has to be caught in `make ci`).

**Functions are checked by `llb.quality.shell_symbols`, because the linter does not model them.**
`-x` shares VARIABLE knowledge across a followed source, not function definitions, so a call to a
helper that no longer exists passes every scan above and fails as `command not found` on an
operator's host. The module collects `llb_*` definitions over a caller's declared scope and names
any `llb_*` it uses with no definition there, for tracked-or-new `*.sh` **and** `Makefile`/`*.mk`
(the recipes that `source "$(PROJECT_ROOT)/scripts/shared/common.sh"` are call sites too). Scope is
what a caller DECLARES, never what happens to be loaded at run time:

- a `# shellcheck source=` directive (resolved script-dir first, then repo root, as shellcheck
  resolves it -- and the scan above already proves these land),
- a make recipe's literal `$(PROJECT_ROOT)/...` source,
- a `# llb-requires: <sibling>` line, for a shared module whose functions assume the entrypoint
  sourced a sibling. `scripts/shared/{complexity,shell_lint}.sh` carry it for `common.sh`: they
  call `llb_fail_if_output` and never source it themselves, which is a contract their headers
  stated in prose and now state in a line the check reads.

`llb-requires` scope is also minimal at the file boundary. `unused_requirements` reports the
declaration line when the declaring caller names none of the `llb_*` functions its sibling defines,
and the command exits non-zero so `make shell-lint-gate` refuses the stale declaration. One direct
reference justifies the whole sibling; this remains a file-level contract rather than a per-symbol
import list. Real `source` directives are not subject to this rule because a sourced file can supply
variables or intentional side effects without a function call.

The gate decision came from the shipped tree rather than an exception policy: all 19 current
`llb-requires` declarations support direct calls, including the four declarations in `track_c.sh`.
No declaration exists only to document load order. A downstream fragment owns its own sibling
requirements, and transitive scope already follows them, so adding a load-order-only declaration to
an otherwise unrelated caller would widen the symbol contract without resolving one of that
caller's names. The shipped-tree assertion and synthetic stale/used declaration cases live in
`tests/llb/quality/test_shell_symbols.py`.

What counts as a call is a name in command or argument position -- including one handed to a runner
(`llb_fail_if_output "$LABEL" llb_cyclomatic_scan`), which breaks identically when the definition
goes. Prose in a comment, an expansion operand (`${name#llb_prefix}`), a `$llb_var`, and a path
segment are not calls. A call built by `eval` or through a variable is out of reach and stays out of
scope. Coverage is `tests/llb/quality/test_shell_symbols.py`, whose first assertion is the shipped
tree itself.

**The prefix is a rule, not a convention, because the check has no other way to find a call.**
Keying on `llb_*` is what separates a call from an English word, so the scan's coverage used to be
whatever share of the tree happened to follow the convention: 35 of the 92 functions defined in
tracked `*.sh`. The uncovered 57 were the most cross-file-coupled code here -- `scripts/quickstart.sh`
plus the eight `scripts/quickstart/*.sh` fragments, which are sourced into one namespace and share a
vocabulary (`resolve_path`, `run_target`, `prompt_yes_no`, `make_with_data_dir`, ...) that no scan
resolved. Two rules could have closed it, and the tree now carries the first:

- **Adopt the prefix** (taken). All 57 are renamed `llb_*`, and `unprefixed_definitions` refuses a
  function defined in a tracked `*.sh` without it, so the coverage cannot decay back. Three names
  needed more than a prefix, because `source` makes the namespace flat: `main` in `quickstart.sh` is
  `llb_quickstart_main`, and the two different `usage` bodies are `llb_quickstart_usage` and
  `llb_apt_usage`.
- **Drop the prefix and treat a name as a call when it is DEFINED somewhere in the scanned set**
  (rejected). It cannot deliver the outcome the check exists for. A rename removes the old name from
  the defined set, so the stale call site stops looking like a call at the exact commit that broke
  it; the rule only catches a helper that moved out of scope while still being defined elsewhere,
  which is the narrower half of the failure. It also has a false-call surface the prefix does not:
  the uncovered names include bare words (`main`, `usage`, `result`, `heading`), and a scan with no
  command-position parsing reads `KNOWLEDGE_CUTOFF_REVISION ?= main` in `make/config.mk` as a call --
  the one false positive a shipped-tree trial of the rule produced.

The scope declarations were the shared cost either rule had to pay, not a cost of the rename: the
fragments are sourced BY `quickstart.sh` and never source each other, so all 217 of their calls were
out of scope under either rule until each fragment named its siblings with `# llb-requires:`
(`track_c.sh` declares `helpers.sh`, `model_select.sh`, `pdf_draft.sh`, `track_b.sh`; the other
seven are shorter). A function defined inside a make recipe is exempt from the prefix -- `wants_backend`
and `record_failure` in `make/eval/categories-platform.mk` live and die in one `bash -c` and never
enter the sourced namespace -- which is why the prefix scan reads `*.sh` only.

**Residual: a name that never carried the prefix is still invisible.** The check finds a call whose
helper was renamed, moved, or deleted; it cannot find a call to an EXTERNAL command that does not
exist, because a name defined nowhere is indistinguishable from a binary the host is expected to
have without real command-position parsing. That case stays with the `command -v` guards the scripts
already carry.

A MISSING shellcheck fails the gate rather than skipping it -- a linter that reports itself as fine
when it never ran is worse than no linter, and the old sweep did exactly that on a host without the
apt package. `LLB_SHELLCHECK_OPTIONAL=1` downgrades that to a printed skip for a lean venv, and the
pass line then says `shellcheck NOT run` rather than claiming the tree is clean. That escape hatch
is rarely needed: shellcheck now ships in the `dev` extra as the exactly pinned
`shellcheck-py==0.11.0.1` wheel (the real 0.11.0 binary), so `.venv/bin/shellcheck` exists wherever
`make ci` runs, GitHub CI included -- no apt step in the workflow. Scans, severity, and the
missing-binary policy live once in
`scripts/shared/shell_lint.sh`, sourced by both the gate and the sweep; every scan runs before any
of them fails.

**Exactly one binary decides the verdict: `$LLB_SHELLCHECK`, default `.venv/bin/shellcheck`.** The
`command -v shellcheck` fallback is gone. It was what made a green run host-dependent: a distro
package is releases behind the pin (this dev box ships 0.9.0 against the pinned 0.11.0) and
shellcheck ADDS checks between releases, so the same commit could pass on the fallback host and
fail in CI -- the split verdict a pinned linter exists to prevent, moved one level down. The gap is
real, not theoretical: `out="$(echo hi >/tmp/f)"` draws SC2327 (warning) + SC2328 (error) from
0.11.0 and nothing at all from 0.9.0, so a script written that way lints clean on the fallback host.
On the shipped tree the two versions happen to agree today (both scans clean at `-S warning` and at
`-S style -i SC1090,SC1091`), which is why the fix was cheap to take before a divergence landed.
Requiring the wheel costs no workflow -- a venv without the `dev` extra cannot run `make ci` anyway
-- and the missing-binary block now names the path it looked at, since "MISSING" is otherwise
puzzling on a host that has `shellcheck` on `PATH`. `LLB_SHELLCHECK` still overrides the path for a
venv living elsewhere. Consequently the `dev` apt profile is now **empty**
(`scripts/apt/dev.packages` keeps the file and the comment; the profile stays for a future dev-only
OS package) and [dev setup](../../../guides/development/dev-setup.md#apt-dependencies-debianubuntu)
no longer lists `shellcheck` as a fallback. The exact project requirement is intentional and is the
outer of two guarantees. `make venv` and GitHub CI both install through `uv sync` (the local
target with `--inexact`, the workflow with `--locked`), so `uv.lock` already pins the wheel for
every venv either one builds; the `==` requirement in `pyproject.toml` extends the same pin to an
install that bypasses the lock -- plain pip, or a bare `uv pip install` run instead of
`make install-extras` ([overview](../overview.md#an-extras-install-respects-uvlock)). A host
resolving the extra today and one resolving it months later still install the same wheel. An
upgrade therefore costs one deliberate pin edit in `pyproject.toml`, `uv lock`, and verification
with `make shell-lint-gate` plus `make ci`; the lock and the fresh-install requirement move
together.
The binary resolution behavior remains covered by
`tests/llb/quality/test_shell_lint_resolution.py`.

## The informational sweep

Everything else in the sweep stays informational -- in particular the `.py`/`.sh` line-count report,
which backs a target AGENTS.md keeps SOFT on purpose and which has legitimate offenders. The
maintainability-index section is deliberately a report too. It now scans only `src` and `tests`, at
grade C by default (`LLB_MI_MIN_GRADE=C`), rather than sweeping the repository root. The decision
not to add it to `llb_complexity_gate` is evidence-based: the C-only scan is empty, while the next
band contains one file, `tests/llb/bench/context_policy/test_agentic_context.py`, at B (10.66). That
file is large (728 lines, 563 source lines) but regular: Radon finds 48 blocks with average
cyclomatic complexity A (4.15), and its two highest blocks are only B (10). It is already the kind
of volume case covered by the soft line target, so making the nearby MI boundary hard would turn
that soft policy into an indirect gate rather than add an independent complexity signal. Focused
coverage in `tests/llb/quality/test_maintainability_report.py` pins both the `src tests` arguments
and the informational exit behavior when Radon prints a C row.

## The complexity cleanups

The D-grade cyclomatic-complexity cleanup keeps orchestration separate from validation, state
accumulation, and presentation. Ontology dedup now uses an embedded-candidate value object and
named matching/report helpers; the multi-hop expansion audit uses a check accumulator that builds
the final report. Retrieval validation passes an immutable request into
`cli/rag/retrieval_validation.py`, autonomous verification scoring lives in
`auto_rag/verification_auto.py`, and query-prep dependency checks are table-driven. The query
robustness integration test uses a module-level morphology-loader callable and named assertion /
artifact phases. The repository-wide Radon D-or-worse scan is empty; focused coverage lives in the
ontology, auto-RAG verification, query-prep, and query-robustness test suites.

The cognitive-complexity cleanup extends that separation across backend planning, review
workflows, conflict resolution and filtering, query robustness, incremental refresh, ontology
expansion, retrieval fusion, and reporting. Complex branches now live in named policy,
validation, selection, and rendering helpers. Stateful assembly uses
`rag/refresh/merge_assembly.py`; tree leaf filtering, robustness recovery, and hybrid-store
retrieval use focused owner modules. Launcher and morphology closures are module-level callable
adapters. Focused verification covers the affected backend, review, conflict, evaluation, graph,
ontology, retrieval, and scoring paths.

The agentic/bench lanes grew their own peaks after that pass and were cleaned the same way, with
three patterns doing most of the work:

- **A named check per contract area.** Every `validate_*_design` is now a sequence of `_check_*`
  calls in the order a reader of the design file meets them (identity, ledger, roster, wording,
  sampling, gates). `validate_channel_authority_design` went from F(54) to a body of nine calls.
  Typed field reads moved to `bench/agentic/design_fields.py`, so a rule reads as the contract
  (`as_int(matrix, "n_tasks") < 6`) instead of as a cast.
- **A record for what a step accumulates.** `run_episode` was eleven mutable counters threaded
  through one 130-line loop; it is now `_EpisodeTally` (counting plus the one place an episode
  ENDS), `_ControllerSeam` (how the prompt is serialized, guarded, and sent), and named steps for
  the repair round and the tool call. The loop body is the cycle and nothing else: F(47)/93 became
  C(11)/11. `_CaseMeans`, `_ChannelGates`, `_ComparisonSettings`, and `_PlacementContext` play the
  same role in run metrics, the channel-authority reading, retrieval comparison, and the seeded
  placement run.
- **No closures over a caller's locals.** The controller-authority seed run's nested `harness` /
  `record` / `unused_complete` closures became module-level functions taking an explicit
  `_PlacementContext`, leaving a one-line protocol adapter.

Two oversized modules split at their functional seams in the same pass --
`rag/encoders/throughput.py` into measurement / `_summary` / `_report` / `_profile`, and
`bench/context_policy/sweep.py` into `_model` (axes, grids, cells) / `_verdict` (pairing and the
pin-or-expose cut) / the runner -- with call sites repointed at the real submodule rather than a
re-export shim.

The residual band was then finished the same way, and **both complexity scans are now silent**: the
Radon D-or-worse scan is empty (19 functions before, up to F(54)) and the Complexipy scan at the
shipped maximum of 15 is empty (51 functions before, up to 93). Two shapes carried the last 21:

- **A run entry point is a plan plus named steps.** `run_constant_sweep`, `run_agentic_loop_policy`,
  `run_bakeoff`, and `run_draft` now resolve their contract first (`_scored_cells` caching identical
  cells, `_validate_study_design` / `_study_analysis` for whichever prospective study is running,
  `_ScoredCandidates` accumulating rows/vectors/stores together, `_ResolvedDraft` carrying the two
  fields a resume fills in) and then read as the sequence of steps they are.
- **A CLI command drives cells from a record, not from a closure.** The two repeat-feedback commands
  and the crossover restatement built their per-cell run as a nested function closing over a dozen
  locals; each is now a `_Plan` / `_ResolvedDraft` record plus a module-level step bound with
  `functools.partial`, so a cell cannot read a different temperature or policy than its neighbour.

The rest are the same named-check and reading-record patterns applied to `_judge`, `decide_verdict`,
`build_multi_reviewer_worksheets`, `pair_against_shipped`, `aggregate_safe_verdict`,
`collapse_reading`, `_refuse_cycles`, and the remaining `validate_*` / `analyze_*` readings. The
longest tracked source file is 500 lines.
