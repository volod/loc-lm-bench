# AGENTS.md project rules

## Development Guardrails

- **Git:** Do not create git commits or revert user changes unless explicitly asked.
- **Python:** Use `uv` and `pyproject.toml` for all dependency management (Python >= 3.12).
  For direct `uv` commands, source `scripts/shared/common.sh` and run `llb_load_env` first so
  `UV_LINK_MODE` is resolved adaptively: cross-device repo/cache hosts use `copy`, same-device
  hosts keep uv's default linking. Leave `UV_LINK_MODE` unset or `auto` unless forcing a mode is
  the explicit purpose of the command.
- **Typing:** Do not add `from __future__ import annotations`; use normal annotations and
  `TYPE_CHECKING` imports when needed.
- **Paths:** Never hardcode absolute directories (e.g., `/home/...`). Resolve every path from
  the project base directory and honor `.env`/`DATA_DIR` settings.
- **Make aliases:** Quick-start and standard workflow commands must use `make` targets when a
  target exists. Add a Makefile target with a `##` help description before documenting a repeated
  workflow in README, guides, or AGENTS.md; keep raw `llb` or `python -m` commands only for
  low-level CLI reference or one-off debugging.

## Local Run Models Selection

Real-model evidence on this host uses the strongest model that FITS -- never a sub-7B smoke model
(`llama3.2:3b`, `0.5B`, `1B`, `3B`) unless the user asks for one. CI fixtures and injected fakes are
exempt. Before picking one, read the sizing order and the 12-16 GiB roster in [heavy runs and evidence](docs/guides/development/heavy-runs-and-evidence.md).

## Code Organization

- **CLI vs Core:** Use `src/llb/main.py` as the CLI entry point and `src/llb/cli/` for Typer
  command modules. Keep top-level `scripts/` as shell entrypoints only. Put production Python
  implementations inside `src/llb/...`.
- **Modularity & Refactoring:** Keep modules small and focused by organizing them into
  intuitively named subpackages or submodules. Extract long procedural code sequences with a
  small number of input parameters into well-named, self-contained functions. Maximally reuse
  existing code and avoid repeating yourself (DRY). You must proactively evaluate your work
  against these principles after completing any sizable feature implementation.
- **File-size soft limit (~250 lines):** Aim to keep every tracked `.py` and `.sh` file at or
  under ~250 lines. This is a SOFT target, not a hard gate: split a file only along a clear
  functional seam that improves readability (e.g. a CLI module into per-command submodules, a
  runner into phase helpers). Functionality beats the formal count -- when a single cohesive,
  regular structure (one lookup table, one dataclass family, one exhaustive match) reads better
  whole, keep it whole even past 250 lines rather than fragmenting it. `scripts/code_quality.sh`
  reports files over the soft limit so the list stays visible.
- **Artifacts:** Runtime data and run artifacts belong under
  `$DATA_DIR/<method_name>/<run_timestamp>/`. Never write to a module-local data inside `src/`.
- **Shell Scripts:** Reuse `scripts/shared/common.sh` for shared shell root/env/bootstrap
  behavior instead of duplicating logic. Every function defined in a tracked `*.sh` MUST be named
  `llb_*` -- `source` flattens all of them into one namespace, and the prefix is what lets
  `llb.quality.shell_symbols` tell a call from a word, so an unprefixed helper is simply outside
  that check. `make shell-lint-gate` enforces it. A helper defined inside a make recipe is exempt:
  it lives and dies in one `bash -c`. A fragment that is sourced by an entrypoint and calls a
  sibling declares it with `# llb-requires: <sibling>`.

## Documentation

`docs/impl/plan.md` is FORWARD-ONLY -- only work not yet implemented. `docs/impl/current.md` indexes
what is DELIVERED; the detail lives in `docs/impl/current/<area>/<topic>.md`. Running the cycle
below is part of "done" for any task, not an extra.

**The delivered docs are a three-level tree** -- `current.md` (areas) -> `<area>.md` (orientation
plus a table of its pages) -> `<area>/<topic>.md` (one subject). A large area owns a directory, a
small one stays one page.

- **Write to the narrowest page,** and add a new topic page's row to its area page in the same
  change -- a page no index links to is a page nobody finds.
- **An area page is an index, not a container.**
- **Split a page past ~500 lines, or one whose headings describe two subjects,** along the heading
  seam; a section that long with no subheadings gets subheadings first.
- **Links must land.** `make lint-doc-links` (inside `make lint-md`) checks every relative link and
  anchor. An anchor does not depend on heading level, so a moved section keeps its fragment.

**The cycle, after every implemented feature and before reporting done:**

1. **Record it** in the narrowest topic page: what was built, where it lives (modules, commands,
   tests), how to run it, and the result. Results and history belong HERE, never in `plan.md`.
2. **Delete it from `plan.md`** -- the whole item, unless open residual work remains under it.
3. **Route what the work surfaced** to exactly one of four places: a **chore** -> do it now or drop
   it; an **audit of our own output** -> drop it; **more work under an existing capability** -> a
   task in that group, written as future work and flagged `(optional)` if it is; a **capability the
   spec does not describe** -> the lifecycle below, never a task. Surfacing nothing is a normal,
   good outcome.
4. **Keep `plan.md` forward-actionable.** A task needing a delivered fact states it in ONE line and
   links the current-docs topic; it never restates it.

### Citing a measured result

Delivered docs must stay checkable on a machine that never held the run, so a page cites neither a
`$DATA_DIR/<method>/<run-id>/` path (host-local, deleted, absent on the other GPU hosts) nor a bare
run label (a lookup key that identifies nothing). It states what ran on what, the date and host,
every load-bearing number AND its reading, and what would overturn it. **Before writing any measured
result into the docs, read [heavy runs and evidence](docs/guides/development/heavy-runs-and-evidence.md).**

## Product Feature Lifecycle

The spec is a LIVING register of what the product does, not a scope fence. Finding a gap while
implementing is a good outcome; acting on it silently is not. **Capability that reaches `src/` or
`plan.md` with no spec section describing it is never acceptable** -- nobody can evaluate, document,
or later remove it.

The six steps are canonical in [Extending this
specification](docs/design/spec.md#extending-this-specification): state the domain problem, amend
the spec INCLUDING the capability's boundary, declare the evaluation and what a negative result
looks like, register the row as `planned`, write the tasks (each with a `Serves` line), then close
the loop when it lands. Two rules that list does not carry:

- **Timing.** Do steps 2-4 before the code when the capability is known up front, and in the same
  change when the work taught you it was needed -- never in a later ticket.
- **Wrong shape.** When an existing capability is the wrong shape, amend its spec section rather
  than working around it in the implementation: a spec that no longer describes the code is worse
  than none, because people trust it.
## Specification And Plan Integrity

`make lint-spec-plan` (inside `make ci-checks`) makes spec/plan drift a build failure. It enforces
that every task carries a `Serves` line naming a real capability id; that every `shipped` capability
links its implementation docs and every `planned` one has an open task; that every capability
declares a non-empty evaluation; and that capability groups appear in `plan.md` in registry row
order, in both sections. When it fails, fix the DOCUMENTS -- it is reporting a real disagreement
about what the product is, and the fix is never to loosen the check.

**plan.md structure:** `## Agent Implementation Tasks` / `## Human-Assisted Tasks`, then a `###`
group per capability (title, two hyphens, backticked id), then one `####` per task id. Count with
`grep -c '^#### ' docs/impl/plan.md`.

**Every line in `plan.md` answers "what remains to be done".** FORBIDDEN there: "DONE",
"delivered", "implemented", an ISO date, "we/I did", check marks, result values, or any past-tense
status -- all of it lives in the current docs.

**Ordering.** `(optional)` sorts a task behind the non-optional tasks of its OWN group and never
reorders the line: an optional task of an earlier capability still precedes a required task of a
later one. A capability whose remaining work is entirely optional moves DOWN the line -- a shipped
capability with only refinements left is not what blocks the product.

**Self-check before reporting done.** The diff NET-REMOVES the implemented scope from `plan.md` and
ADDS it to the current docs, and grepping `plan.md` for the forbidden words returns nothing left
over. Report the task count before and after and say which capabilities moved: growth is allowed --
that is what extensibility means -- but it is REPORTED, never silent, and growth concentrated in the
capability you happened to be working in is the signal to stop and ask whether the line still
reflects what matters.

## Leaving The Host Clean

A "done" report claims the host is back to the state a reader would expect, so verify before
reporting: every background task this session started has EXITED (stop the rest, and say so if any
was killed rather than finished), `git status` shows only the files the task intended to change, and
nothing this task started still holds the GPU or a port. **Run artifacts are NOT temporary** --
anything a real run wrote under `$DATA_DIR/<method_name>/<run_timestamp>/` stays; clean up what YOU
scaffolded, not what the pipeline produced. Winding a run down safely -- pollers, self-matching
watchers, the GPU check -- is in
[heavy runs and evidence](docs/guides/development/heavy-runs-and-evidence.md).

## Formatting & Conventions

- **ASCII Only:** Use ASCII in logs, docs, comments, and generated shell output. No emojis or
  Unicode box-drawing characters (use `[ok]`, `->`, `=`, `-`, `[info]`, `*`).
- **Constants:** Avoid magic numbers. Create constant modules with well-described variables to
  improve readability.
- **Logging:** Use Python's `logging` module instead of `print()`.
- **Optimization Stack:** Prefer Python-native packages (`pytorch`, `numpy`, `scipy`) to keep the
  stack Pythonic. Use `Optuna` and `MLflow` for tuning and tracking when necessary.
- **Markdown:** Lint docs with `make lint-md` (pymarkdown; config in `pyproject.toml`
  `[tool.pymarkdown]`). It runs in the full `make test` precommit flow, NOT in `make ci`. Fix any
  findings BY HAND. **Do NOT run `pymarkdown fix`** -- it is unreliable on this version (0.9.38): it
  crashes mid-run and rewrites a line-leading `+`/`-` "plus" connector (e.g. `model + config`
  wrapped to a new line) into a Markdown list bullet, corrupting prose. When the wide-table rule
  fires spuriously, the `markdown-tables` extension + `MD013.tables = false` already handle it;
  reach for a per-rule `enabled = false` in `[tool.pymarkdown]` over editing content to match a
  cosmetic rule.
- **Examples:** Use abstract placeholders in docs and examples (for example
  `<answered-jsonl>` or `<corpus-dir>`) instead of current-run, user-specific, or local artifact
  filenames unless the path is a committed fixture.

## Heavy compilation (ninja / cmake / CUDA)

Any install that compiles C++/CUDA from source MUST cap parallelism with `llb_max_jobs()` from
`scripts/shared/common.sh` -- never inline the formula. Only wheels built from a local git checkout
may be exported under `$DATA_DIR/wheels/`; registry and prebuilt wheels install with `uv`. Details
and the ABI-key layout: [dev setup](docs/guides/development/dev-setup.md).
