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

- **Local evidence models:** When running local heavy / evidence / acceptance pipelines on this
  machine (e.g. multi-objective RAG tune, joint search, auto-rag, host validation with real
  backends), pick the strongest local model that fits the host -- never use tiny smoke models
  under 7B parameters (no `llama3.2:3b`, `0.5B`, `1B`, `3B`, etc.) unless the user explicitly
  asks for a tiny model.
- **Model selection order:** 1. Prefer UA-capable instruct models already on the
  host (Ollama / vLLM / llama.cpp). 2. Prefer the largest parameter count that fits the GPU with
  RAG headroom (KV cache + embedder + optional reranker). On a ~16 GiB GPU, prefer ~12B-14B class
  over 24B+ GGUF that barely fits and thrash-swaps. 3. Floor: at least 7B parameters for any
  real-model evidence run recorded in `docs/impl/current/`. 4. CI / `make ci` fixtures and
  injected fakes stay unchanged -- this rule is for live local backend runs only.
- **Model examples (12-16 GiB class):** Prefer: MamayLM-Gemma v2.0 GGUF, `gemma4:31b`,
  `qwen3.6:27b`, when present. Avoid for evidence: `llama3.2:3b` and other sub-7B tags.

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
  behavior instead of duplicating logic.

## Documentation

`docs/impl/plan.md` is FORWARD-ONLY: it contains ONLY work that is not yet implemented.
`docs/impl/current.md` is the compact index for everything DELIVERED; detailed delivered facts live
in topic files under `docs/impl/current/`. Running the cycle below is PART OF "done" for any
feature, Ordered-Implementation-Sequence entry, or ad-hoc task -- not an optional extra. The user
should never have to ask you to make `plan.md` forward-only again.

**The delivered docs are a three-level tree.** `docs/impl/current.md` (areas) ->
`docs/impl/current/<area>.md` (orientation + the tree of pages under it) ->
`docs/impl/current/<area>/<topic>.md` (one subject). A large area owns a directory; a small one
stays a single page. Rules for keeping it navigable:

- **Write to the narrowest page.** Add delivered detail to the topic page that owns the subject.
  Create a new topic page when a subject is genuinely new, and add its row to the area page in the
  same change -- a page no index links to is a page nobody finds.
- **An area page is an index, not a container.** It carries orientation plus a table of its pages;
  it does not accumulate the detail itself.
- **Split before a page becomes a scroll.** A topic page past ~500 lines, or one whose headings
  describe two subjects, splits along the heading seam; a section that grows past ~500 lines with
  no subheadings gets subheadings first.
- **Links are the navigation, so they must land.** `make lint-doc-links` checks every relative
  link (file + `#anchor`) and runs inside `make lint-md`. A heading's anchor does NOT depend on its
  level, so a section moved between files keeps its fragment and only the path changes.

**The plan/current update cycle (run after every implemented feature, before reporting done):**

1. **Record in current docs.** Add or refresh the delivered behavior in the narrowest matching
   topic page (`docs/impl/current/<area>/<topic>.md`, or `docs/impl/current/<area>.md` for a
   single-page area): what was built, where it lives (modules / commands / tests), how to run it,
   and the result if any (numbers, decisions, file locations, dates). Update the area page when a
   new topic page is added, and `docs/impl/current.md` only when a new area or frequent lookup
   path is needed. Results, "DONE" status, and history belong HERE.
2. **Delete from plan.md.** Remove the implemented item's description ENTIRELY. Do NOT leave a
   "DONE" bullet, a result line, a date, or a "we did X" note: if a sentence describes the past it
   is history and must not stay in `plan.md`. Keep the item's stable sequence number ONLY if open
   residual work remains under it; if fully delivered with no residual, delete the whole item.
3. **Promote insights to forward TODOs.** Implementation surfaces insights you only get by doing the
   work -- a weakness in the result, a gap, a sharp edge, a research-grade improvement. Capture each
   as a NEW forward task written as FUTURE work (what to do + why it helps + rough how), NOT as
   commentary on what you did; flag it optional if it is. (E.g. "strengthen the borderline judge
   calibration split -- its CI dips below the gate -- by adding harder + fluent-but-wrong items,
   then re-run the calibrate loop".)
4. **Keep only forward-actionable context.** When a remaining task needs a fact about delivered
   behavior to be implementable, state that fact in ONE line and LINK to `docs/impl/current.md` or
   the specific `docs/impl/current/*.md` topic for the detail -- never restate the delivered
   description in `plan.md`.

**plan.md content rules:**

- Every line must answer "what remains to be done": well-defined specs, dependencies / blockers, and
  explicit AGI-vs-human to-do instructions, ordered by priority and development sequence.
- FORBIDDEN in `plan.md`: "DONE", "delivered", "implemented", an ISO date, "we/I did", check marks,
  result values, or any past-tense status narrative -- all of that lives in `current.md`.
- Self-check before finishing: the task diff NET-REMOVES the implemented scope from `plan.md` and
  ADDS it to the current docs (`current.md` index or `current/*.md` topic), and a grep of `plan.md`
  for `DONE` / `delivered` / `implemented` / an ISO date returns nothing left over from this task.

## Leaving The Host Clean

A "done" report claims the host is back to the state a reader would expect. Verify that before
reporting -- a leftover process keeps burning a GPU or a core, and a leftover file is the next
task's mystery diff.

- **Background shells:** Before reporting done, list every background task this session started and
  confirm each has EXITED. Stop the ones that have not (`TaskStop`), and say so in the report if any
  was killed rather than finished. A task still running is not a finished task.
- **Do not arm a poller for work the harness tracks:** a backgrounded command notifies on exit, so a
  second shell that waits on it is pure waste. Poll only for state the harness cannot see (an
  external queue, a remote job).
- **A watcher must not match itself:** `pgrep -f` / `pgrep -a` match the FULL command line, so a
  waiter that greps for the command it is waiting on finds its own shell and loops forever. Match on
  the process name, the PID, or a pattern the watcher's own command line does not contain.
- **Host processes:** No model server, run, or tool this task started may be left holding the GPU or
  a port unless the user asked for it to stay up. Check with `nvidia-smi` after any heavy local run.
- **Temporary artifacts:** `git status` must show only the files the task intended to change.
  Throwaway roots, scratch scripts, half-written fixtures, and debug output go to the session
  scratchpad directory or are deleted -- never left in the repo, `src/`, or `samples/`.
- **Run artifacts are NOT temporary:** anything a real run wrote under
  `$DATA_DIR/<method_name>/<run_timestamp>/` stays. Clean up what YOU scaffolded, not what the
  pipeline produced.

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

Any installation that compiles C++/CUDA from source (git+, --no-binary, --no-build-isolation)
MUST cap parallelism using formila MAX_JOBS=min(cpu_core_num//2, RAM // 14)

Do not inline the formula — the helpers are the single source of truth. The canonical helper is
`max_jobs()` in `scripts/shared/common.sh` (source it; see `scripts/build_vllm.sh` for usage).

Only wheels deliberately built from a local git checkout (flash-attn forks, vLLM forks, xformers
forks, etc.) may be exported under
`$DATA_DIR/wheels/<package-name>_<abi-key>_git<revision>/`. The key MUST encode the
ABI-relevant dimensions (Python, torch, CUDA, GPU compute capability) and the exact git
revision; source checkouts must be clean before building.

Registry wheels, prebuilt wheels, and all ordinary build/runtime dependencies MUST be installed
directly with `uv` and left in uv's standard shared cache. Never use `pip wheel` or a
dependency-resolving wheelhouse under `$DATA_DIR/wheels`; that directory contains only
intentional local-source build outputs.
