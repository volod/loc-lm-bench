# AGENTS.md project rules

## Development Guardrails

- **Git:** Do not create git commits or revert user changes unless explicitly asked.
- **Python:** Use `uv` and `pyproject.toml` for all dependency management (Python >= 3.10).
- **Typing:** Do not add `from __future__ import annotations`; use normal annotations and
  `TYPE_CHECKING` imports when needed.
- **Paths:** Never hardcode absolute directories (e.g., `/home/...`). Resolve every path from
  the project base directory and honor `.env`/`DATA_DIR` settings.

## Code Organization

- **CLI vs Core:** Use `src/llb/main.py` as the CLI entry point and `src/llb/cli/` for Typer
  command modules. Keep top-level `scripts/` as shell entrypoints only. Put production Python
  implementations inside `src/llb/...`.
- **Modularity & Refactoring:** Keep modules small and focused by organizing them into
  intuitively named subpackages or submodules. Extract long procedural code sequences with a
  small number of input parameters into well-named, self-contained functions. Maximally reuse
  existing code and avoid repeating yourself (DRY). You must proactively evaluate your work
  against these principles after completing any sizable feature implementation.
- **Artifacts:** Runtime data and run artifacts belong under
  `$DATA_DIR/<method_name>/<run_timestamp>/`. Never write to a module-local data inside `src/`.
- **Shell Scripts:** Reuse `scripts/shared/common.sh` for shared shell root/env/bootstrap
  behavior instead of duplicating logic.

## Documentation

`docs/impl/plan.md` is FORWARD-ONLY: it contains ONLY work that is not yet implemented.
`docs/impl/current.md` is the system of record for everything DELIVERED. Running the cycle below is
PART OF "done" for any feature, Ordered-Implementation-Sequence entry, or ad-hoc task -- not an
optional extra. The user should never have to ask you to make `plan.md` forward-only again.

**The plan/current update cycle (run after every implemented feature, before reporting done):**

1. **Record in current.md.** Add or refresh the delivered behavior in `docs/impl/current.md`: what
   was built, where it lives (modules / commands / tests), how to run it, and the result if any
   (numbers, decisions, file locations, dates). Results, "DONE" status, and history belong HERE.
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
   behavior to be implementable, state that fact in ONE line and LINK to `docs/impl/current.md` for
   the detail -- never restate the delivered description in `plan.md`.

**plan.md content rules:**

- Every line must answer "what remains to be done": well-defined specs, dependencies / blockers, and
  explicit AGI-vs-human to-do instructions, ordered by priority and development sequence.
- FORBIDDEN in `plan.md`: "DONE", "delivered", "implemented", an ISO date, "we/I did", check marks,
  result values, or any past-tense status narrative -- all of that lives in `current.md`.
- Self-check before finishing: the task diff NET-REMOVES the implemented scope from `plan.md` and
  ADDS it to `current.md`, and a grep of `plan.md` for `DONE` / `delivered` / `implemented` / an ISO
  date returns nothing left over from this task.

## Formatting & Conventions

- **ASCII Only:** Use ASCII in logs, docs, comments, and generated shell output. No emojis or
  Unicode box-drawing characters (use `[ok]`, `->`, `=`, `-`, `[info]`, `*`).
- **Constants:** Avoid magic numbers. Create constant modules with well-described variables to
  improve readability.
- **Logging:** Use Python's `logging` module instead of `print()`.
- **Optimization Stack:** Prefer Python-native packages (`pytorch`, `numpy`, `scipy`) to keep the
  stack Pythonic. Use `Optuna` and `MLflow` for tuning and tracking when necessary.

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
