# Overview

loc-lm-bench is a local-first benchmark for Ukrainian LLM work on private or
domain-specific corpora. The implementation centers on verified corpus data, local model serving,
immutable run artifacts, and tier-separated leaderboards.

## Implementation Principles

- **Verified data gates.** `run-eval` scores only `verified: true` gold items. Category composite
  rows require verified run bundles. Drafts remain useful for review but cannot silently become
  headline data.
- **Source-span truth.** Gold labels point to document ids plus exact character offsets. Retrieval
  metrics compare returned chunks with those spans, so chunking and vector-store changes do not
  invalidate labels.
- **OpenAI-compatible backend seam.** Ollama, vLLM, and llama.cpp are launcher details behind
  `BackendLauncher` plus `openai_client.chat_once`. Evaluation code should not grow
  provider-specific branches.
- **Tier separation.** Public screens, private RAG runs, and category suites have different metric
  semantics. `rank_board` rejects mixed tiers instead of pretending they are comparable.
- **Canonical artifacts first.** Run bundles write `manifest.json` and per-case scores before
  optional MLflow mirroring. MLflow is an analysis mirror, not the source of record.

## Code Organization

Tracked Python and shell files target the ~250-line soft limit in `AGENTS.md`. Split only at a
clear functional boundary; a cohesive schema or lookup family may remain whole. Run
`scripts/code_quality.sh` to see file size and complexity findings. The line target stays a
report; the two complexity thresholds and the shell-lint scans are CI gates
(`make complexity-gate` / `make shell-lint-gate`, both inside `make ci` -- see
[host validation](host-validation/quality-gate.md#code-quality-checks)).

Callers import symbols from their concrete owner module. Package `__init__.py` files contain only
the package docstring, except CLI area packages whose imports register Typer commands. Runnable
packages may provide `__main__.py`; compatibility re-exports are not part of the package design.

Production procedures live in focused owner modules for command construction, resolver
feasibility/reporting, executor durability and retrieval, benchmark scoring/persistence, curation
input/dispatch, fine-tuning manifests/runtime, RAG store construction/validation, and report
formatting. Tests are grouped by behavior with shared factories in intent-named helper modules.

Cross-package `TypedDict` contracts live in `src/llb/core/contracts/`, split into concrete owner
modules: `common`, `rag`, `benchmarks`, `results`, `runs`, `models`, `judging`, `hardware`, and
`screening`. Callers import from those modules directly. The package `__init__.py` contains only
its docstring and provides no compatibility re-exports.

Validation result (2026-07-15): `make ci` passes Ruff formatting/lint, mypy over 524 source files,
and the lightweight suite with 1,414 passed and 38 slow tests deselected; `make lint-md` also
passes. Every contract owner module is at most 132 lines, below the 250-line soft target.

Mutation-heavy construction uses objects instead of branch-heavy procedures:
`DraftResumeBuilder` restores a draft request,
`EndpointConfigBuilder` validates and creates endpoint data, `AgreementReportBuilder` assembles
review statistics, and `ConsensusBuilder` resolves reviewer rows. This keeps validation and state
transitions with the data they govern and avoids parallel procedural entry points.

Current focused package boundaries:

| Concern | Modules |
| --- | --- |
| Make workflows | `make/eval/` and `make/data-prep/` grouped by functional target family |
| Shared typed contracts | `core/contracts/` with domain-specific owner modules and no package-level facade |
| CLI registration | `src/llb/cli/<area>/` with command-specific submodules |
| Draft request construction | `cli/prep/draft_request.py`, `draft_resume.py`, `draft_endpoints.py`, and `draft_execution.py` |
| Host feasibility | `backends/planner/` for architecture, weights, KV sizing, plans, and formatting |
| Evaluation execution | `executor/runner.py` plus `runner_backend.py`, `runner_judge.py`, `runner_metrics.py`, `runner_retrieval.py`, `runner_setup.py`, and `runner_target.py` |
| Board analysis | `board/miss_analysis/`, `board/recommend/`, and `board/agent_profile/` |
| Fine-tuning workflows | `finetune/campaign/`, `distill/`, `hparam_search/`, `registry/`, and `serving/` |
| Fine-tuning execution | `finetune/trainer.py` for backend orchestration and `training_runtime.py` for PEFT/TRL runtime helpers |
| Gold verification | `goldset/verify_acceptance*.py`, `verify_card*.py`, `verify_commands.py`, `verify_ref*.py`, `verify_sampling/`, `verify_multi/agreement_metrics.py`, `verify_multi/agreement_report.py`, `verify_multi/consensus.py`, and `verify_session/` |
| Ontology and PDF preparation | `prep/ontology/pipeline/`, `prep/ontology/artifacts/`, and `prep/pdf/` |
| Ontology endpoint construction | `prep/ontology/endpoints/config.py` for immutable data and `endpoints/builder.py` for validation/construction |
| RAG preparation | `rag/chunking/` and `rag/query_prep/` |
| External RAG review | `scoring/external_rag/` and `scoring/external_rag_session/` |
| Judge scoring and rating | `scoring/judge/` and `judge/rate/` |
| Text-analysis benchmark | `bench/text_analysis/` |
| Prompt rendering and harness lookup | `prompts/engine.py`, `prompts/registry.py`, `prompts/registry_generation.py`, and `bench/harness/registry.py` |
| Optimization internals | `optimize/multi_objective_{trial,runtime,study}.py` and `optimize/joint_search/{schedule,schedule_steps}.py` |

The maintainability refactor validated on 2026-07-18 keeps the changed Python modules and tests
below the 250-line soft target. It uses direct owner-module imports rather than compatibility
facades: knowledge-tree callers select `knowledge_tree_source` or `knowledge_tree_render`, and the
bilingual cutoff flow selects `translation_models`, `translation_artifacts`, or
`translation_workflow`. Chain-context result/report projection lives in
`bench/chain_context/report.py`; benchmark orchestration remains in `bench/chain_context/run.py`.
Focused tests follow the same behavioral boundaries, including separate joint-search halving,
schedule, finalist-resume, Optuna-resume, and pick-resume modules. Validation: `make ci` passes
Ruff formatting/lint, mypy over 558 source files, and 1,477 lightweight tests with 42 slow tests
deselected; `make lint-md` also passes.

The long-module readability refactor validated on 2026-07-27 split every Python file from the
reported top-20 list at a domain boundary while leaving `make/config.mk` intact:

- embedder-adoption screen contracts, bundle recovery, sampling, rendering, sweep execution,
  cross-model execution, roster contracts, and property-separation statistics have focused owner
  modules under `eval/embedder_adoption/`;
- corpus governance metadata and corpus fingerprinting have separate owners in
  `prep/corpus/governance.py` and `prep/corpus/fingerprints.py`;
- paired evidence interpretation lives in `rag/fusion_evidence/paired.py`, separate from bootstrap
  primitives in `rag/fusion_evidence/stats.py`;
- fourteen catch-all test modules are grouped into behavior-named test files with shared setup in
  adjacent intent-named helper modules.

Every Python module produced by the split is at most 234 lines. Cohesive pre-existing classes,
algorithms, reports, and command surfaces outside the supplied list remain whole under the
soft-limit rule. Validation: `make ci` passes Ruff format/lint, mypy over 735 source files, and
2,230 lightweight tests with 45 tests deselected.

The follow-up long-module refactor validated on 2026-07-27 reduces every module in the next
reported list to at most 247 lines, except the intentionally unified `make/config.mk`. Symbols
move to one canonical owner and all callers import that owner directly; there are no forwarding
exports, legacy aliases, or compatibility modules.

- RAG store construction and persistence, lexical indexing, duplicate/noise-floor contracts, and
  conflict-vector scalar and batch operations have focused owner modules.
- PDF repeat contracts and offset remapping, fusion-policy evaluation, report-only sections, graph
  state contracts, and CLI execution/input helpers are separated from their orchestrators.
- Embedder-adoption screen registration is a distinct command module, and the RAG Make targets are
  directly included from store, comparison, and run target-family files.
- Refresh-merge contracts, duplicate-residue contracts, normalization tables, answer-routing
  report text, and query-preparation test fixtures have focused owners after the next files surfaced
  at the soft-limit boundary.
- Four catch-all test modules are split by statistics/verdict, comparison/CLI, audit/CLI, and
  gate/report behavior with adjacent shared fixtures.

`make/config.mk` remains the one intentional larger code/config file; the source files that sit
over the 250-line soft target are cohesive lanes kept whole on purpose, and
`scripts/code_quality.sh` lists them so the set stays visible. Validation: `make ci` passes Ruff
format/lint, mypy over 917 source files, and 2,880 lightweight tests with 60 deselected.
`scripts/code_quality.sh`, `git diff --check`, and the direct-owner stale-import audit also pass.

The readability pass validated on 2026-08-19 works the next `scripts/code_quality.sh` list from the
top down, splitting only where a module carried two subjects and leaving flat configuration and
declaration files (`make/config.mk`, `core/config_fields.py`, `quality/acceptance_gate_registry.py`,
the Typer option blocks) whole. It takes the reported production maximum from 392 to 302 lines and
the over-limit count from 58 to 45. Symbols move to one canonical owner and every caller imports
that owner directly; there are no forwarding exports or compatibility modules.

- Console rendering and pre-run resolution leave their command modules:
  `cli/prep/conflicts_output.py`, `cli/prep/conflict_null_research_support.py`, and
  `cli/rag/compare_embeddings_setup.py`.
- The independent-null research matrix separates dispatch from payload: `conflicts/null_research/run.py`
  validates a generation and prepares the shared geometries, `null_research/summaries.py` builds the
  four per-generation records around one envelope and one `ResearchBudget`, and
  `controls/synthesis_bank.py` assembles a verified control bank.
- The lost-pair attribution becomes three modules -- `conflicts/governance/stage_rule.py` (stage
  vocabulary and the per-pair rule), `governance/stage_search.py` (which lost pair is named), and
  `governance/stage.py` (the payload and its sentence).
- Scoring one built store moves to `rag/embedding_bakeoff/scoring.py`, leaving
  `embedding_bakeoff/run.py` as the roster driver.
- Agentic lanes separate the prospective contract from the reading of its runs
  (`bench/loop_feedback/transfer_design.py`,
  `loop_feedback/adaptation_design.py`), the grid from the sweep
  (`loop_policy/grid.py`), one replayed episode from the arm comparison
  (`policy_change/replay_episode.py`), the elision solver from the interaction conditions
  (`interaction/elision.py`), and the per-policy bundle writes into
  `context_policy/report_persist.py`.
- The stdlib reach measurement separates on-disk layout evidence
  (`quality/gpu_guard/reach/layout.py`) from the coverage partition and its report.
- Three duplicate copies of the paired-delta reader collapse into
  `bench/loop_feedback/outcomes.py`; `drive_with_backend` in
  `bench/common_backend.py` and `tune` in `optimize/tuner.py` are decomposed in place; the search
  space commentary orphaned in `tuner.py` is re-homed onto the constants it describes in
  `optimize/tuning_space.py`.

Validation: `make ci` passes Ruff format/lint, mypy over 1,105 source files, and 3,743 lightweight
tests with 50 deselected; `scripts/code_quality.sh` reports no complexity or shell-lint finding.

### Package shape

A package root is a table of contents, so it names SUBJECTS rather than listing every module. The
subpackage refactor validated on 2026-08-20 applied that to the roots a reader could no longer scan:
`bench` went from 142 modules in one directory to 6, `conflicts` from 91 to 8, `rag` from 64 to 9.
Modules keep one canonical owner and every caller imports it directly -- the subpackage `__init__.py`
holds its docstring and nothing else, so no import path is served by two names.

Inside a subpackage a module drops the prefix its package now carries: `agentic_memory_fold_step.py`
became `bench/memory/fold_step/run.py`, and `null_research_conformal.py` became
`conflicts/null_research/statistics/conformal.py`. The rule for reading a path is that each segment
narrows the subject, and the leaf names the part.

| Package | Root holds | Subpackages |
| --- | --- | --- |
| `bench` | the shared model seam and tool world | `agentic` (the loop), `harness`, `context_policy`, `controller_authority`, `loop_policy`, `loop_feedback`, `memory` (10 study packages), `policy_change`, `published_value`, `chain_context`, `security`, `summarization`, `tooling`, `text_analysis`, `knowledge_cutoff` |
| `conflicts` | the audit entry, its contracts, and store access | `tiers`, `claim`, `semantic_tree`, `calibration`, `governance`, `bundle`, `grouping`, `resolution`, `report`, `null_research` (generations, controls, statistics, report) |
| `rag` | the source-span metric and the cross-cutting seams | `encoders`, `embedding_bakeoff`, `vector_store`, `chunking`, `query_prep`, `duplicates`, `noise_floor`, `comparison`, `fusion`, `fusion_evidence`, `fusion_calibration`, `refresh`, `rerank_bakeoff`, `multihop_probe`, `paired_reading_audit` |
| `prep` | ingestion entry points | `corpus`, `frontier`, `security`, `squad`, `goldset`, `ontology` (endpoints, extraction, drafting, coverage, compare), `pdf`, `curation` |
| `quality` | the repo's own gates | `gpu_guard` (with `reach` for the stdlib and installed scans) |
| `scoring` | the board, aggregate, and per-signal scorers | `composite`, `structured`, `text_analysis`, `security`, `tooling`, `judge`, `policy`, `external_rag`, `frontier_agreement` |
| `eval` | the shared lane contracts | `query_robustness`, `restoration_sweep`, `answer_quality`, `context_ablation`, `embedder_adoption` |
| `cli/bench` | the shared command helpers | `categories`, `context`, `loop`, `memory`, `knowledge_cutoff` |

The test tree mirrors those groups, so the tests for a module sit under the same subject name:
`tests/llb/bench`, `rag`, `conflicts`, `eval`, `prep`, `prep/ontology`, `quality`, `scoring`,
`backends`, `board`, `finetune`, and `goldset` are all grouped this way, and shared fixture modules
stay at the package test root. A test that reads a fixture from another test module imports it by
its full `tests.llb....` path rather than by bare module name: a bare import only resolves because
pytest puts the importing file's own directory on `sys.path`, which stops being true as soon as the
tests are grouped, and the dotted form resolves from anywhere.

Validation: `make ci` passes Ruff format/lint, mypy over 1,174 source files, and 3,743 lightweight
tests with 50 deselected; `make lint-md`, `lint-doc-links`, and `scripts/code_quality.sh` also pass.

`scripts/quickstart.sh` is the process/configuration entry point and sources functional fragments
from `scripts/quickstart/`: `helpers`, `model_select`, `pdf_draft`, `serving`, `track_a`, `track_b`,
`track_c`, and `dispatch`.

## Current-Schema Policy

Persisted artifacts are interpreted through one current schema:

- board manifests declare `split`; score rows are not consulted to infer it;
- miss analysis requires `manifest.json`, `scores.jsonl`, and complete per-item
  `retrieval.jsonl` evidence;
- ontology extraction-journal rows require `parsed=true`;
- calibration worksheet headers match their canonical column list exactly;
- merged tokenizer chat templates come from the Transformers 5 `chat_template.jinja` file.

These boundaries fail visibly when state is incomplete, which keeps the core readers compact and
prevents an inferred artifact meaning from entering a leaderboard or human-review workflow.

The project README stays at capability and navigation level. End-to-end commands live in
`docs/guides/quickstart/quick-start.md`, and focused citation-preserving conversion lives in
`docs/guides/data-prep/pdf-corpus-prep.md`.

## Setup Surface

The repo uses `uv` and `pyproject.toml` for Python dependency management. Project metadata requires
Python `>=3.12`; pytest and build-helper tests derive their behavior and fake wheel ABI tags from
the running supported interpreter.

```bash
make
make venv
make test-fast
make ci
```

`make venv` creates `.venv`, installs the editable package with extras, and seeds `.env` from
`.env.example`. Adding an extra afterwards goes through `make install-extras EXTRAS=<groups>` and
never a bare `uv pip install`, and `make lock-drift` names anything already off the lock (see
[An extras install respects uv.lock](#an-extras-install-respects-uvlock)).
GitHub CI uses the lighter dev dependency set and does not require GPU services.
`scripts/shared/common.sh` resolves `UV_LINK_MODE` adaptively: when uv's cache and this checkout
are on different devices it exports `copy`, otherwise it leaves uv's default link mode in place.
The Quick Start guide keeps each Make wrapper annotated with command purpose, default inputs,
outputs or artifacts, and the expected result. Descriptive quickstart wrappers provide both
all-in-one and grouped execution:

- `make quickstart-goldset`: committed-goldset leaderboard flow; grouped targets are
  `quickstart-goldset-setup`, `quickstart-goldset-rag`, `quickstart-goldset-models`,
  `quickstart-goldset-eval`, `quickstart-goldset-security`, and `quickstart-goldset-prompt`.
- `make quickstart-pdf-corpus`: PDF corpus conversion, RAG indexing, interactive local/frontier
  drafter selection, full-corpus draft goldset/ontology, graph, and validation up to the human
  verification gate; grouped targets are
  `quickstart-pdf-corpus-convert`, `quickstart-pdf-corpus-index`,
  `quickstart-pdf-corpus-draft`, `quickstart-pdf-corpus-graph`,
  `quickstart-pdf-corpus-validate`, `quickstart-pdf-corpus-review`,
  `quickstart-pdf-corpus-accept`, and `quickstart-pdf-corpus-score`.

`scripts/quickstart.sh` owns the grouped orchestration and writes timestamped logs under
`$DATA_DIR/llb/logs/quickstart/` with step headings, called commands, metrics emitted by each tool,
and `[result]` artifact summaries.

The top-level `Makefile` is the public entry point: it sets root variables, includes grouped make
fragments, and defines `help`. Target implementations live under `make/`: `config.mk` for shared
defaults and exported environment, `quickstart.mk` for grouped quickstarts, `dev.mk` for local
development and tests, and `models.mk` for model and serving setup. `data-prep.mk` and `eval.mk`
are ordered include manifests that retain the public help sections while delegating target bodies
to functional fragments:

- `make/data-prep/`: corpus ingestion, curation/calibration, verification, draft generation, and
  draft comparison;
- `make/eval/`: RAG evaluation, fine-tuning, orchestration, prompt systems, security/agentic
  benchmarks, knowledge cutoff, and category/platform runs.

Each fragment owns its `.PHONY` declarations. `make help` scans the complete include graph through
`$(MAKEFILE_LIST)` and uses the `##@` section markers plus `make/help.awk` to print the same grouped,
standard CLI-style target list.

This layout keeps target ownership local while `make help` remains the single discoverable command
surface. Make parsing, representative dry-runs, and the help listing validate the complete include
graph.

The goldset quickstart uses `QUICKSTART_SETUP_VENV=auto`, so it reuses an existing `.venv` and
only syncs dependencies when the venv is missing or `QUICKSTART_SETUP_VENV=1` is set. On CUDA hosts,
`make venv` installs vLLM binary wheels through `scripts/build_vllm.sh` by default
(`VENV_INSTALL_VLLM=auto`; set `0` to skip). The grouped wrappers default the uv cache to
`$DATA_DIR/uv-cache`, skip apt provisioning unless `QUICKSTART_SKIP_APT=0`, and re-export the
Make-level `DATA_DIR` after `.env` is loaded so wrapper artifacts stay under the requested
quickstart root. The goldset quickstart passes `QUICKSTART_SWEEP_LIMIT` to each sweep cell
(defaulting to the Make `LIMIT`, currently 20) so the all-in-one path is bounded on offload-heavy
hosts; set `QUICKSTART_SWEEP_LIMIT=` to run every item in each cell.
The PDF draft wrapper defaults to all converted documents and `QUICKSTART_DRAFT_MODEL=auto` with
`QUICKSTART_MODEL_SELECTION=auto`, which resolves the strongest context-capable Gemma 4 CUDA-tier
target before it estimates and confirms the full draft runtime. Model selection itself does not
prompt, so an approved unattended run proceeds with `QUICKSTART_ASSUME_YES=1`. Benchmark, manual
local, and `frontier` `litellm` routes remain explicit overrides.

### The vLLM install respects uv.lock

`make venv` syncs the lock and THEN installs vLLM, whose own requirements are mostly unpinned, so
the second step was free to upgrade a package the first step had just pinned. Six of vLLM's
requirements are packages this project also declares (`mcp`, `numpy`, `openai`, `psutil`,
`pydantic`, `pyyaml`), and the one that actually moved is `mcp`: no `make venv` extra installs it,
vLLM requires it with no specifier, so the install pulled the newest major -- `mcp` 2.0.0 over the
1.28.1 the lock resolves. `mypy` then failed on `src/llb/bench/mcp_server.py`, which reads like a
source bug and is a dependency resolution, and `make ci` stayed red until someone repaired the venv
by hand.

`llb.build.lock_guard` closes that at the point of drift. `llb.build.lock_reader` answers the three
inputs -- which packages `pyproject.toml` declares, which versions `uv.lock` resolved for each, and
which of those the interpreter holds -- and the guard turns them into a uv `--constraint` file that
every `uv pip install` in `llb.build.vllm` runs under. A package is constrained only to versions the
lock actually carries: an exact pin when the environment already holds a locked version or the lock
carries exactly one, and the closed range the lock spans when conflicting extras forked it (`mcp` is
1.26.0 under the crewai extra and 1.28.1 elsewhere, so the constraint is `>=1.26.0,<=1.28.1` --
which excludes 2.x without evaluating an extra marker to guess which fork this venv is).

After the install the environment is re-read and compared with a pre-install snapshot, because the
two kinds of drift are different findings. A package this install MOVED off the lock fails the
install by default; a package that was already off the lock is an earlier install's leftover (see
[An extras install respects uv.lock](#an-extras-install-respects-uvlock)) and is logged as a
warning naming the package, so an unrelated drift never fails someone's vLLM install.
`LLB_VLLM_LOCK_GUARD` sets the cost of a caused drift: `refuse` (default), `report`, or `off`. The
one failure the constraint introduces -- a future vLLM needing a version the lock lacks -- is
reported with its remedy (`uv lock --upgrade-package <name>`, then commit the lock).

Measured on the 16 GiB CUDA host (2026-08-16): the same clean-venv resolution installs `mcp==2.0.0`
unconstrained and `mcp==1.28.1` under the 48-package lock constraint, and a real `make build-vllm`
reports `the vLLM install moved nothing off uv.lock`. Coverage is
`tests/llb/build/test_lock_guard.py` (constraint planning, drift attribution, all three guard
modes) plus a `--constraint` assertion in `tests/llb/build/test_build_helper.py`.

The paired decision on the SDK itself is recorded with the transport in
[Category suite](category-benchmark-suite.md#tooling): the `[mcp]` extra now carries a `<2` bound,
because mcp 2.x replaces the low-level server API rather than renaming a field.

### An extras install respects uv.lock

Running that guard surfaced the same failure class one step over. A hand-installed optional extra
went through a bare `uv pip install -e ".[review]"`, and uv's pip interface has NO lockfile: it
re-resolves the whole requirement set and takes the newest version each specifier admits. Measured
on a clean venv, the `dev` extra alone lands **ten** declared packages off the lock -- `ruff`
0.16.3 against 0.15.18, `mypy` 2.3.1 against 2.1.0, plus `numpy`, `openai`, `typer`,
`huggingface-hub`, `anyio`, `pymupdf4llm`, `pymarkdownlnt`, and `textual`. Two of those are the
linter and the type checker `make ci` runs, and the `dev` extra pins them EXACTLY on purpose, so a
local `make ci` verdict stops matching GitHub CI's -- which is what the pins exist to prevent.

`llb.build.extras` gives the extras install the vLLM treatment. `make install-extras EXTRAS=<groups>`
assembles `uv pip install --python <venv> --constraint <lock-derived> -e ".[<groups>]"`, so the
resolution is held to versions `uv.lock` carries, and re-checks the environment afterwards. Nothing
in it is extras-specific beyond the `.[a,b]` requirement: `llb.build.lock_guard` grew a `Guard`
record carrying the three strings that differ per caller (what the install calls itself, which
variable relaxes it, the command that repairs it), and the constraint planner, drift report, and
`refuse | report | off` modes are shared with the vLLM path. `LLB_EXTRAS_LOCK_GUARD` is the extras
guard's variable; the vLLM one keeps `LLB_VLLM_LOCK_GUARD`, so relaxing one never relaxes the other.

Only DECLARED packages are constrained, and that boundary is load-bearing rather than incidental:
this host's `torch` is 2.11.0 (what vLLM 0.24.0 pinned) while the lock resolves 2.12.1, so a lock
that reached past `pyproject.toml` would replace the hardware-matched torch on every extras
install. A constrained `.[finetune]` install leaves torch untouched and moves only the declared
packages.

`make lock-drift` is the report half. `uv.lock` says which version each declared package should be
at, and `pyproject.toml` says which extra declares it, so the report names both plus the single
command that resolves the whole set:

```text
5 declared package(s) sit off uv.lock:
  ruff 0.15.20 is off the lock (pinned: 0.15.18) -- declared by dev
  textual 8.2.8 is off the lock (pinned: 8.2.7) -- declared by dev,review
  ...
put them back with: make install-extras EXTRAS=dev,finetune,prep,rag,review
```

It exits non-zero on drift; `LLB_EXTRAS_LOCK_GUARD=report` downgrades that while an upgrade is
deliberately being tested.

Measured on the 16 GiB CUDA host (2026-08-16): the host carried five off-lock declared packages
(`deepeval` 4.0.7/4.0.6, `litellm` 1.90.0/1.89.3, `ruff` 0.15.20/0.15.18, `textual` 8.2.8/8.2.7,
`trl` 1.8.0/1.7.1). The printed command resolved all five in one install -- no hand-editing, no
package named twice -- and left `torch` 2.11.0, `vllm` 0.24.0, and `flashinfer-python` 0.6.12 in
place. `make lock-drift` now reports `every declared package matches uv.lock`. Coverage is
`tests/llb/build/test_extras.py`: argument assembly, the group-ownership report, the guard-mode
exit codes, an unknown-extra refusal that never reaches uv, an end-to-end run of
`scripts/install_extras.sh` against a fake `uv`, and a repo-level assertion that this venv's
declared packages all sit on the lock.

The one extra outside this path is `[crewai]`, which pyproject declares as a CONFLICTING extra and
which lives in a dedicated environment; it installs lock-exactly with
`UV_PROJECT_ENVIRONMENT=<dir> uv sync --frozen --extra crewai` (see
[CrewAI harness](../../guides/benchmarking/crewai-harness.md)), and `make lock-drift` reads the
project `.venv` only.

### `make venv` says reuse or rebuild

Both guards above protect what is IN the venv; this one protects the venv itself. `make venv`
printed `reusing .venv -- updating deps` whenever `.venv/bin/python` existed, and `uv sync
--inexact` leaves packages the lock does not name alone -- but neither holds once the SYSTEM
interpreter the venv points at is patched underneath it. `pyvenv.cfg` records `version_info` at
creation time and never again, so an OS python upgrade leaves the recorded version behind the real
one, and uv then calls the environment stale and REPLACES it. Measured here: `uv sync --inexact
--frozen --dry-run` reports `Would replace project environment at: .venv` and 140 packages to
download, where a venv whose recorded version still matches reports `Would use`. The replacement
installs the lock's `torch` 2.12.1 (CUDA 13) over the 2.11.0 (CUDA 12) vLLM 0.24.0 requires. On a
CUDA host the `VENV_INSTALL_VLLM=auto` step afterwards pins torch back, so the damage is a silent
full reinstall; with `VENV_INSTALL_VLLM=0`, or on a host that skips that step, the venv is left
holding a torch its vLLM cannot use while the target reported a reuse.

`llb.build.venv_interpreter` reads the fact -- what `pyvenv.cfg` recorded, which interpreter `home`
resolves to now, and what that interpreter's version actually is -- and `llb.build.venv_state` turns
it into the plan `make venv` announces BEFORE syncing: `create`, `reuse`, `rebuild`, or `unchecked`.
A rebuild is priced in the packages the sync will not put back, which is a different question from
`lock_guard.find_drift`: that one asks whether a DECLARED package sits off the lock, while a replace
also drops everything the lock never carried (vLLM, flashinfer, the CUDA wheels). The hardware-matched
names are listed and the rest counted. Three consequences follow the plan:

- **the message is honest** -- `REBUILDING ... records python 3.13.14; /usr/bin/python3.13 is now
  3.13.15` instead of `reusing`;
- **the vLLM reinstall stops being skippable** -- a sync that moved the stack forces
  `scripts/build_vllm.sh` afterwards, `VENV_INSTALL_VLLM=0` included, because that flag was set for
  an environment the sync no longer leaves in place;
- **an unrequested rebuild is refused** -- `LLB_VENV_STALE_GUARD` takes the same `refuse` (default)
  `| report | off` as the other two guards, and `RECREATE_VENV=1` is the explicit "yes, replace it".

**A REUSE moves the stack too, and that is the case running it found.** `--inexact` promises only
not to REMOVE what the lock does not name; a package INSIDE the resolution is still installed at the
locked version, and `torch` is inside it (`rag` -> sentence-transformers). Measured here on a venv
that was not stale at all: `make venv VENV_INSTALL_VLLM=0` reported `reusing` and downgraded
`torch` 2.13.0 -> 2.12.1 under a vLLM 0.27.1 that needs 2.13.0. So the plan prices a reuse as well,
over the packages the lock also carries, and forces the reinstall on the same rule. The two actions
put different things at stake, and the pricing says so: a rebuild loses everything (including what
the lock never carried -- vLLM, flashinfer, the CUDA wheels), while a reuse only re-pins what the
lock names.

A rebuild has a third bucket the first run missed: a package can match the lock exactly and still
not come back, because `uv sync` installs the extras it was ASKED for. `bitsandbytes` 0.49.2 matched
`uv.lock` and vanished in the measured rebuild -- only `[finetune]` declares it, and `make venv`'s
default extras do not include that group. Those are now named separately, with the
`make install-extras EXTRAS=<group>` that restores them. Transitive-only packages stay out of reach
without resolving the lock's graph, so the line names what an operator installed on purpose rather
than claiming to enumerate every casualty.

The refusal has to be actionable, so it names the cheap way out first. Within one `major.minor` a
CPython patch release keeps the ABI tag (`cp313`) and the `site-packages` layout, so the venv is
ALREADY running the patched interpreter through its `bin/python` symlink and every compiled wheel in
it still loads -- only the recorded string is behind. `make venv-restamp` records the truth and the
whole stack stands. A MINOR move is a different environment (new stdlib, new layout, unloadable
extensions), so the restamp is refused there and the rebuild is the only answer -- the venv is built
against the SYSTEM interpreter by design (a managed interpreter is out of scope), so that residual
case remains a real rebuild.

`make venv` now delegates the whole lifecycle to `scripts/setup_venv.sh` (plan, then sync, then the
vLLM step), which also rejects an unusable `VENV_INSTALL_VLLM` before the sync rather than after it.
A planner that cannot run at all logs one line and lets uv decide, because `make venv` is the
command an operator runs to repair a half-broken venv.

Measured on the 16 GiB CUDA host (2026-08-16): `.venv` recorded python 3.13.14 against a patched
`/usr/bin/python3.13` at 3.13.15, and `make venv` refused, naming `flashinfer-python 0.6.12`,
`torch 2.11.0 -> 2.12.1`, `torchaudio 2.11.0`, `torchvision 0.26.0 -> 0.27.1`, `vllm 0.24.0`, and
91 more -- while uv's own dry run agreed (`Would replace`).

The rebuild was then accepted and run for real (`make venv RECREATE_VENV=1 VENV_INSTALL_VLLM=0`):
uv downloaded 140 packages, the forced vLLM step ran despite `=0`, and the venv came back with
`vllm` 0.27.1 and `torch` 2.13.0+cu130 -- the torch that vLLM pinned, not the lock's 2.12.1.
`VLLM_SPEC` is unpinned, so the reinstall takes the newest vLLM (0.24.0 -> 0.27.1 here) and its
torch with it. `make lock-drift` reported `every declared package matches uv.lock`, `make ci` was
green (3479 passed), and a served smoke request on the rebuilt stack answered in Ukrainian:
`llb run-eval --backend vllm --model google/gemma-4-E4B-it-qat-w4a16-ct --limit 1` at 64.4 tok/s,
peak VRAM 15893 MB, served ctx 8192, retrieval recall@5 = 1.000. The two findings above -- the
`bitsandbytes` bucket and the reuse-side re-pin -- both came out of that run rather than out of
review. Coverage is
`tests/llb/build/test_venv_interpreter.py` (the recorded-vs-running comparison, the restamp and its
minor-move refusal) and `tests/llb/build/test_venv_state.py` (the four actions, rebuild and reuse
pricing, the unsynced-extra bucket, the refusal and its remedies, and end-to-end runs of
`scripts/setup_venv.sh` against a fake `uv` proving the refusal happens before any sync, that
`VENV_INSTALL_VLLM=0` is obeyed when the sync moves nothing, and that it is overridden when the
sync moves the stack).

Host-specific acceptance procedures and serving constraints live in
[Host validation](host-validation.md) and [Platform matrix](platform-vector-matrix.md).

Runtime paths resolve from the project root and honor `DATA_DIR`; the default is `.data`.
Generated artifacts must stay under `DATA_DIR`.

## Main Command Areas

| Area | Commands |
| --- | --- |
| Corpus prep and hygiene | `pdf-to-markdown`, `ingest-corpus`, `strip-corpus-repeats`, `audit-repeat-yield`, `audit-corpus-conflicts`, `resolve-corpus-conflicts`, `compare-conflict-granularity`,
`recompute-conflict-stage`, `calibrate-conflict-adjudicator`, `measure-duplicate-residue` |
| Gold data | `prepare-goldset-draft`, `validate-goldset`, `ingest-squad`, `ingest-uk-squad`, `curate-drafts`, `import-external-draft` |
| Verification and review | `cross-check-goldset`, `verify-sample`, `verify-review`, `verify-adjudicate`, `verify-accept`, `review` (`make review-workbench`) |
| Judge calibration | `calibration-worksheet`, `calibration-run`, `calibration-rate`, `calibration-score`, `judge-experiment`, `frontier-judge-agreement` |
| RAG retrieval | `build-index`, `build-graph`, `refresh-index`, `validate-retrieval`, `build-query-glossary` |
| Retrieval evidence | `compare-retrieval`, `compare-embeddings`, `compare-vector-stores`, `compare-graph-fusion`, `calibrate-fusion-routing`, `compare-context-strategies`, `compare-answer-quality`, `sweep-restoration-constraints` |
| RAG scoring | `run-eval`, `sweep`, `tune`, `joint-search`, `pipeline`, `screen-public`, `board`, `recommend`, `mlflow-ui` |
| Diagnostics | `analyze-misses`, `probe-context-position`, `probe-multihop-hops`, `bench-query-robustness`, `score-external-rag` |
| Fine-tuning and adapters | `export-finetune-set`, `finetune-hparams`, `finetune-adapter`, `finetune-compat`, `self-improve`, `distill`, `finetune-campaign`, `register-adapter`, `list-adapters`, `serve-adapter`, `gc-adapters` |
| Backends | `prep-models`, `list-models`, `resolve-models`, `preflight-vllm`, `build-vllm`, `build-llamacpp` |
| Category suites | `bench-security`, `bench-*`, `bench-chain-context`, `bench-composite`, `composite-headline` |
| Knowledge cutoff | `bench-knowledge-cutoff`, `knowledge-cutoff-ua-*`, `bench-knowledge-cutoff-bilingual` |
| Prompt systems | `prompt-system-prepare`, `prompt-system-review`, `prompt-system-compare` |
| Platform matrix | `platform-matrix`, `detect-gpu-vram`, `gen-serving-config`, `prep-serving-targets` |
| Orchestration | `auto-rag`, `quickstart-goldset`, `quickstart-pdf-corpus`, `quickstart-corpus` |

The CLI entry point is `src/llb/main.py`; command modules live under `src/llb/cli/`.

## Source Layout

```text
src/llb/
  cli/              Typer command modules and config helpers
  core/             canonical RunConfig, contracts, env and filesystem helpers
  goldset/          canonical gold schema, validation, splits, review ledger tooling
  prep/             ingestion, drafting, cross-check, public-source adapters
  conflicts/        corpus-hygiene tiers, conflict detection and reversible resolution
  review/           unified terminal review workbench and its per-ledger adapters
  rag/              chunking, embeddings, vector stores, retrieval comparison and evidence
  graph/            GraphRAG model, store, retrieval, summaries
  backends/         launchers, hardware detection, planning, resolver, telemetry
  inference/        serving selection and generated serve/run configs per GPU tier
  eval/             retrieve-generate graph templates, context ablation, answer quality
  executor/         run orchestration, isolation, VRAM and contention gates
  optimize/         Optuna tuning space, multi-objective study, joint model+config search
  scoring/          correctness, judge, board aggregation, category metrics
  judge/            judge calibration statistics, worksheets, experiments
  screen/           Tier-1 public UA screen and its report
  finetune/         dataset export, LoRA trainers, hparam search, adapter registry
  bench/            category benchmark runners, chain-context policies, tool worlds
  prompts/          shared prompt-template engine, templates, generated registry
  prompt_system/    prompt-system packages, review state, selection
  auto_rag/         autonomous corpus-to-recommendation stage machine and journal
  quickstart/       quickstart model-selection helpers
  board/            run loaders, category/harness/prompt-system comparisons, UI
  standalone/       stdlib-only client for scoring a closed remote RAG service
  build/            source-build helpers (vLLM wheels)
  tracking/         canonical manifests and MLflow mirror
```

`samples/` contains committed fixtures and seeds. It is data, not runtime output. Root-level
YAML/JSON fixture files are grouped by use:

| Path | Contents |
| --- | --- |
| `samples/configs/` | candidate model manifest and run-eval config examples |
| `samples/benchmarks/` | category-suite case seeds and tool catalogs |
| `samples/data-prep/` | import and synthetic RAG-item fixtures |
| `samples/goldsets/` | verified committed gold-set bundles with corpus files |
| `samples/verification/` | human-review sample manifests and worksheets |

See `samples/README.md` for the full fixture map.

`tests/` mirrors the package layout instead of holding a flat pile of modules. Tests for
`src/llb/<package>/...` live under `tests/llb/<package>/...`; package submodules may get matching
subdirectories such as `tests/llb/prep/ontology/`. Repository fixture checks that are not tied to
one `llb` package live under `tests/samples/`. The root of `tests/` should stay free of
`test_*.py` files. Pytest explicitly allows recursion into `tests/llb/build/` so it can mirror
`src/llb/build/`.

## Artifact Roots

| Path | Meaning |
| --- | --- |
| `$DATA_DIR/llb/rag/` | chunk records, vector-store metadata, local vector indexes |
| `$DATA_DIR/llb/graph/` | GraphRAG nodes, edges, communities, optional summaries |
| `$DATA_DIR/run-eval/<run>/` | RAG run bundle |
| `$DATA_DIR/<category>/<run>/` | category-suite run bundle |
| `$DATA_DIR/sweep/<id>/` | isolated sweep markers and reports |
| `$DATA_DIR/joint-search/<run>/` | successive-halving ledger, screen/pick/finalist resume markers, joint scoreboard |
| `$DATA_DIR/prompt-system/<run>/` | prompt-system candidates, manifest, review JSON |
| `$DATA_DIR/mlflow/` | local MLflow mirror |
| `$DATA_DIR/llb/serving/gpu-<tier>gb/` | generated serving scripts and run configs |

Tracked human calibration worksheets live in `calibration/` when they are intentionally part of
the reproducible benchmark state. Generated worksheets stay under `$DATA_DIR/llb/calibration/`.

## Test Split

Three suites, split by three markers (`slow`, `heavy_env`, `opt_in_env`; registered in
`pyproject.toml`):
`make test` runs the full local flow, including slow tests and markdown lint; `make ci` /
`make test-fast` run the lightweight suite against the default full local install, deselecting
both `slow` and `opt_in_env`; `make ci-github` (the GitHub workflow target) additionally deselects
`heavy_env` so the base `[dev]`-only GitHub env runs every selected
test with no optional-dependency skips. A test is marked `slow` only when its cost is
intrinsic to the behavior being checked: recursive/langchain chunking integration, multi-trial
Optuna or fine-tune campaign simulations, optional chart rendering, real embedder/model loading,
DeepEval, or subprocess build helpers. A test is marked `heavy_env` when it is quick but needs
an optional extra the base `[dev]` install lacks -- real FAISS store builds (`[rag]`, the
refresh-equivalence suite) or the DuckDB graph engine (`[graph]`). `opt_in_env` is reserved for a
dependency deliberately omitted from the default local environment, such as the LanceDB adapter;
regular CI deselects that lane instead of reporting a skip. `importorskip` inside test bodies
still guards manual partial installs. The lightweight suite keeps pure span math,
fake-backed retrieval/fusion, hparam slice and guard checks, and small manifest integrations;
the full suite keeps the recursive splitter, resume/prune sweeps, and committed-corpus
regressions.

A fourth marker, `gpu_env`, selects nothing: it is the escape hatch for the autouse guard in
`tests/conftest.py` that fails an unmarked test which initializes a CUDA context or imports
`flashinfer`, and that starts an unmarked test's subprocesses with no visible CUDA device, so the
lightweight tier's no-GPU promise is checked rather than assumed (see
[host validation](host-validation/quality-gate.md#code-quality-checks)).

Tests target durable specifications and business rules. Internal builders, helper splits, and
deterministic intermediate values do not get dedicated tests when workflow or domain tests already
guard the observable behavior.

Tool caches (ruff, mypy, pytest, deepeval) resolve from `DATA_DIR` at RUNTIME, never from a static
path in `pyproject.toml`: `make/config.mk` exports `RUFF_CACHE_DIR` / `MYPY_CACHE_DIR` /
`DEEPEVAL_*` (and passes `-o cache_dir=` for pytest, which has no env var), and
`llb_export_tool_caches` in `scripts/shared/common.sh` exports the same values for shell-driven
runs. A config file cannot read `.env`, so a literal `.data/cache/...` default there would keep
writing into the project root after an operator moved `DATA_DIR` -- and `rm -rf $DATA_DIR` would
not clear it. A tool invoked with neither layer loaded falls back to its own gitignored default
(`.ruff_cache/`, `.mypy_cache/`, `.pytest_cache/`), which never masquerades as `DATA_DIR`.

Lint contract: `[tool.ruff.lint].select` in `pyproject.toml` names the enforced rule groups
(`E4`, `E7`, `E9`, `F`) instead of relying on ruff's default set, and the `dev` extra caps the
version (`ruff>=0.6,<0.17`). Ruff 0.16 widened its defaults (import sorting, pyupgrade,
flake8-bugbear, implicit string concatenation, ...), so a GitHub runner resolving a newer ruff than
the dev box reported 1391 findings on unchanged code while the dev box was green. A linter release
must never be able to fail CI on code nobody touched; widening the rule set is a deliberate,
separate change to that `select` list.

## Documentation And Specification Gates

Four checks keep the written product and the built product from drifting apart. The first three run
in `make lint-md` (the full local `make test` flow); the last runs in `make ci-checks`, so it fails
in the change that caused it rather than at the next doc lint.

| Check | Command | What it refuses |
| --- | --- | --- |
| Markdown style | `make lint-md` | pymarkdown findings; config in `pyproject.toml` `[tool.pymarkdown]`. Fix BY HAND -- `pymarkdown fix` corrupts prose on 0.9.38 |
| Link landing | `make lint-doc-links` | A relative link whose file is missing or whose `#anchor` no heading produces (`llb.quality.doc_links`) |
| Citation form | `make lint-doc-links` | A measured result cited by a run directory or a bare run label (`llb.quality.doc_citations`) |
| Spec/plan integrity | `make lint-spec-plan` | A disagreement between the spec's capability registry and `plan.md` (`llb.quality.spec_plan_integrity`) |

### Why a run path is not a citation

`$DATA_DIR/<method>/<run-id>/` is host-local and temporary: it is gone after a cleanup, absent on a
fresh checkout, and absent on every other GPU host. A bare `<timestamp>-<slug>` run label is no
better -- it is a lookup key into a directory the reader does not have. Either one leaves a page
that stops being checkable the moment the run is deleted, which is what `llb.quality.doc_citations`
now refuses. The positive form is in [AGENTS.md](../../../AGENTS.md) ("Citing a measured result")
and [heavy runs and evidence](../../guides/development/heavy-runs-and-evidence.md): what ran on
what, the date and host, every load-bearing number AND its reading, and what would overturn it.

Three shapes stay legal, because none of them makes a host-local claim: a `$DATA_DIR` TEMPLATE with
no run segment (that documents where a command WRITES), a run label inside a table row (where the
row is the description the label trails), and a fenced block quoting a command someone ran. The
guide that STATES the rule is exempt by name in `RULE_DEFINING_DOCS`, because stating it means
quoting the shape it forbids -- and it is the only exemption, `plan.md` included.

The conversion behind the check ran on both GPU hosts, since inlining a run's numbers requires the
machine that holds the bundle. Where a bundle survived on neither host, the page says so and the
numbers already written into it are the record -- 12 of the citations converted on the 12 GiB
Blackwell host were in that state, chiefly older 16 GiB-host runs under `context-ablation`,
`compare-embeddings`, and `query-robustness`.

`lint-spec-plan` is the join between three documents that are otherwise only related by convention:
[the spec](../../design/spec.md) says what the product does, its
[capability registry](../../design/spec.md#capability-registry) says how each capability is
evaluated and where it is implemented, and [plan.md](../plan.md) holds the remaining work. It
enforces four invariants:

- every plan task carries a `Serves` line naming a registered capability id, and sits in that
  capability's `###` group;
- every `shipped` capability links to implementation docs; every `planned` one has at least one open
  task;
- every capability declares a non-empty evaluation;
- the plan's capability groups appear in the registry's row order, in both task sections.

The last one is what keeps "what do I work on next" a position rather than an argument: the registry
row order IS the implementation line, and it runs down the trust chain. The checker never judges
whether a capability is worth having -- only whether both documents say the same thing about it. A
capability found while implementing is added through
[Extending this specification](../../design/spec.md#extending-this-specification) before it becomes
tasks, which is what stops undocumented capability from accumulating in `src/`.
