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

## Module Size & Structure

Tracked `.py` / `.sh` files target a ~250-line SOFT limit (AGENTS.md "File-size soft limit"):
split along clear functional seams, but keep a single cohesive structure whole rather than
fragment it. `scripts/code_quality.sh` reports every tracked file over the limit
(`LINE_SOFT_LIMIT`, default 250) so the backlog stays visible.

CLI command modules follow a package-per-area shape: `llb/cli/<area>/__init__.py` imports its
submodules purely to register their `@app.command` handlers on the shared Typer `app` (the same
registration contract as a flat module), and the commands themselves live in intent-named
submodules. The oversized flat modules `cli/prep.py`, `cli/rag.py`, `cli/bench.py`, `cli/eval.py`,
`cli/models.py`, and `cli/finetune.py` are now such packages (e.g. `cli/prep/{corpus, goldset,
security, benchmarks, draft, draft_support, curation}`, `cli/rag/{index, validate,
compare_retrieval, compare_stores}`). Tests that exercised a former flat module's internals import
them from the specific submodule now.

The same seam-based split has been applied to the largest core-path modules. Each keeps its public
import path stable -- a former flat module either becomes a package whose `__init__` re-exports the
public API, or a thin module that re-exports from sibling submodules -- so callers and tests are
unchanged except for a few monkeypatch targets repointed at the new call-site module:

- `executor/runner.py` (910 lines) is now the `run_eval` orchestrator plus sibling modules
  `runner_setup` (eval inputs / store / query-prep / probes), `runner_backend` (launcher lifecycle
  + VRAM guard + runner resolution), `runner_judge` (judge scoring + calibration worksheet),
  `runner_metrics` (aggregation + telemetry), and `runner_target` (run-target/config payload).
- `board/miss_analysis.py` (859) -> package `board/miss_analysis/{model, load, classify,
  recommendations, rec_retrieval, report}`.
- `finetune/hparam_search.py` (845) -> package `finetune/hparam_search/{model, dev_slice, space,
  objective, search, manifest_io}`.
- `prep/ontology/pipeline.py` (786) -> package `prep/ontology/pipeline/{settings, journaling,
  stages, bundle, run}`.
- `prep/pdf_corpus.py` (739) -> orchestration moved into the existing `prep/pdf` package
  (`furniture, render, quality, manifest, reuse, ingest`); `pdf_corpus.py` re-exports.
- `goldset/verify_session.py` (730) -> package `goldset/verify_session/{report, commands,
  decision, loop}` (presentation half already in `verify_card.py`).
- `board/recommend.py` (722) -> package `board/recommend/{model, build, render, sections}`.

`core/contracts.py` stays whole as the plan's justified cohesive exception (one dataclass/TypedDict
family). A backlog of ~77 more `src/` modules and ~36 test modules still sits over the soft limit;
`scripts/code_quality.sh` lists them largest-first.

## Setup Surface

The repo uses `uv` and `pyproject.toml` for Python dependency management. Project metadata requires
Python `>=3.12`; pytest has no legacy interpreter-specific warning filters, and build-helper tests
derive fake wheel ABI tags from the running supported interpreter.

```bash
make
make venv
make test-fast
make ci
```

`make venv` creates `.venv`, installs the editable package with extras, and seeds `.env` from
`.env.example`. GitHub CI uses the lighter dev dependency set and does not require GPU services.
`scripts/shared/common.sh` resolves `UV_LINK_MODE` adaptively: when uv's cache and this checkout
are on different devices it exports `copy`, otherwise it leaves uv's default link mode in place.
The README Quick Start keeps each Make wrapper annotated with command purpose, default inputs,
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
development and tests, `data-prep.mk` for corpus/goldset/verification work, `eval.mk` for
RAG/evaluation/pipeline targets, and `models.mk` for model and serving setup. `make help` scans
all included fragments through `$(MAKEFILE_LIST)` and uses the `##@` section markers plus
`make/help.awk` to print a grouped, standard CLI-style target list.

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
`QUICKSTART_MODEL_SELECTION=gemma4`, which resolves the strongest Gemma 4 CUDA-tier target before
it estimates and confirms the full draft runtime. Benchmark, manual local, and `frontier`
`litellm` routes remain explicit overrides.

Latest validated goldset quickstart evidence on the 16 GiB RTX 4060 Ti host:
`$DATA_DIR/llb/logs/quickstart/quickstart-goldset-20260630-142055.log`. The run detected
`gpu_tier=16`, built 311 FAISS chunks, passed retrieval with `recall@10=0.980` and `mrr=0.847`,
prepared MamayLM, Lapa, Gemma 4, and Qwen 3.6 serving targets from the generated tier config,
resumed four completed default-family sweep cells (Qwen 3.6, MamayLM 12B, MamayLM 27B, and Lapa),
ran one platform-matrix Ollama row for `gemma4:e4b` with quality `0.420` and `61.37` tok/s,
skipped missing vLLM and llama.cpp serving binaries with actionable log lines, ran
`bench-security` on MamayLM 27B, and created 18 prompt-system candidates.

Latest 12 GiB CUDA-host quickstart evidence on the RTX PRO 3000 Blackwell laptop: the setup wrapper
detected `gpu_tier=12` and selected `.data/quickstart-leaderboard/llb/serving/gpu-12gb/tier.json`
despite a stale `gpu-16gb` directory. `make build-vllm` installed/reused vLLM 0.24.0 and recorded
the native sampler fallback for driver 610.43.02. The PDF quickstart now selects
`google/gemma-4-12B-it-qat-w4a16-ct` at `max_model_len=16384`, `gpu_memory_utilization=0.90`,
`cpu_offload_gb=16`, and `kv_offloading_size_gb=32`; a bounded drafter probe confirmed vLLM served
the long-context target with CPU/KV offload on this 12 GiB GPU.

Runtime paths resolve from the project root and honor `DATA_DIR`; the default is `.data`.
Generated artifacts must stay under `DATA_DIR`.

## Main Command Areas

| Area | Commands |
| --- | --- |
| Gold data | `validate-goldset`, `ingest-squad`, `ingest-uk-squad` |
| Verification | `cross-check-goldset`, `verify-sample`, `verify-review`, `verify-accept` |
| Judge calibration | `calibration-worksheet`, `calibration-run`, `calibration-rate`, `calibration-score` |
| RAG retrieval | `build-index`, `validate-retrieval`, `compare-retrieval`, `compare-vector-stores` |
| RAG scoring | `run-eval`, `sweep`, `tune`, `pipeline`, `board` |
| Backends | `prep-models`, `list-models`, `resolve-models`, `build-vllm`, `build-llamacpp` |
| Category suites | `bench-security`, `bench-*`, `bench-composite`, `composite-headline` |
| Prompt systems | `prompt-system-prepare`, `prompt-system-review`, `prompt-system-compare` |
| Platform matrix | `platform-matrix`, `detect-gpu-vram`, `gen-serving-config` |
| Quickstart flows | `quickstart-goldset`, `quickstart-pdf-corpus` |

The CLI entry point is `src/llb/main.py`; command modules live under `src/llb/cli/`.

## Source Layout

```text
src/llb/
  cli/              Typer command modules and config helpers
  goldset/          canonical gold schema, validation, splits, review ledger tooling
  prep/             ingestion, drafting, cross-check, public-source adapters
  rag/              chunking, embeddings, vector stores, retrieval comparison
  graph/            GraphRAG model, store, retrieval, summaries
  backends/         launchers, hardware detection, planning, resolver, telemetry
  eval/             retrieve-generate graph templates
  executor/         run orchestration, isolation, VRAM and contention gates
  scoring/          correctness, judge, board aggregation, category metrics
  bench/            category benchmark runners and deterministic tool worlds
  prompts/          shared prompt-template engine, templates, generated registry
  prompt_system/    prompt-system packages, review state, selection
  board/            run loaders, category/harness/prompt-system comparisons, UI
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
| `$DATA_DIR/prompt-system/<run>/` | prompt-system candidates, manifest, review JSON |
| `$DATA_DIR/mlflow/` | local MLflow mirror |
| `$DATA_DIR/llb/serving/gpu-<tier>gb/` | generated serving scripts and run configs |

Tracked human calibration worksheets live in `calibration/` when they are intentionally part of
the reproducible benchmark state. Generated worksheets stay under `$DATA_DIR/llb/calibration/`.

## Test Split

`make test-fast` runs the lightweight suite used by CI. `make test` runs the full local flow,
including slow tests and markdown lint. A test should be marked slow only when its cost is
intrinsic to the behavior being checked: recursive/langchain chunking integration, multi-trial
Optuna or fine-tune campaign simulations, optional chart rendering, real embedder/model loading,
DeepEval, or subprocess build helpers. The lightweight suite keeps pure span math, fake-backed
retrieval/fusion, hparam slice and guard checks, and small manifest integrations in CI; the full
suite keeps the recursive splitter, resume/prune sweeps, and committed-corpus regressions.
