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
- `make quickstart-pdf-corpus`: PDF corpus conversion, RAG indexing, draft goldset, graph, and
  validation up to the human verification gate; grouped targets are
  `quickstart-pdf-corpus-convert`, `quickstart-pdf-corpus-index`,
  `quickstart-pdf-corpus-draft`, `quickstart-pdf-corpus-graph`,
  `quickstart-pdf-corpus-validate`, `quickstart-pdf-corpus-review`,
  `quickstart-pdf-corpus-accept`, and `quickstart-pdf-corpus-score`.

`scripts/quickstart.sh` owns the grouped orchestration and writes timestamped logs under
`$DATA_DIR/llb/logs/quickstart/` with step headings, called commands, metrics emitted by each tool,
and `[result]` artifact summaries.
The goldset quickstart uses `QUICKSTART_SETUP_VENV=auto`, so it reuses an existing `.venv` and
only syncs dependencies when the venv is missing or `QUICKSTART_SETUP_VENV=1` is set. It also
defaults the wrapper uv cache to `$DATA_DIR/uv-cache`, skips apt provisioning unless
`QUICKSTART_SKIP_APT=0`, and re-exports the Make-level `DATA_DIR` after `.env` is loaded so
wrapper artifacts stay under the requested quickstart root.

Latest validated goldset quickstart evidence on the 16 GiB RTX 4060 Ti host:
`$DATA_DIR/llb/logs/quickstart/quickstart-goldset-20260630-142055.log`. The run detected
`gpu_tier=16`, built 311 FAISS chunks, passed retrieval with `recall@10=0.980` and `mrr=0.847`,
prepared MamayLM, Lapa, Gemma 4, and Qwen 3.6 serving targets from the generated tier config,
resumed four completed default-family sweep cells (Qwen 3.6, MamayLM 12B, MamayLM 27B, and Lapa),
ran one platform-matrix Ollama row for `gemma4:e4b` with quality `0.420` and `61.37` tok/s,
skipped missing vLLM and llama.cpp serving binaries with actionable log lines, ran
`bench-security` on MamayLM 27B, and created 18 prompt-system candidates.

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

`samples/` contains committed fixtures and seeds. It is data, not runtime output.

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
intrinsic to the behavior being checked: Optuna sweeps, real embedder/model loading, DeepEval, or
subprocess build helpers.
