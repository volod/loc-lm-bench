# loc-lm-bench -- Current Implementation Index

This file is the short entry point for delivered implementation facts. Forward work lives in
[`plan.md`](plan.md); the full spec is [`spec.md`](../design/spec.md).

Use `rg <term> docs/impl/current` for detailed facts. Do not bulk-append history here: add new
delivered behavior to the narrowest topic file below, then update this index only when a new topic
or lookup path is needed.

## Topic Map

| Need | Read |
| --- | --- |
| Setup, repo layout, operator workflows, sample data map | [Overview](current/overview.md) |
| Gold-item schema, splits, validation, SQuAD ingestion, chunking, judge calibration basics | [Data prep](current/data-prep.md) |
| Run config, CLI, RAG store, retrieval metrics, eval graph, scoring, manifests, executor | [RAG core](current/rag-eval-skeleton.md) |
| vLLM launcher, telemetry hook, real-model validation path | [Backend telemetry](current/backend-telemetry.md) |
| Backend resolution, sweeps, Optuna tuning, public screen, board, local judge | [Evaluation rigor, board, judge](current/rigor-board-judge.md) |
| VRAM estimates, contention guard, llama.cpp, ontology-assisted drafting | [Robustness, ontology, backends](current/robustness-ontology-backends.md) |
| Composite headline, category benches, text-analysis bundle, sample verification refs | [Category benchmark suite](current/category-benchmark-suite.md) |
| GraphRAG modules, CLI, manifests, tests, graph-vs-FAISS verification | [GraphRAG backend](current/graphrag-backend.md) |
| 16 GB backend matrix, power telemetry, GPU-class extension path, validated multi-vector-store adapters | [Platform and vector-store matrix](current/platform-vector-matrix.md) |
| Agentic harness comparison (LangGraph/CrewAI/loop), judge diagnostics + smoke, RAG prompt-system lane | [Extended workflows](current/extended-workflows.md), [RAG prompt-system guide](../guides/prompt-system-rag.md) |
| Real-host run evidence and hardware validation notes | [Real-host verification](current/real-host-verification.md) |
| Settled design choices, rejected scope, out-of-scope boundaries | [Scope boundaries](current/scope-boundaries.md) |

## Update Discipline

- Keep this file short enough to read before any task.
- Put module paths, commands, run results, and detailed decisions in `docs/impl/current/*.md`.
- If a delivered feature spans topics, record the primary facts once and link from related topics.
- Keep `docs/impl/plan.md` forward-only: remove completed scope from the plan and add only future
  work there.

## Code Quality And Naming Cleanup

Numbered implementation labels have been replaced with descriptive feature names across source,
tests, scripts, and docs. Current-state docs under `docs/impl/current/` now use topic filenames
such as `category-benchmark-suite.md`, `platform-vector-matrix.md`, and `rag-eval-skeleton.md`.

The board loader was split from one large module into focused modules:
`llb.board.runs`, `llb.board.categories`, `llb.board.harnesses`,
`llb.board.prompt_systems`, and `llb.board.io`. `llb.board.data` remains the import facade.
The Streamlit board renderer now delegates to section helpers in `llb.board.app`.

The guarded category composite now uses descriptive public names:
`load_category_composite`, `build_category_composite_rows`,
`CATEGORY_COMPOSITE_RAW_WEIGHTS`, and `normalized_composite_weights`. Its implementation is split
across `llb.scoring.composite_builder`, `composite_types`, `composite_stats`, and
`composite_format`, with `llb.scoring.composite` as the compatibility facade.

How to run the touched paths:

- `uv run --no-sync pytest tests/test_board.py tests/test_harness.py \
  tests/test_prompt_system.py tests/test_ontology_hardening.py -q`
- `python -m compileall -q src/llb tests`
- `scripts/code_quality.sh`

The focused tests and compile check pass. The final quality run no longer lists `board.app`,
`board.categories`, or the category-composite builder in the cognitive-complexity output. It still
reports existing markdown line-length warnings and older high-complexity hotspots in the judge,
verification, benchmark-runner, and scoring modules.
