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
| Gold-item schema, splits, validation, SQuAD ingestion, chunking, judge calibration basics | [Milestone 0 data prep](current/milestone-0-data-prep.md) |
| Run config, CLI, RAG store, retrieval metrics, eval graph, scoring, manifests, executor | [Milestone 1 eval skeleton](current/milestone-1-eval-skeleton.md) |
| vLLM launcher, telemetry hook, real-model validation path | [Milestone 2 backend telemetry](current/milestone-2-backend-telemetry.md) |
| Backend resolution, sweeps, Optuna tuning, public screen, board, local judge | [Milestone 3 rigor, board, judge](current/milestone-3-rigor-board-judge.md) |
| VRAM estimates, contention guard, llama.cpp, ontology-assisted drafting | [Milestone 4 robustness, ontology, backends](current/milestone-4-robustness-ontology-backends.md) |
| Composite headline, M5 category benches, text-analysis bundle, sample verification refs | [Milestone 5 composite benchmarks](current/milestone-5-composite-benchmarks.md) |
| GraphRAG modules, CLI, manifests, tests, graph-vs-FAISS verification | [Milestone 6 GraphRAG](current/milestone-6-graphrag.md) |
| Real-host run evidence and hardware validation notes | [Real-host verification](current/real-host-verification.md) |
| Settled design choices, rejected scope, out-of-scope boundaries | [Scope boundaries](current/scope-boundaries.md) |

## Update Discipline

- Keep this file short enough to read before any task.
- Put module paths, commands, run results, and detailed decisions in `docs/impl/current/*.md`.
- If a delivered feature spans topics, record the primary facts once and link from related topics.
- Keep `docs/impl/plan.md` forward-only: remove completed scope from the plan and add only future
  work there.
