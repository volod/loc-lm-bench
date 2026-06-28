# loc-lm-bench Current Implementation

This index is for agents and maintainers who need the current implementation shape: what exists,
where it lives, how the major flows run, and why the important design choices were made.

For the product design, read [`docs/design/spec.md`](../design/spec.md). For future work, read
[`docs/impl/plan.md`](plan.md).

## Topic Map

| Need | Read |
| --- | --- |
| System shape, setup, repo layout, artifact roots | [Overview](current/overview.md) |
| Gold data, verification, calibration, ingestion, chunking | [Data prep](current/data-prep.md) |
| RAG run path, retrieval, scoring, manifests, MLflow | [RAG core](current/rag-core.md) |
| vLLM launcher, telemetry fields, backend build rules | [Backend telemetry](current/backend-telemetry.md) |
| Model resolution, sweeps, tuning, screens, board, judge | [Evaluation rigor](current/rigor-board-judge.md) |
| VRAM planning, contention guard, llama.cpp, ontology drafting | [Robust backends](current/robustness-ontology-backends.md) |
| Security, tooling, agentic, summarization, structured, text analysis | [Category suite](current/category-benchmark-suite.md) |
| Prompt template registry and review workflow | [Prompt templates](current/prompt-templates.md) |
| Knowledge-graph retrieval and graph-vs-vector comparison | [GraphRAG](current/graphrag-backend.md) |
| Backend matrix, power telemetry, vector-store adapters | [Platform matrix](current/platform-vector-matrix.md) |
| Agentic harnesses, judge diagnostics, prompt-system packages | [Extended workflows](current/extended-workflows.md) |
| Host acceptance checklist and repeatable smoke runs | [Host validation](current/host-validation.md) |
| Settled scope and decision motivation | [Product decisions](current/scope-boundaries.md) |
