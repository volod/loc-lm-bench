# loc-lm-bench Current Implementation

This index is for agents and maintainers who need the current implementation shape: what exists,
where it lives, how the major flows run, and why the important design choices were made.

For the product design, read [`docs/design/spec.md`](../design/spec.md). For future work, read
[`docs/impl/plan.md`](plan.md).

## Topic Map

| Need | Read |
| --- | --- |
| System shape, setup, repo layout, artifact roots | [Overview](current/overview.md) |
| Autonomous corpus-to-RAG orchestration, resume, verification, recommendation | [Auto-RAG](current/auto-rag.md) |
| Gold data, verification, calibration, ingestion, chunking | [Data prep](current/data-prep.md) |
| Unified terminal review UI, adapters, keys, ledger compatibility | [Review workbench](current/review-workbench.md) |
| RAG run path, retrieval, scoring, manifests, MLflow | [RAG core](current/rag-core.md) |
| vLLM launcher, telemetry fields, backend build rules | [Backend telemetry](current/backend-telemetry.md) |
| Model resolution, sweeps, tuning, joint-search, screens, board, judge, miss analysis | [Evaluation rigor](current/rigor-board-judge.md) |
| VRAM planning, contention guard, llama.cpp, ontology drafting | [Robust backends](current/robustness-ontology-backends.md) |
| Security, tooling, agentic, summarization, structured, text analysis | [Category suite](current/category-benchmark-suite.md) |
| Effective real-world knowledge cutoff for local models | [Knowledge cutoff](current/knowledge-cutoff.md) |
| Prompt template registry and review workflow | [Prompt templates](current/prompt-templates.md) |
| Knowledge-graph retrieval, graph-vs-vector comparison, multi-hop retrieval and answer-quality evidence | [GraphRAG](current/graphrag-backend.md) |
| Backend matrix, power telemetry, vector-store adapters | [Platform matrix](current/platform-vector-matrix.md) |
| Agentic harnesses, judge diagnostics, prompt-system packages, local fine-tuning, adapter registry and lifecycle | [Extended workflows](current/extended-workflows.md) |
| Host acceptance checklist and repeatable smoke runs | [Host validation](current/host-validation.md) |
| Settled scope and decision motivation | [Product decisions](current/scope-boundaries.md) |
