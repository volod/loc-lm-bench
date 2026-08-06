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
| Headline token precision/recall/found-rate decomposition and declared format weight | [RAG core](current/rag-core.md#headline-decomposition-and-declared-ranking-policy) |
| Whether RAG pays for itself: closed-book vs rag vs long-context lanes | [RAG core](current/rag-core.md#context-ablation-does-rag-pay-for-itself-rag-vs-long-context-ablation) |
| How much repeated text an index still holds, and which collapse tier to build with | [RAG core](current/rag-core.md#near-duplicate-residue-and-the-collapse-tiers) |
| Whether a paired verdict may be read at all, and the item count an unreadable one needs | [RAG core](current/rag-core.md#the-minimum-evidence-gate-on-a-paired-reading) |
| Family-wise error control when a verdict selects a grid row, cell, or candidate | [RAG core](current/rag-core.md#selection-adjusted-grid-verdicts) |
| Predeclared MDE, paired-power item counts, and realized sensitivity in comparison lanes | [RAG core](current/rag-core.md#paired-power-contract-for-comparison-lanes) |
| Cold/warm encoder throughput on CUDA hosts (load vs compile vs steady encode) | [RAG core](current/rag-core.md#blackwell-encoder-throughput-decomposition) |
| Cheap CUDA embedder (e5-small) when quality is flat on a 12 GiB host | [RAG core](current/rag-core.md#blackwell-sub-base-encoder-roster-e5-small) |
| vLLM launcher, telemetry fields, backend build rules | [Backend telemetry](current/backend-telemetry.md) |
| Model resolution, sweeps, tuning, joint-search, screens, board, judge, miss analysis | [Evaluation rigor](current/rigor-board-judge.md) |
| VRAM planning, contention guard, llama.cpp, ontology drafting | [Robust backends](current/robustness-ontology-backends.md) |
| Security, tooling, agentic, summarization, structured, text analysis | [Category suite](current/category-benchmark-suite.md) |
| Effective real-world knowledge cutoff for local models | [Knowledge cutoff](current/knowledge-cutoff.md) |
| Prompt template registry and review workflow | [Prompt templates](current/prompt-templates.md) |
| Knowledge-graph retrieval, graph-vs-vector comparison, multi-hop retrieval and answer-quality evidence | [GraphRAG](current/graphrag-backend.md) |
| Backend matrix, power telemetry, vector-store adapters | [Platform matrix](current/platform-vector-matrix.md) |
| Agentic harnesses, agent context policies, judge diagnostics, prompt-system packages, local fine-tuning, adapter registry and lifecycle | [Extended workflows](current/extended-workflows.md) |
| Aggregate-safe agent observation trim + compact finish recovery (count-slice) | [Extended workflows](current/extended-workflows.md#aggregate-safe-trimming) |
| Active compact vs observation-cap on long and memory-dependent transcripts, including summarizer cost | [Extended workflows](current/extended-workflows.md#compact-versus-cap-with-active-compaction) |
| Where compact stops repaying its summary call: cap-fitting cost crossover over depth and prompt guard | [Extended workflows](current/extended-workflows.md#cap-fitting-boundary-surface) |
| Compact routing on one axis: trigger (`compact_share * guard`) and the fold step it selects | [Extended workflows](current/extended-workflows.md#the-routing-rule-lives-on-the-trigger-axis) |
| The compact crossover as a fold-step boundary ("fold no later than step k") rather than a char guard | [Extended workflows](current/extended-workflows.md#the-crossover-is-a-fold-step-not-a-char-guard) |
| The compact summarize call's input bound (`summary_input_cap`), the residual it removes, and what its elision costs | [Extended workflows](current/extended-workflows.md#the-summarize-input-cap-is-step-aligned) |
| Model-free audit of which published compact evidence a summarize-bound change can move, and the crossovers restated under the shipped bound | [Extended workflows](current/extended-workflows.md#published-crossovers-under-the-shipped-cap) |
| Which published agentic numbers any context-policy constant change invalidates (prompt-sequence replay, no GPU) | [Extended workflows](current/extended-workflows.md#what-a-policy-constant-change-invalidates) |
| The committed geometry where a compound policy change and a per-field reading disagree (the compound guarantee's own test) | [Extended workflows](current/extended-workflows.md#the-compound-guarantee-has-a-geometry-that-tests-it) |
| Every pair of auditable policy constants, which one couples enough to separate the two readings, and what blocks the other fourteen | [Extended workflows](current/extended-workflows.md#one-pair-separates-and-the-other-fourteen-are-answered) |
| The CI gate that pins the shipped context-policy constants and fails a drift with its re-run scope | [Extended workflows](current/extended-workflows.md#the-audit-runs-in-ci-on-the-act-that-creates-the-problem) |
| Agent context-policy constant sweep (cap / head-share / keep_last_n pin-or-expose) | [Extended workflows](current/extended-workflows.md#agent-context-policy-constants) |
| keep_last_n on longer transcripts (medium-search keep grid) | [Extended workflows](current/extended-workflows.md#keep_last_n-on-longer-transcripts) |
| Host acceptance checklist and repeatable smoke runs | [Host validation](current/host-validation.md) |
| Settled scope and decision motivation | [Product decisions](current/scope-boundaries.md) |
