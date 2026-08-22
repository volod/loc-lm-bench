# RAG Core

The RAG core evaluates one model over a verified gold split:

```text
retrieve -> generate -> classify -> score -> aggregate -> persist
```

It is intentionally backend-neutral. Backends launch differently, but the evaluator talks to an
OpenAI-compatible chat endpoint and receives normalized response classes.

This page is the AREA INDEX: each stage of that pipeline, and each measured decision about it,
lives in its own page under [`rag-core/`](rag-core/). Read down the pipeline order, or jump by the
question you have.

## The pipeline, in order

| Stage | Page | What it answers |
| --- | --- | --- |
| Enter | [Configuration and command path](rag-core/run-path.md) | How a run is configured and invoked, the standalone closed-service runner, and scoring an answer log produced elsewhere |
| Index | [Retrieval store and lifecycle](rag-core/retrieval-store.md) | Store layout, duplicate-chunk collapse, near-duplicate residue and the collapse tiers, and refreshing a store against a changed corpus |
| Chunk | [Chunking strategies](rag-core/chunking.md) | Which splitter to build with, the paired `sentence`-versus-`recursive` re-read, what table-aware chunking guarantees that `recursive` already achieved, and why `size` is a hard cap on every strategy |
| Embed | [Embedder conventions and bake-off](rag-core/embedders.md) | Per-family query/passage conventions, the bake-off lane and its verdict, encoder throughput, cheap-GPU rosters, and the context budget |
| Embed | [The scoring stack and the card-parity gate](rag-core/stack-and-card-parity.md) | What a candidate must clear before it gets a row: reproducing its own model card, and the transformers major its repository code targets; plus the declared load precision that makes two passes comparable |
| Retrieve | [Hybrid retrieval](rag-core/hybrid-retrieval.md) | Dense + BM25 + RRF, the fusion-weight verdict re-read two ways, and what apostrophe-variant tokenization is worth |
| Retrieve | [Graph-vector fusion](rag-core/graph-vector-fusion.md) | Span identity, candidate depth, and question-type routing across the fused lane |
| Re-rank | [Reranking, context order, and query-side processing](rag-core/rerank-and-query.md) | Cross-encoder reranking, context ordering, Ukrainian query preparation, HyDE and decomposition |
| Re-rank | [Reranker bake-off](rag-core/reranker-bakeoff.md) | Which cross-encoder to run, what it buys in first-hit rank, and what it costs in latency and VRAM beside the generator |
| Assemble | [Prompt-side context assembly](rag-core/context-assembly.md) | What the retrieved chunks look like once they are laid into the prompt: restoring a table row block's column names, and why that is off by default |
| Generate | [Generation graph and scoring](rag-core/scoring.md) | The generation graph, the headline objective's verbosity confound, its decomposition and declared ranking policy, groundedness and citation metrics |
| Measure | [Retrieval metrics](rag-core/retrieval-metrics.md) | Recall@k / MRR by source span, the evidence-intactness pair beside them, the paired lane verdict, the per-question-type slices, and the measurement floor |
| Persist | [Backends, persistence, and execution](rag-core/persistence-and-execution.md) | Backend seam, the persisted retrieval record, executor, durability, and the RAG-config sweep grid |

## Reading a verdict

Every comparison lane in this area ends in a verdict, and the rules for reading one are shared
rather than restated per lane:

| Question | Page |
| --- | --- |
| May a paired verdict be read at all, and what does an unreadable one need? | [Paired uncertainty and the adopt-or-retain verdict](rag-core/paired-verdicts.md) |
| Predeclared MDE, paired-power item counts, realized sensitivity | [Paired-power contract](rag-core/paired-verdicts.md#paired-power-contract-for-comparison-lanes) |
| Family-wise error control when a verdict selects a grid row, cell, or candidate | [Selection-adjusted grid verdicts](rag-core/paired-verdicts.md#selection-adjusted-grid-verdicts) |
| What a withdrawn reading needs before it may be read again | [The re-decision](rag-core/paired-verdicts.md#the-re-decision-what-a-withdrawn-reading-needs) |
| When a rank-quality gain is worth adopting at all | [The scoped first-hit-rank adoption bar](rag-core/first-hit-rank-adoption.md) |
| Does RAG pay for itself against closed-book and long-context lanes? | [Context ablation](rag-core/context-ablation.md) |

## Where the evidence sits

Measured evidence stays with the decision it supports rather than in an evidence appendix, so each
page above ends in the runs that settled it. The heaviest evidence chains are the
[embedder bake-off verdict](rag-core/paired-verdicts.md), the
[first-hit-rank adoption bar](rag-core/first-hit-rank-adoption.md) with its five-model roster, and
the [context ablation](rag-core/context-ablation.md) cohort.
