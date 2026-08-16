# GraphRAG

GraphRAG can run alone with `--retrieval-backend graph` or fuse with the vector lane through
`--retrieval-backend fused --graph-weight <share>`. Both reuse the RAG store seam, so generation,
scoring, manifests, judge gating, and boards do not need separate graph-specific code.

FAISS remains the default for factoid QA. GraphRAG is most useful when the corpus has connected
entities, multi-hop facts, or narrative community structure.

This page is the AREA INDEX: the mechanics and each evidence chain live under
[`graphrag-backend/`](graphrag-backend/).

| Page | What it answers |
| --- | --- |
| [Modules, strategies, and CLI](graphrag-backend/modules-and-cli.md) | The store decision, the module map, the retrieval strategies, the CLI, and what extraction reads |
| [Fusion sweep evidence](graphrag-backend/fusion-sweep-evidence.md) | The graph-weight sweep lane, the accepted-ledger and multi-hop slice readings, the widened review handoff, and the sweep re-read against its measurement floor |
| [Span identity and candidate depth evidence](graphrag-backend/span-and-depth-evidence.md) | How deep to take graph candidates, how a graph hit maps to a source span, and what the merge threshold is worth |
| [Retrieval budget and per-hop evidence](graphrag-backend/retrieval-budget-evidence.md) | The per-hop probe lane, and whether a stuck multi-hop `all-spans@k` is limited by the retrieval budget or by the query |
| [Answer-quality evidence](graphrag-backend/answer-quality-evidence.md) | Whether fused retrieval moves ANSWER quality rather than retrieval rank, and the sidecar-free heuristic calibration |

## Ontology Scope

Graph nodes use the closed 13-type vocabulary in
[`docs/design/graph-ontology-schema.md`](../../design/graph-ontology-schema.md). The closed
vocabulary matters because graph retrieval needs stable typed nodes and relation caps; allowing a
model to invent schema labels would make graph quality and comparison unstable across runs.
