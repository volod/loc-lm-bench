# GraphRAG

GraphRAG can run alone with `--retrieval-backend graph` or fuse with the vector lane through
`--retrieval-backend fused --graph-weight <share>`. Both reuse the RAG store seam, so generation,
scoring, manifests, judge gating, and boards do not need separate graph-specific code.

FAISS remains the default for factoid QA. GraphRAG is most useful when the corpus has connected
entities, multi-hop facts, or narrative community structure.

## Store Decision

The graph store persists inspectable node and edge JSONL and loads them into DuckDB for query-time
retrieval. DuckDB is used because it is already a Python dependency, supports recursive CTEs, and
keeps the graph runtime local without introducing a separate graph database service.

Community ids are computed offline and stored. Query time only needs DuckDB table queries.

## Modules

`src/llb/graph/model.py`
: Defines `KnowledgeGraph`, `GraphNode`, `GraphEdge`, and `GraphMention`. Mentions and edge
  evidence keep doc ids, offsets, exact text, section titles, ontology type confidence, and
  community ids.

`src/llb/graph/build.py`
: Converts ontology-assisted `DocExtraction` records into nodes and directed edges. Fact endpoints
  that are not known entities become fact-only nodes so grounded evidence is not dropped.

`src/llb/graph/community.py`
: Deterministic label-propagation community assignment. No graph-analytics dependency is needed at
  query time.

`src/llb/graph/retrieval.py`
: Links question terms to graph nodes and serializes subgraphs back into offset-bearing chunk
  records. Linking uses exact alias/name hits plus a conservative Ukrainian stem key for inflected
  forms.

`src/llb/graph/store.py`
: Implements `GraphStore.build`, `save`, `load`, and `retrieve(question, k)`.

`src/llb/graph/summary.py`
: Optional diagnostic community summaries. Summaries are stored separately and are not returned as
  retrieval context because they are abstractive and not span-scored.

The prompt-system knowledge-tree lane may reuse those summaries as an explicitly experimental
system-prompt candidate. This does not change GraphRAG retrieval or span scoring: the summary stays
out of retrieved chunks, its graph-store digest and tree knobs are recorded in prompt-system
provenance, and the candidate is evaluated against its exact no-tree control before pinning.

`src/llb/rag/compare.py`
: Compares FAISS, both graph strategies, and both graph-vector fused strategies through the shared
  `.retrieve` seam. When an ontology bundle's `needle_items.jsonl` sidecar is available, the report
  also emits question-type slices, including explicit `comparative` and `multi-hop` rows.

`src/llb/rag/fusion.py`
: Implements `FusedRetriever`. It queries the vector store (dense or hybrid) and `GraphStore`,
  fuses span ids with n-way weighted reciprocal-rank fusion, drops exact duplicate source spans,
  and preserves the selected record's exact text and offsets. Fused metadata records which lanes
  returned each span and the graph weight.

## Retrieval Strategies

`local_khop`
: Entity-link the question to seed nodes, expand `graph_khop_depth` hops, and serialize node
  mentions plus edge evidence. This is the graph path for connected fact questions.

`global_community`
: Link the question to communities and serialize member nodes and edges from those communities.
  This is the narrative/theme path.

Both strategies return chunk-like records with exact source spans so the normal retrieval metric
applies.

## CLI

```bash
llb build-graph --bundle <prepare-goldset-dir>
llb build-graph --extraction <extraction.jsonl> --corpus-root <dir>
llb build-graph --corpus-root <dir> --extract-model llama3.2:3b
llb build-graph --bundle <dir> --summarize --summarize-model llama3.2:3b
llb validate-retrieval --retrieval-backend graph --retrieval-strategy local_khop
llb validate-retrieval --retrieval-backend fused --graph-weight 0.3
llb compare-retrieval --graph-weight 0.3 --k 10 --out report.json
llb run-eval --retrieval-backend graph --retrieval-strategy global_community ...
llb run-eval --retrieval-backend fused --graph-weight 0.3 ...
```

`RunConfig` carries `retrieval_backend`, `retrieval_strategy`, `graph_khop_depth`, and
`graph_weight` (default 0.3). These values are part of the config fingerprint and manifest. The
sweep grid accepts `graph_weight=...` and selects the fused backend; the Optuna space samples the
graph weight when its base config is fused. `graph_weight=0.0` is an exact vector passthrough and
does not query the graph lane; `1.0` is an exact graph passthrough.

Graph-vector fusion uses undamped reciprocal ranks (`k=0`) because graph evidence spans and vector
chunks rarely share exact boundaries. With the standard hybrid damping constant of 60, a graph
weight of 0.3 cannot place a graph-only candidate above any vector candidate in a top-10 result,
making the advertised graph share ineffective. Dense+BM25 hybrid retrieval keeps the standard
constant of 60.

## Extraction Inputs

Graph build inputs come from ontology-assisted drafting:

- a full draft bundle with `extraction.jsonl`, `corpus/`, and `ontology.json`;
- an explicit extraction file plus corpus root;
- fresh local extraction over a corpus.

Fresh extraction can disable hidden reasoning with `--extract-no-think`. For Ollama reasoning
models this uses the native `/api/chat` path because the OpenAI-compatible `/v1` path does not
honor the `think` control.

The graph build path has been smoke-tested with
`.data/prepare-goldset/{timestamp}-smoke`: it loaded two drafted extractions and wrote 19 nodes,
7 edges, and 12 communities under `$DATA_DIR/llb/graph/`.

The PDF ontology draft artifacts feed this same path. Build the graph from the completed draft
bundle, then run the vector/graph retrieval comparison before using graph context in scoring:

```bash
make build-index CORPUS=<draft-bundle>/corpus
make build-graph BUNDLE=<draft-bundle>
make compare-retrieval GOLDSET=<draft-bundle>/goldset.jsonl RAG_K=10
```

## Graph-Vector Fusion Evidence

CUDA-host evidence is under
`$DATA_DIR/graph-vector-fusion-retrieval/20260721T052842Z/`. The run built a matched hybrid vector
store (1,124 recursive chunks, multilingual E5 base, CUDA) and graph store (625 nodes, 213 edges)
from one accepted ontology bundle, then scored all 40 human-accepted questions at k=10.

| backend | recall@10 | MRR |
| --- | ---: | ---: |
| vector | 0.925 | 0.869 |
| graph/local_khop | 0.325 | 0.086 |
| graph/global_community | 0.350 | 0.245 |
| fused/local_khop, graph weight 0.3 | 0.925 | 0.864 |
| fused/global_community, graph weight 0.3 | 0.925 | 0.865 |

The two accepted comparative questions score recall 1.000 / MRR 1.000 for vector and both fused
rows; each graph-only row scores recall 0.500. The accepted ledger contains no multi-hop item, so
the report records that slice explicitly with `n=0`; it does not claim multi-hop quality evidence.
At graph weight 0.0, both fused rows exactly match vector recall 0.925 / MRR 0.869, while CI checks
the stronger per-query ranking equality and verifies that the graph lane is not called.

The evidence supports opt-in fusion, not a default change: it preserves recall on this corpus but
reduces MRR slightly. Reports are `comparison.json` and `comparison_graph_weight_0.json`; the
matched store config is `run_config.yaml` in the same artifact directory.

## Ontology Scope

Graph nodes use the closed 13-type vocabulary in
`docs/design/graph-ontology-schema.md`. The closed vocabulary matters because graph retrieval needs
stable typed nodes and relation caps; allowing a model to invent schema labels would make graph
quality and comparison unstable across runs.
