# Modules, Strategies, And CLI

Part of the [GraphRAG backend](../graphrag-backend.md) area of the
[current implementation index](../../current.md).

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
: Implements `FusedRetriever`. It queries the vector store (dense or hybrid) and `GraphStore` for
  `lane_depth(graph_fusion_candidates, k)` candidates each, maps both rankings onto one candidate
  set, fuses them with n-way weighted reciprocal-rank fusion, cuts to `k`, and preserves the
  surviving record's exact text and offsets. Fused metadata records which lanes returned each
  candidate, the graph weight, the span-identity policy, its merge threshold (folding policies
  only), and any folded spans. The fusion itself is the standalone `fuse_lane_hits`, so a
  weight/depth/identity/threshold sweep can reuse the production rule over cached lane rankings;
  `lane_agreement` counts the candidates both lanes vouch for.

`src/llb/rag/fusion_spans.py`
: The span-identity policies -- `exact` (identical `(doc_id, char_start, char_end)`) and `overlap`
  (fold a graph span into the vector chunk that contains it) -- plus the merge rule, its
  configurable threshold, its invariants, and the `LaneCandidates` view both lanes are ranked over.
  See
  [RAG core](../rag-core/graph-vector-fusion.md#fusion-span-identity-graph_fusion_span_identity).

`src/llb/rag/fusion_evidence/`
: The graph-weight sweep and its multi-hop verdict (see Graph-Vector Fusion Evidence below).
  `rows.py` builds the compared row set and caches each lane once per question, `sweep.py` scores
  every row per question-type slice, `stats.py` is the paired bootstrap plus exact sign test,
  `verdict.py` is the adopt-or-reject rule, and `report.py` renders the Markdown artifact.

`src/llb/rag/multihop_probe/`
: The per-hop retrievability probe that diagnoses a stuck `all-spans@k` (see
  [retrieval budget evidence](retrieval-budget-evidence.md#the-per-hop-probe-lane)). `probe.py`
  ranks each labeled span by the item's question and by its own text, `aggregate.py` builds the
  per-budget slice curves, and `diagnose.py` turns ranks into the per-item
  budget/query/unreachable classification. `prepared.py` reuses one query-prep result across every
  depth, `conversion.py` pairs outcomes by raw diagnosis, and `report.py` plus
  `conversion_report.py` render the raw and paired ASCII artifacts.

`src/llb/rag/fusion_calibration/`
: The held-out sidecar-free router calibration. It parses the deterministic threshold grid,
  evaluates routing error and paired retrieval deltas on tuning, freezes one policy before final
  retrieval, and renders the recommendation-gated JSON and Markdown artifacts.

`src/llb/eval/answer_quality/`
: The end-to-end companion to that sweep: it scores the SAME items under two retrieval lanes with
  the standard `run-eval` and compares the ANSWERS per question-type slice. `lanes.py` parses a
  sweep row label (`vector`, `fused/<strategy>@<weight>[/d<depth>][/i<identity>][/r<ratio>]`) back
  into retrieval knobs, `run.py` selects the item set once and drives one run bundle per lane,
  `coverage.py` recomputes the multi-span coverage columns from each bundle's `retrieval.jsonl`,
  `compare.py` is the pure per-slice comparison (reusing the fusion-evidence bootstrap),
  `verdict.py` decides answer-gain versus retrieval-only, and `report.py` renders the artifact.
  `budgets.py` adds the retrieval-BUDGET dimension: it expands the lane selection into
  `(lane x top_k)` cells labelled `<row>#k<budget>`, which `lanes.py` parses back, and names each
  raised cell's pairing against the same row at the smallest budget. `conversion.py` decides those
  pairings with the same `judge_lane`, adds the cost scan over the non-focus slices, and
  `report_budgets.py` renders the section. `coverage.py` also reports `context_chars`, the served
  context measured from the sidecar offsets, so a budget's coverage is always readable beside its
  price.

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
llb compare-answer-quality --from-comparison <sweep>/comparison.json --split final
llb compare-answer-quality --from-comparison <sweep>/comparison.json --budgets 10,50
llb probe-multihop-hops --budgets 10,25,50 --retrieval-backend faiss --out-dir <dir>
llb probe-multihop-hops --query-prep decompose --query-prep-model <model> --out-dir <dir>
```

`RunConfig` carries `retrieval_backend`, `retrieval_strategy`, `graph_khop_depth`, `graph_weight`
(default 0.3), and `graph_fusion_candidates` (default `None` == each lane asked for exactly `top_k`;
see [RAG core](../rag-core/graph-vector-fusion.md#fusion-candidate-depth-graph_fusion_candidates)).
These values are part of the config fingerprint and manifest. The sweep grid accepts
`graph_weight=...` and `graph_fusion_candidates=...` and selects the fused backend for either; the
Optuna space samples the graph weight when its base config is fused. `graph_weight=0.0` is an exact
vector passthrough and does not query the graph lane; `1.0` is an exact graph passthrough.

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
