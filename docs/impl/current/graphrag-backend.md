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
: Implements `FusedRetriever`. It queries the vector store (dense or hybrid) and `GraphStore` for
  `lane_depth(graph_fusion_candidates, k)` candidates each, fuses span ids with n-way weighted
  reciprocal-rank fusion, drops exact duplicate source spans, cuts to `k`, and preserves the
  selected record's exact text and offsets. Fused metadata records which lanes returned each span
  and the graph weight. The fusion itself is the standalone `fuse_lane_hits`, so a weight/depth
  sweep can reuse the production rule over cached lane rankings.

`src/llb/rag/fusion_evidence/`
: The graph-weight sweep and its multi-hop verdict (see Graph-Vector Fusion Evidence below).
  `rows.py` builds the compared row set and caches each lane once per question, `sweep.py` scores
  every row per question-type slice, `stats.py` is the paired bootstrap plus exact sign test,
  `verdict.py` is the adopt-or-reject rule, and `report.py` renders the Markdown artifact.

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

`RunConfig` carries `retrieval_backend`, `retrieval_strategy`, `graph_khop_depth`, `graph_weight`
(default 0.3), and `graph_fusion_candidates` (default `None` == each lane asked for exactly
`top_k`; see [RAG core](rag-core.md#fusion-candidate-depth-graph_fusion_candidates)). These values
are part of the config fingerprint and manifest. The sweep grid accepts `graph_weight=...` and
`graph_fusion_candidates=...` and selects the fused backend for either; the Optuna space samples
the graph weight when its base config is fused. `graph_weight=0.0` is an exact vector passthrough
and does not query the graph lane; `1.0` is an exact graph passthrough.

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

### The graph-weight sweep lane

`llb compare-graph-fusion` (`make compare-graph-fusion`) is the lane that decides a graph weight.
`compare-retrieval` ranks backends over a whole gold set at ONE weight; this lane sweeps the weight
and answers the narrower question a recommendation needs -- on items whose answer requires more
than one source span, does fusing graph evidence retrieve more of that evidence, at which weight,
and at what cost elsewhere.

Three things separate it from the flat comparison:

- **A multi-span metric.** `recall@k` credits an item as soon as ANY labeled span is retrieved,
  which a two-hop item satisfies by returning only one hop. The lane reports `recall@k` beside
  `all-spans@k` (every labeled span covered) and `span coverage` (the fraction covered), all from
  [RAG core](rag-core.md#retrieval-metrics).
- **Uncertainty.** A multi-hop slice is tens of items, so every cell carries a paired percentile
  bootstrap interval over shared resample index sets, plus the item-level win/loss/tie ledger and
  an exact two-sided sign test. The verdict gates on the INTERVAL: a positive mean whose interval
  still includes no difference is recorded as `inconclusive`, never as an adopt.
- **One retrieval pass per lane.** Neither lane's ranking depends on the weight or on the
  candidate depth, so the sweep retrieves each lane once at the DEEPEST compared pool and re-fuses
  those same candidates through the production `fuse_lane_hits` at every (weight, depth) point.

The lane sweeps both fusion knobs. `GRAPH_FUSION_CANDIDATES` (`--graph-fusion-candidates`) is the
per-lane candidate depth grid, where `k` names the scored cutoff itself; each fused row is labeled
`fused/<strategy>@<weight>/d<depth>`. Depths resolve against `k` and de-duplicate, endpoint weights
carry no depth variants (they are lane passthroughs), and the verdict ranks across weights and
depths together, preferring the shallower pool on a tie.

```bash
make compare-graph-fusion CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> \
  GRAPH_WEIGHTS=0,0.1,0.2,0.3,0.5,0.7,1.0 GRAPH_FUSION_CANDIDATES=k,50
llb compare-graph-fusion --config <cfg> --k 10 --graph-weights 0,0.3,1.0 \
  --graph-fusion-candidates k,50 --focus-slice multi-hop --out-dir <dir>
```

Artifacts per run: `report.md` (verdict, focus slice, overall, per-type slices, item ledger),
`comparison.json`, and `run_config.json`.

### Accepted-ledger evidence, single graph weight

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
rows; each graph-only row scores recall 0.500. That accepted ledger contains no multi-hop item, so
the report records that slice explicitly with `n=0`; it does not claim multi-hop quality evidence.
At graph weight 0.0, both fused rows exactly match vector recall 0.925 / MRR 0.869, while CI checks
the stronger per-query ranking equality and verifies that the graph lane is not called.

That evidence supports opt-in fusion, not a default change: it preserves recall on this corpus but
reduces MRR slightly. Reports are `comparison.json` and `comparison_graph_weight_0.json`; the
matched store config is `run_config.yaml` in the same artifact directory.

### Multi-hop slice evidence, swept graph weight

CUDA-host evidence is under `$DATA_DIR/graph-vector-fusion-multihop/20260722T100231Z/`; the scored
draft bundle is the sibling `goods-draft/`. A five-document, 1.15 MB converted Ukrainian goods-PDF
corpus was drafted with `MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M` over Ollama at a 16,384-token
context (62 extraction windows, 255 entities, 242 grounded facts), yielding 95 items: 60 flat plus
**35 multi-hop items, every one carrying exactly two grounded spans, 17 of them citing two
different documents**. `validate-goldset` passes. The matched stores are a hybrid recursive vector
store (1,139 chunks, multilingual E5 base) and a graph store of 423 nodes, 242 edges, and 211
communities. All 95 items were scored at k=10 over a 7-point weight grid with 2,000 bootstrap
resamples (seed 13).

Multi-hop slice (n=35), 95% bootstrap CI, paired against the vector row:

| row | recall@10 | all-spans@10 | span coverage | MRR |
| --- | ---: | ---: | ---: | ---: |
| vector | 0.686 [0.543, 0.829] | 0.057 [0.000, 0.143] | 0.371 | 0.360 |
| graph/local_khop | 0.514 [0.371, 0.686] | 0.086 [0.000, 0.200] | 0.300 | 0.164 |
| graph/global_community | 0.543 [0.371, 0.714] | 0.057 [0.000, 0.143] | 0.300 | 0.397 |
| fused/local_khop @0.30 | 0.714 [0.571, 0.857] | 0.029 [0.000, 0.086] | 0.371 | 0.347 |
| fused/global_community @0.10 | 0.771 [0.629, 0.914] | 0.086 [0.000, 0.200] | 0.429 | 0.369 |
| fused/global_community @0.30 | 0.771 [0.629, 0.914] | 0.057 [0.000, 0.143] | 0.414 | 0.384 |

Overall (n=95):

| row | recall@10 | all-spans@10 | span coverage | MRR | recall delta vs vector |
| --- | ---: | ---: | ---: | ---: | ---: |
| vector | 0.705 [0.611, 0.789] | 0.474 | 0.589 | 0.421 | 0.000 |
| graph/local_khop | 0.368 [0.274, 0.463] | 0.211 | 0.289 | 0.121 | -0.337 [-0.463, -0.200] |
| graph/global_community | 0.326 [0.232, 0.421] | 0.147 | 0.237 | 0.221 | -0.379 [-0.505, -0.242] |
| fused/local_khop @0.30 | 0.705 [0.611, 0.789] | 0.453 | 0.579 | 0.411 | 0.000 [-0.053, +0.053] |
| fused/global_community @0.10 | 0.747 [0.663, 0.832] | 0.495 | 0.621 | 0.425 | +0.042 [+0.000, +0.095] |
| fused/global_community @0.20 | 0.758 [0.674, 0.842] | 0.495 | 0.626 | 0.430 | +0.053 [+0.000, +0.105] |

What the run establishes:

- **`recall@10` hides the multi-hop problem entirely.** The vector lane looks acceptable on the
  multi-hop slice at recall 0.686, but its `all-spans@10` is 0.057: it retrieves BOTH hops for 2 of
  35 two-hop questions. No row in the sweep exceeds 0.086 (3 of 35). At k=10 multi-hop evidence
  coverage is essentially unsolved on this corpus by every backend, fused or not -- which is the
  measurement the flat comparison could not produce.
- **The best fused row is `global_community` at a LOW graph weight**, not the 0.3 default and not
  `local_khop`. It gains multi-hop recall +0.086 [0.000, 0.200] (3 wins, 0 losses, 32 ties, sign
  test p=0.250) and overall recall +0.042 to +0.053, so it does not trade factoid ranking away.
  Every one of those intervals touches zero.
- **The verdict is therefore `inconclusive`, not `adopt`.** The direction is consistently positive
  and never negative, but 35 items cannot separate it from the vector lane. Fusion stays opt-in and
  the default weight is unchanged.
- **Graph-only retrieval loses decisively overall** (-0.337 and -0.379 recall, sign test p=0.000),
  reproducing the accepted-ledger run's ordering on a second, multi-document corpus.
- **Graph weight 0.0 is an exact vector passthrough**: 0 wins, 0 losses, 95 ties on every metric.

### Candidate depth evidence

CUDA-host evidence is under `$DATA_DIR/graph-vector-fusion-multihop/20260722T102219Z-depth/`. The
same 95 items, matched stores, weight grid, and seed as the sweep above were re-scored at TWO
per-lane candidate depths -- `k` (10, the historical pool) and 50 -- through
`GRAPH_FUSION_CANDIDATES=k,50`. Fused rows are labeled `fused/<strategy>@<weight>/d<depth>`, so a
depth sweep and a weight sweep are one table.

Two reproduction checks passed before the comparison was read: every `d10` row equals the prior
run's fused row on every metric, interval, and item-level outcome (17 of 17 rows, exact), and the
lanes really do deepen (the vector lane returns 50 of 50 candidates; `local_khop` averages 32.1 and
`global_community` 45.1 at depth 50).

**Result: depth 50 is byte-identical to depth 10 on every row, at every weight, for both graph
strategies.** Not "no significant gain" -- no difference at all: re-fusing the same lanes at depth
50 changes the fused top-10 for **0 of 93 questions** at each of `graph_weight` 0.1 / 0.2 / 0.3 /
0.5 / 0.7.

The mechanism is measured, not incidental. Under undamped RRF a single-lane candidate below rank k
can never enter the top-k (see
[RAG core](rag-core.md#fusion-candidate-depth-graph_fusion_candidates) for the argument), so only
spans BOTH lanes return can be promoted by a deeper pool. Across all 93 questions the two lanes
share an exact `(doc_id, char_start, char_end)` span **twice at depth 50** (once at depth 10) per
strategy -- graph evidence spans are entity mentions and edge evidence, whose boundaries essentially
never coincide with an 800-character recursive chunk.

Verdict: **reject as a default**. `graph_fusion_candidates` stays `None` (each lane asked for
exactly `top_k`), and the operator's answer is that the fused rows are limited by the graph
WEIGHT, not by candidate depth -- on this corpus depth is not a knob at all. The knob ships opt-in
because it becomes live for any corpus whose lanes share exact spans, and because it is the
prerequisite half of the span-identity work tracked in [`plan.md`](../plan.md)
(`fusion-span-overlap-identity`): once fusion keys candidates by OVERLAP instead of exact
boundaries, cross-lane agreement stops being a 2-in-93 event and depth starts to matter.

Boundary: the 35 multi-hop items are DRAFTED, not human-accepted. They are span-exact, Ukrainian
gated, and each names its bridge or end entity in the reference answer, but only a reviewer can
confirm that a drafted two-hop question truly needs both cited facts. The stratified 95-row
worksheet is already drawn at `goods-draft/verify_sample.csv`, so the gate is
`make verify-review VERIFY_WS=<that file>` followed by `make verify-accept`; accepting the ledger
and re-running the sweep is tracked as forward work in [`plan.md`](../plan.md)
(`multihop-ledger-human-acceptance`).

## Ontology Scope

Graph nodes use the closed 13-type vocabulary in
`docs/design/graph-ontology-schema.md`. The closed vocabulary matters because graph retrieval needs
stable typed nodes and relation caps; allowing a model to invent schema labels would make graph
quality and comparison unstable across runs.
