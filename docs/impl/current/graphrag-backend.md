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
  [RAG core](rag-core.md#fusion-span-identity-graph_fusion_span_identity).

`src/llb/rag/fusion_evidence/`
: The graph-weight sweep and its multi-hop verdict (see Graph-Vector Fusion Evidence below).
  `rows.py` builds the compared row set and caches each lane once per question, `sweep.py` scores
  every row per question-type slice, `stats.py` is the paired bootstrap plus exact sign test,
  `verdict.py` is the adopt-or-reject rule, and `report.py` renders the Markdown artifact.

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

The lane sweeps all three fusion knobs and adds question-type-routed rows.
`GRAPH_FUSION_CANDIDATES` (`--graph-fusion-candidates`) is
the per-lane candidate depth grid, where `k` names the scored cutoff itself;
`GRAPH_FUSION_SPAN_IDENTITY` (`--graph-fusion-span-identity`) is the span-identity grid (`exact`
and/or `overlap`); `GRAPH_FUSION_SPAN_MERGE_RATIO` (`--graph-fusion-span-merge-ratio`) is the merge
threshold that identity policy folds by. Each fused row is labeled
`fused/<strategy>@<weight>/d<depth>`, with `/i<identity>` and then `/r<ratio>` appended for a
non-default policy or threshold -- so an `exact` row keeps the exact label, and therefore the exact
comparability, it had before either knob existed. Depths resolve against `k` and de-duplicate,
endpoint weights carry no depth, identity, or threshold variants (they are lane passthroughs,
nothing is fused), the threshold grid expands only the folding identity policies (`exact` has no
partial overlap to threshold), and the verdict ranks across all four knobs together, preferring the
shallower pool, the default policy, and the default threshold on a tie.

`ROUTED_GRAPH_WEIGHT` (`--routed-graph-weight`, default 0.3) also emits
`routed/<strategy>@<weight>/d<depth>[/i<identity>]`. Its weight is applied only to questions the
router calls multi-span; all other questions use the exact vector endpoint. The report records
graph/vector and sidecar/heuristic decision counts, including a breakdown by question-type slice.
`--no-routing-sidecar` masks the question-type map so routed rows use only deterministic text
signals; the two `--heuristic-*` options select their fixed thresholds. The corresponding Make
variables are `FUSION_HIDE_ROUTING_SIDECAR`, `FUSION_HEURISTIC_LONG_QUESTION_WORDS`, and
`FUSION_HEURISTIC_MIN_LINKED_ENTITIES`.

```bash
make compare-graph-fusion CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> \
  GRAPH_WEIGHTS=0,0.1,0.2,0.3,0.5,0.7,1.0 GRAPH_FUSION_CANDIDATES=k,50 \
  GRAPH_FUSION_SPAN_IDENTITY=exact,overlap GRAPH_FUSION_SPAN_MERGE_RATIO=0.25,0.5,0.75,1.0 \
  ROUTED_GRAPH_WEIGHT=0.3
llb compare-graph-fusion --config <cfg> --k 10 --graph-weights 0,0.3,1.0 \
  --graph-fusion-candidates k,50 --graph-fusion-span-identity exact,overlap \
  --focus-slice multi-hop --out-dir <dir>
```

Every fused row also reports its **cross-lane agreement**: how many questions produced a candidate
BOTH lanes returned, and how many such candidates per question. That is the number a span-identity
policy is read against, and the precondition for candidate depth to matter at all.

Artifacts per run: `report.md` (verdict, focus slice, overall, per-type slices, agreement table,
item ledger), `comparison.json`, and `run_config.json`.

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

Verdict under the `exact` identity rule: **reject as a default**. `graph_fusion_candidates` stays
`None` (each lane asked for exactly `top_k`), and the operator's answer is that those fused rows
are limited by the graph WEIGHT, not by candidate depth. The knob ships opt-in because it becomes
live as soon as the lanes agree -- which is exactly what the span-identity policy changes: under
`overlap` the same depth grid moves every fused row (see
[span-identity evidence](#span-identity-evidence)).

Boundary: the 35 multi-hop items are DRAFTED, not human-accepted. They are span-exact, Ukrainian
gated, and each names its bridge or end entity in the reference answer, but only a reviewer can
confirm that a drafted two-hop question truly needs both cited facts. The stratified 95-row
worksheet is already drawn at `goods-draft/verify_sample.csv`, so the gate is
`make verify-review VERIFY_WS=<that file>` followed by `make verify-accept`; accepting the ledger
and re-running the sweep is tracked as forward work in [`plan.md`](../plan.md)
(`multihop-ledger-human-acceptance`).

### Span-identity evidence

CUDA-host evidence is under
`$DATA_DIR/graph-vector-fusion-multihop/20260722T145615Z-span-identity/`. The same 95 drafted items
(35 multi-hop), matched stores, weight grid, depth grid, and seed as the two runs above were
re-scored under BOTH span-identity policies -- 47 rows in one table -- with
`GRAPH_FUSION_SPAN_IDENTITY=exact,overlap GRAPH_FUSION_CANDIDATES=k,50`.

The reproduction check passed before the comparison was read: all 27 `exact` rows equal the prior
depth run on every metric, interval, win/loss/tie ledger, and per-item focus outcome, and every
fused chunk is still a verbatim corpus slice at its own offsets.

**Cross-lane agreement, the number the policy exists to move** (questions out of 95 whose fused
pool contains a candidate BOTH lanes returned):

| policy | depth | local_khop | global_community |
| --- | ---: | ---: | ---: |
| exact | 10 | 1 (0.011 per question) | 1 (0.011) |
| exact | 50 | 2 (0.021) | 2 (0.021) |
| overlap | 10 | 53 (0.842) | 47 (0.716) |
| overlap | 50 | 87 (3.853) | 93 (4.095) |

Multi-hop slice (n=35), 95% bootstrap CI, paired against the vector row:

| row | recall@10 | all-spans@10 | span coverage | MRR | recall delta vs vector |
| --- | ---: | ---: | ---: | ---: | ---: |
| vector | 0.686 [0.543, 0.829] | 0.057 | 0.371 | 0.360 | 0.000 |
| fused/global_community@0.10/d10 (exact) | 0.771 [0.629, 0.914] | 0.086 | 0.429 | 0.369 | +0.086 [+0.000, +0.200] |
| fused/global_community@0.30/d50 (exact) | 0.771 [0.629, 0.914] | 0.057 | 0.414 | 0.384 | +0.086 [+0.000, +0.200] |
| fused/global_community@0.30/d10/ioverlap | 0.800 [0.657, 0.914] | 0.086 | 0.443 | 0.399 | +0.114 [+0.029, +0.229] |
| fused/global_community@0.30/d50/ioverlap | 0.800 [0.657, 0.914] | 0.086 | 0.443 | 0.403 | +0.114 [+0.029, +0.229] |
| fused/local_khop@0.30/d10/ioverlap | 0.714 [0.571, 0.857] | 0.057 | 0.386 | 0.352 | +0.029 [+0.000, +0.086] |
| fused/local_khop@0.30/d50/ioverlap | 0.743 [0.600, 0.886] | 0.057 | 0.400 | 0.356 | +0.057 [+0.000, +0.143] |

Overall (n=95):

| row | recall@10 | all-spans@10 | span coverage | MRR | recall delta vs vector |
| --- | ---: | ---: | ---: | ---: | ---: |
| vector | 0.705 [0.611, 0.789] | 0.474 | 0.589 | 0.421 | 0.000 |
| fused/global_community@0.10/d10 (exact) | 0.747 [0.663, 0.832] | 0.495 | 0.621 | 0.425 | +0.042 [+0.000, +0.095] |
| fused/global_community@0.30/d50/ioverlap | 0.768 [0.684, 0.842] | 0.505 | 0.637 | 0.447 | +0.063 [-0.011, +0.137] |
| fused/local_khop@0.30/d50/ioverlap | 0.768 [0.684, 0.853] | 0.516 | 0.642 | 0.430 | +0.063 [+0.000, +0.126] |
| fused/local_khop@0.50/d50/ioverlap | 0.779 [0.695, 0.853] | 0.537 | 0.658 | 0.417 | +0.074 [+0.000, +0.158] |

What the run establishes:

- **Exact identity made fusion structurally inert on this corpus.** One shared candidate in 95
  questions at depth 10 is not fusion; it is two disjoint rankings trading result seats. Containment
  is the common case the exact rule could not see: `overlap` finds a shared candidate for 47-53 of
  95 questions at depth 10 and 87-93 at depth 50.
- **The multi-hop gain becomes separable from zero.** The best `exact` row gains +0.086
  [+0.000, +0.200] multi-hop recall (interval touches zero, `inconclusive`); the best `overlap` row,
  `fused/global_community@0.30/d50/ioverlap`, gains +0.114 [+0.029, +0.229] with 4 wins, 0 losses,
  31 ties, and does not pay for it overall (+0.063 [-0.011, +0.137]). The lane's verdict on this
  ledger is therefore **adopt**.
- **Candidate depth is now a live knob, and only because of the identity rule.** Under `exact`,
  all 10 (strategy, weight) pairs are byte-identical at depth 10 and 50; under `overlap`, all 10
  differ. Depth 50 is what turns `local_khop@0.30` from +0.029 to +0.057 multi-hop recall and from
  +0.011 to +0.063 overall.
- **`all-spans@10` still does not move.** The best row carries BOTH hops for 3 of 35 two-hop
  questions (0.086), the same ceiling every earlier row hit. Fusion improves WHICH single hop is
  retrieved and how the pool ranks; it does not solve two-hop coverage at k=10.
- **The weight optimum shifted with the policy.** Under `exact` the best row was `global_community`
  at weight 0.10; under `overlap` it is weight 0.30 -- unsurprising once a graph vote reinforces a
  chunk instead of displacing it, since a graph candidate no longer costs a result seat.

Verdict: `graph_fusion_span_identity` ships **opt-in with `exact` as the default**, despite the
adopt. The evidence is measured on the DRAFTED multi-hop ledger (see the boundary above), and the
project's standing rule is that a drafted slice does not move a default. `overlap` with
`graph_fusion_candidates=50` is the setting to enable when multi-hop retrieval coverage is the
goal -- but the end-to-end run below measures an answer-side cost on the factoid slice, so it is
not a free upgrade (see
[the overlap answer-quality result](#measured-result-the-overlap-span-identity-carries-more-evidence-and-costs-factoid-answers)).
Flipping the shipped default is gated on the accepted-ledger re-run tracked in
[`plan.md`](../plan.md) (`multihop-ledger-human-acceptance`).

### Span merge-threshold evidence

CUDA-host evidence is under
`$DATA_DIR/graph-vector-fusion-multihop/20260722T194026Z-span-merge-ratio/`. The same 95 drafted
items (35 multi-hop), matched stores, weight grid, depth grid, identity grid, and seed as the
span-identity run were re-scored across a four-point merge-threshold grid --
`GRAPH_FUSION_SPAN_MERGE_RATIO=0.25,0.5,0.75,1.0`, where `1.0` is containment-only -- for 127 rows
in one table. The threshold is a parameter of a FOLDING policy, so the grid expands `overlap` rows
only; `exact` has no partial overlap to threshold.

The reproduction check passed before the comparison was read: all 47 rows of the span-identity run
are byte-identical here, including the verdict row and its item ledger.

**Result: on this corpus the threshold moves no multi-hop metric at any setting.** Across the 24
`overlap` row families (fixed and routed, both strategies, both depths, every interior weight):

| threshold | row families whose metrics differ from 0.5 | max multi-hop change | max overall change |
| --- | ---: | ---: | ---: |
| 0.25 | 0 of 24 | 0.0000 | 0.0000 |
| 0.75 | 0 of 24 | 0.0000 | 0.0000 |
| 1.0 (containment only) | 16 of 24 | 0.0000 | 0.0105 (1 of 95 questions) |

Cross-lane agreement barely moves either: `global_community` reports the identical 47/95 (depth 10)
and 93/95 (depth 50) questions with a shared candidate at all four thresholds; `local_khop` moves
54 / 53 / 53 / 52 at depth 10 and 87 at every threshold at depth 50, and its mean shared candidates
per question falls only from 3.853 to 3.832 between the loosest and strictest setting. The
sweep-winning row is the default-threshold `fused/global_community@0.30/d50/ioverlap` at
+0.114 [+0.029, +0.229] multi-hop recall -- unchanged, since the threshold does not reach it.

The mechanism is measured, not incidental (`span_overlap_histogram.py` beside the run artifacts
re-derives it). For every graph evidence span in the depth-50 pool, bucketed by its strongest
overlap with any retrieved vector chunk:

| strategy | graph spans | no overlap | contained (1.0) | [0.75, 1.0) | below 0.75 |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_khop | 3,010 | 2,448 | 557 | 5 | 0 |
| global_community | 4,291 | 3,683 | 601 | 7 | 0 |

A graph span either misses the retrieved chunks entirely or sits **wholly inside** one: 99.1% and
98.9% of the spans that overlap at all are fully contained, and **not one span in the corpus lands
below 0.75 while still overlapping a chunk**. The threshold therefore has nothing to decide between
0.25, 0.5, and 0.75, and containment-only differs on the dozen partially-covered spans alone. An
~800-character recursive chunk with a 120-character overlap is two orders of magnitude longer than
an entity mention, and a mention landing in the shared tail is contained in BOTH neighbours -- so
straddling a boundary needs a mention to sit exactly on a cut, which happens ~0.2% of the time.

Verdict: **pin `graph_fusion_span_merge_ratio=0.5`**. The value is exposed (see
[RAG core](rag-core.md#fusion-span-identity-graph_fusion_span_identity)) because the sweep needed
it and a corpus with shorter chunks or longer graph spans could put mass in the buckets this one
leaves empty, but on Ukrainian goods PDFs at `chunk_size=800` it is not a tuning surface: the
operator should not spend a sweep on it. The one directional signal is that containment-only never
helps -- where it moves overall recall at all it loses a question -- so 0.5 is not merely
arbitrary among the insensitive settings. Re-run the histogram probe before trusting the pin on a
corpus with materially different chunking.

### Answer-quality evidence

The sweep above is model-independent: it measures what the context CARRIES, never what the model
does with it. `llb compare-answer-quality` (`make compare-answer-quality`) closes that gap. It
scores the identical item set END TO END under two retrieval lanes with the standard `run-eval`,
then compares the ANSWERS per question-type slice with the same paired bootstrap the sweep uses.

Three properties make the comparison readable:

- **The lanes are named by sweep row label.** `vector`, `graph/<strategy>`,
  `fused/<strategy>@<weight>[/d<depth>][/i<identity>][/r<ratio>]`, and
  `routed/<strategy>@<weight>[/d<depth>][/i<identity>][/r<ratio>]` parse back into retrieval knobs,
  and `--from-comparison <sweep>/comparison.json` reads the baseline plus the row that sweep's
  verdict named best -- so
  the scored lane is the row the retrieval sweep actually recommended, not a retyped approximation.
- **One shared item set, ordinary bundles.** The selection happens once and every lane is a plain
  `run-eval` bundle under `$DATA_DIR/run-eval/`, so any lane's number is reproducible with a bare
  `run-eval` and the per-item pairing is legitimate. Lanes that scored different item sets are a
  hard error, never a silent intersection. A comma-separated `--split` scores one bundle per split
  and pools them into one compared set.
- **Multi-span coverage beside the objective.** `retrieval_hit` in a score row is `recall@k`, which
  a two-hop item satisfies with one hop, so the lane recomputes `span_coverage` and `all_spans_at_k`
  from each bundle's `retrieval.jsonl` and reports them next to the objective. The verdict states a
  coverage claim on `span_coverage` (graded) rather than the `all_spans_at_k` gate, because on a
  hard multi-hop slice the gate can be near-zero for every lane and therefore blind to a lane that
  nonetheless carried more evidence.

The verdict is one of `answer_quality_gain` (the objective delta's interval clears zero),
`retrieval_only` (the coverage delta's interval clears zero while the objective's does not),
`inconclusive`, or `no_gain`. `retrieval_only` is checked BEFORE `inconclusive` on purpose: a
measured coverage gain paired with a noisy objective is a result about retrieval, and reporting it
as merely inconclusive would drop the half that was measured.

More than two lanes are allowed, and then EVERY candidate keeps its own decision in the verdict's
`lane_decisions` (rendered as a "Per-lane decisions" list): the headline verdict names only the
strongest candidate, so a three-lane comparison that collapsed to one sentence would silently drop
the others.

```bash
make compare-answer-quality CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> \
  FUSION_COMPARISON=<sweep-dir>/comparison.json SPLIT=final,tuning,calibration INCLUDE_DRAFTED=1
llb compare-answer-quality --config <cfg> --lanes vector,fused/global_community@0.10 --split final
```

Artifacts per run: `report.md` and `comparison.json` under
`$DATA_DIR/graph-vector-fusion-multihop/<run>/answer-quality/`.

#### Measured result: the multi-hop coverage gain does not reach the answer

CUDA-host evidence is under
`$DATA_DIR/graph-vector-fusion-multihop/20260722T133033Z-answer-quality/answer-quality/`. The same
95-item drafted goods ledger, matched stores, and k=10 as the sweep above were scored end to end by
`MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M` over Ollama under two lanes -- `vector` and the sweep's
best row `fused/global_community@0.10/d10` -- across all three splits (one run bundle per lane and
split, pooled), 2,000 bootstrap resamples, seed 13.

Multi-hop slice (n=35), fused minus vector, 95% paired bootstrap CI:

| metric | delta | interval | w/l/t | sign p |
| --- | ---: | ---: | :-: | ---: |
| objective | -0.005 | [-0.071, +0.072] | 9/8/18 | 1.000 |
| span coverage | **+0.057** | **[+0.014, +0.114]** | 4/0/31 | 0.125 |
| recall@10 | +0.086 | [+0.000, +0.200] | 3/0/32 | 0.250 |
| all-spans@10 | +0.029 | [+0.000, +0.086] | 1/0/34 | 1.000 |

Verdict: **`retrieval_only`**. The fused lane carries measurably more of the multi-hop evidence --
span coverage 0.429 versus 0.371, the only interval in the table that clears zero -- and the model
turns none of it into better answers: the objective is 0.321 versus 0.326, an interval straddling
zero with 9 wins against 8 losses. Paying for a graph build buys multi-hop RETRIEVAL on this
corpus, not multi-hop ANSWERS.

Two things corroborate the measurement:

- **The retrieval columns reproduce the sweep exactly.** Scored through `run-eval` rather than
  through the sweep's replay wrappers, the fused lane still reports multi-hop recall 0.771 vs
  0.686, all-spans 0.086 vs 0.057, and span coverage 0.429 vs 0.371 -- every figure identical to
  the swept row. The two lanes are measuring the same retrieval through independent code paths.
- **Overall answer quality is flat to slightly negative**: objective -0.027 [-0.062, +0.009]
  (15 wins, 22 losses), with the `procedural` slice at -0.021 [-0.050, -0.001] on n=14 -- the only
  answer-side interval anywhere in the run that excludes zero, and it points DOWN. Extra graph
  candidates displace vector chunks the model was using, and on a 12B model that costs a little
  more than the multi-hop coverage gains back.

Boundaries, both recorded in the artifact rather than inferred:

- **The ledger is drafted.** No reviewer has accepted these 95 items, so the objective is
  diagnostic, not a leaderboard result. Scoring them at all required `--include-drafted`, and every
  bundle manifest carries `config.item_grounding: drafted` (see
  [RAG core](rag-core.md#executor)). Re-running on the accepted ledger is tracked in
  [`plan.md`](../plan.md) (`multihop-ledger-human-acceptance`).
- **The answer-side metric cannot see hops.** `objective_score` is reference-answer token F1, so an
  answer stating one fact fluently and omitting the other scores about the same as a vague answer
  touching both. The retrieval side distinguishes partial from complete evidence; the answer side
  does not, which bounds how sharply this verdict can be read. Building the answer-side counterpart
  is tracked in [`plan.md`](../plan.md) (`answer-side-span-coverage-metric`), and repeating the
  comparison on a second model -- since "did the model use the extra hop" is a model property -- is
  tracked as `fusion-answer-quality-second-model`.

#### Measured result: the overlap span identity carries more evidence and costs factoid answers

CUDA-host evidence is under
`$DATA_DIR/graph-vector-fusion-multihop/20260722T151635Z-overlap-answer-quality/answer-quality/`.
THREE lanes were scored end to end over the identical 95-item drafted ledger, same model, splits,
bootstrap, and seed as the two-lane run above: `vector`, the best `exact` row
(`fused/global_community@0.10/d10`), and the best `overlap` row
(`fused/global_community@0.30/d50/ioverlap`) named by the span-identity sweep's verdict.

Both shared lanes reproduced the earlier run EXACTLY -- every metric mean of `vector` and of the
`exact` row is identical to the two-lane comparison, overall and on the multi-hop slice. Generation
is deterministic for these grounded lanes, so the three-lane table and the earlier two-lane table
are one measurement, not two.

Multi-hop slice (n=35), lane minus vector, 95% paired bootstrap CI:

| lane | objective | span coverage | recall@10 | all-spans@10 |
| --- | ---: | ---: | ---: | ---: |
| exact `@0.10/d10` | -0.005 [-0.071, +0.072] | **+0.057 [+0.014, +0.114]** | +0.086 [+0.000, +0.200] | +0.029 [+0.000, +0.086] |
| overlap `@0.30/d50` | -0.000 [-0.075, +0.078] | **+0.071 [+0.014, +0.129]** | +0.114 [+0.029, +0.229] | +0.029 [+0.000, +0.086] |

Verdict: **`retrieval_only` for BOTH lanes.** The overlap row carries the most multi-hop evidence
of any lane measured (span coverage 0.443 versus the vector lane's 0.371, 5 wins and 0 losses) and
converts none of it into better answers: its multi-hop objective is 0.326 against the vector lane's
0.326 -- a delta of -0.000 with 12 wins against 7 losses, pure churn.

The new finding is on the other side of the ledger:

- **The overlap lane measurably HURTS factoid answers.** On the 40 factoid items the objective
  falls -0.053 [-0.111, -0.001] (4 wins, 13 losses, sign test p=0.049) -- the only interval in the
  run that clears zero, and it points down. The `exact` row costs less and does not clear zero
  (-0.040 [-0.096, +0.005]). Factoid retrieval itself is flat (span coverage -0.025 [-0.100,
  +0.050]), so this is the CONTEXT changing under a single-span question, not evidence being lost:
  a stronger graph vote re-ranks the chunk the model was already answering from.
- **Overall answer quality stays slightly negative for both**: -0.027 [-0.062, +0.009] for `exact`
  and -0.029 [-0.067, +0.008] for `overlap`, both straddling zero.
- **The retrieval columns reproduce the sweep exactly through a second code path.** Scored through
  `run-eval` rather than the sweep's replay wrappers, the overlap lane reports multi-hop recall
  0.800, all-spans 0.086, and span coverage 0.443 -- every figure identical to the swept row.

What this means for the recommendation: on this corpus and this model, `graph_fusion_span_identity=
overlap` buys strictly more multi-hop RETRIEVAL than `exact` and pays for it with a measured
factoid ANSWER cost, so the fixed-weight row stays opt-in. The question-type route below removes
that cost by never fusing a factoid. Whether a different model uses the extra hop is tracked in
[`plan.md`](../plan.md) (`fusion-answer-quality-second-model`). The ledger is DRAFTED and the
answer-side metric still cannot see hops; both boundaries above apply unchanged.

#### Measured result: question-type routing keeps the gain and clears the factoid loss

CUDA-host evidence is under
`$DATA_DIR/graph-vector-fusion-multihop/20260722T160531Z-question-routing/`; answer comparison is
in its `answer-quality/` child. The same 95-item drafted goods ledger, stores, k=10, split pool,
2,000 bootstrap resamples, seed 13, and 12B MamayLM model as the fixed overlap result were used.
The sweep reports the routed rows beside the complete fixed grid; the end-to-end comparison scores
`vector` against `routed/global_community@0.30/d50/ioverlap`.

All 95 questions had sidecar labels. The router sent 37 to graph fusion -- all 35 `multi-hop` and
both `comparative` items -- and sent 58 to vector -- all 40 `factoid`, 4 `numeric`, and 14
`procedural` items. No heuristic decision contributes to this measurement.

Multi-hop slice (n=35), routed minus vector, 95% paired bootstrap CI:

| metric | delta | interval | w/l/t | sign p |
| --- | ---: | ---: | :-: | ---: |
| objective | -0.000 | [-0.075, +0.078] | 12/7/16 | 0.359 |
| span coverage | **+0.071** | **[+0.014, +0.129]** | 5/0/30 | 0.062 |
| recall@10 | **+0.114** | **[+0.029, +0.229]** | 4/0/31 | 0.125 |
| all-spans@10 | +0.029 | [+0.000, +0.086] | 1/0/34 | 1.000 |

The routed row is retrieval-identical to the best fixed overlap row on every multi-hop metric:
recall 0.800, all-spans 0.086, and span coverage 0.443. It therefore keeps the full measured
multi-hop gain, including both intervals that clear zero. The answer verdict remains
**`retrieval_only`** because objective 0.326 is unchanged from vector despite the extra evidence.

The safety result is exact on the slice that motivated routing:

- **All 40 factoid retrieval and answer rows are vector ties.** Objective delta is 0.000
  [0.000, 0.000], 0/0/40 wins/losses/ties; recall, all-spans, and span coverage are also exact
  40-item ties. The fixed overlap row's -0.053 [-0.111, -0.001] factoid answer loss is absent.
- **Overall retrieval improves while answer quality stays flat.** Recall rises +0.063
  [+0.021, +0.116] and span coverage +0.047 [+0.016, +0.089]. Objective is -0.001
  [-0.029, +0.027], compared with -0.029 [-0.067, +0.008] for the fixed overlap row.
- **The exact endpoint is production behavior.** Each routed lane manifest records
  `graph_fusion_router: question_type`; CI verifies that a vector route does not query the graph
  lane, while the live factoid results reproduce vector generation exactly.

Recommendation: use the routed overlap row when the bundle has the documented question-type
sidecar and multi-hop coverage is the goal. Keep `fixed` as the shipped default: the evidence is
drafted and the multi-hop answer gain is still absent. The sidecar-free policy has a separate
held-out result below and does not support changing its defaults.

### Sidecar-free heuristic calibration

CUDA-host evidence is under
`$DATA_DIR/graph-vector-fusion-multihop/20260722T180211Z-routing-calibration/`. The run used the
same drafted goods ledger and matched stores, `global_community@0.30/d50/ioverlap`, k=10,
multilingual E5 on the RTX 4060 Ti, 2,000 bootstrap resamples, and seed 13. Question-type labels
were hidden from every routing decision; the evaluation truth was only whether an item carried
more than one gold span.

`make calibrate-fusion-routing` swept word thresholds 10/12/14/16/18/20 crossed with linked-entity
thresholds 0/1/2 on the 31-item tuning split. It froze `w12/e0` before constructing the final-split
retrieval caches, then evaluated that one policy on the untouched 31-item final split.

| split | tp/fp/tn/fn | precision | recall | multi-span coverage delta | single-span recall delta |
| --- | :-: | ---: | ---: | ---: | ---: |
| tuning | 9/7/14/1 | 0.562 [0.333, 0.800] | 0.900 [0.667, 1.000] | +0.050 [0.000, +0.150] | +0.048 [0.000, +0.143] |
| final | 8/6/14/3 | 0.571 [0.308, 0.833] | 0.727 [0.444, 1.000] | +0.091 [0.000, +0.227] | +0.000 [-0.150, +0.150] |

Verdict: **no recommendation**. The frozen policy's tuning coverage interval touches zero, so it
does not pass the predeclared positive-gain gate. Final points in the same positive direction but
does not repair a failed tuning gate, and its single-span interval includes regression. The
production fallback therefore stays at 16 words plus 2 linked entities.

The standard sweep path independently reproduced the frozen policy with
`FUSION_HIDE_ROUTING_SIDECAR=1` on each split. The routed row gained multi-hop recall +0.100
[0.000, +0.300] on tuning and +0.182 [0.000, +0.455] on final, while all-spans@10 was unchanged;
both sweep verdicts were `inconclusive`. Their reports are the calibration artifact's
`tuning-compare/` and `final-compare/` children. The remaining limitation is statistical power,
not an invitation to select on final; a larger accepted-ledger repeat is forward work in
[`plan.md`](../plan.md) (`fusion-routing-calibration-power`).

## Ontology Scope

Graph nodes use the closed 13-type vocabulary in
`docs/design/graph-ontology-schema.md`. The closed vocabulary matters because graph retrieval needs
stable typed nodes and relation caps; allowing a model to invent schema labels would make graph
quality and comparison unstable across runs.
