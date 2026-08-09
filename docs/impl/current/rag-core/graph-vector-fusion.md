# Graph-Vector Fusion Retrieval

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

`retrieval_backend=fused` composes the configured vector lane (flat, parent-child, or hybrid) and
the selected GraphRAG strategy behind one `.retrieve(question, k)` wrapper. The wrapper in
`src/llb/rag/fusion.py` maps both lane rankings onto one candidate set through the selected
span-identity policy (`src/llb/rag/fusion_spans.py`), fuses them with generalized weighted RRF,
and keeps the surviving record's source offsets unchanged for recall@k and MRR. Reranking wraps
the fused result once, rather than independently reranking each input lane.

`graph_weight` is in `RunConfig`, run manifests, sweep cell keys, and fused Optuna trials. The Make
aliases forward `RETRIEVAL_BACKEND`, `RETRIEVAL_STRATEGY`, and `GRAPH_WEIGHT`; the comparison alias
also accepts `CONFIG`, `SPLIT`, and `COMPARE_RETRIEVAL_OUT` for a repeatable matched-store report.

```bash
make run-eval MODEL=<m> RETRIEVAL_BACKEND=fused GRAPH_WEIGHT=0.3
make compare-retrieval CONFIG=<run-config.yaml> GRAPH_WEIGHT=0.3 \
  GOLDSET=<answered-jsonl> COMPARE_RETRIEVAL_OUT=<report-json>
make sweep SWEEP_RAG_GRID="graph_weight=0,0.3,0.5"
make compare-graph-fusion CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> \
  GRAPH_WEIGHTS=0,0.1,0.3,0.5,1.0 GRAPH_FUSION_SPAN_IDENTITY=exact,overlap
```

## Fusion span identity (`graph_fusion_span_identity`)

The identity rule decides WHEN the two lanes are talking about the same candidate, which is the
precondition for RRF to reward agreement at all.

`exact` (the default) keys candidates by the exact `(doc_id, char_start, char_end)` triple. A graph
mention is a few dozen characters cut around an entity and a vector chunk is an ~800-character
recursive window, so the two lanes almost never agree by construction and fusion degenerates into
two disjoint rankings competing for the same result seats.

`overlap` folds a graph span into the vector chunk that CONTAINS it (and, for a span no chunk
covers, into whichever graph span it mutually overlaps), so the pair becomes one candidate both
lanes voted for. Two invariants keep the rule safe for span-level scoring:

- **Vector chunks are never merged with each other.** Consecutive recursive chunks share their
  `chunk_overlap` tail, so a transitive union would chain a whole document into one candidate. Only
  the vector lane creates anchors; the graph lane joins them, and a mention sitting in a shared
  tail joins the better-ranked chunk.
- **The survivor is an input record, verbatim.** A merge never synthesizes a union span, so the
  fused chunk's text stays an exact corpus slice at its own offsets. `metadata` records the policy
  in `fusion_span_identity` and every folded span in `fusion_merged_spans`.

A merge needs the intersection to cover at least `graph_fusion_span_merge_ratio` of the SHORTER
span: containment scores 1.0, a mention clipped by a chunk boundary scores its covered share, and
an incidental one-character touch between neighbouring chunks stays separate. Both endpoint weights
fuse nothing, so they are identity-independent.

The threshold is a knob in `RunConfig` (default `SPAN_MERGE_MIN_RATIO` = 0.5, valid over `(0, 1]`
where 1.0 is containment-only), recorded in the manifest fingerprint, and settable through `run-eval
--graph-fusion-span-merge-ratio`, `make sweep
SWEEP_RAG_GRID="graph_fusion_span_merge_ratio=0.25,0.5"`, and the evidence lane's
`GRAPH_FUSION_SPAN_MERGE_RATIO` grid. It is dead under `exact` (there is no partial overlap to
threshold), so the sweep grid expands `overlap` rows only and a non-default value extends a row
label as `/r<ratio>`. **The measured verdict is to pin 0.5 and not sweep it**, and it holds at two
chunk scales: at `chunk_size=800` the threshold decides essentially nothing (0.25 / 0.5 / 0.75 are
byte-identical on every row, because 99% of the graph spans touching a retrieved chunk are wholly
INSIDE it), and at `size=200` -- where a chunk and an entity mention are finally the same order of
magnitude -- it re-decides merges on up to a quarter of the questions yet still moves one headline
metric in one row, in 0.5's favor. See
[GraphRAG](../graphrag-backend/span-and-depth-evidence.md#span-merge-threshold-evidence) for the
grid, the agreement table, the overlap histogram, and [the smaller-chunk
re-run](../graphrag-backend/span-and-depth-evidence.md#does-the-pin-survive-a-smaller-chunk-size).

The knob rides `RunConfig`, the manifest fingerprint, `run-eval --graph-fusion-span-identity`,
`make sweep SWEEP_RAG_GRID="graph_fusion_span_identity=exact,overlap"`, and the sweep lane's
`GRAPH_FUSION_SPAN_IDENTITY` grid. `exact` remains the default: the measured adopt verdict for
`overlap` rests on a drafted multi-hop ledger, and the end-to-end run of the same two rows finds
the extra evidence is retrieval-only and costs measurable factoid answer quality -- see
[GraphRAG](../graphrag-backend/span-and-depth-evidence.md#span-identity-evidence) for both halves.

## Fusion candidate depth (`graph_fusion_candidates`)

`graph_fusion_candidates` is the per-lane candidate pool the graph share is applied over, the
graph-vector counterpart of the hybrid store's `fusion_candidates`. `None` (the default) asks each
lane for exactly `top_k`; a larger value retrieves that many from BOTH lanes, fuses, and then cuts
to `top_k`. A value below `top_k` is lifted to `top_k`, and both endpoint weights stay exact
single-lane passthroughs at `top_k` (a pool cannot change a ranking that is never fused). The knob
rides `RunConfig`, the manifest fingerprint, `run-eval --graph-fusion-candidates`,
`make sweep SWEEP_RAG_GRID="graph_fusion_candidates=10,50"`, and the sweep lane's
`GRAPH_FUSION_CANDIDATES` grid.

**A deeper pool cannot move a single-lane candidate into the top-k.** Graph-vector fusion uses
undamped reciprocal ranks, so a span that only ONE lane returns, at rank `r > k`, scores
`lane_weight / r`. That lane's own top-k spans are k distinct candidates each scoring at least
`lane_weight / k > lane_weight / r`, so at least k candidates outrank it at every graph weight. Only
a span BOTH lanes return, with at least one of its ranks below `k`, can be promoted by depth. That
makes the knob's usefulness a property of the corpus AND of the span-identity policy above: under
`exact` the measured Ukrainian goods corpus shares a candidate in 2 of 95 questions and depth
changes nothing at all, while under `overlap` it shares one in 93 of 95 and every depth row moves
(see [GraphRAG](../graphrag-backend/span-and-depth-evidence.md#candidate-depth-evidence) and [span
identity](../graphrag-backend/span-and-depth-evidence.md#span-identity-evidence) for both measured
verdicts). Depth is therefore a live knob exactly when the identity rule lets the lanes agree.

`compare-retrieval` ranks backends at ONE graph weight; `compare-graph-fusion` sweeps the weight and
decides it on the multi-hop slice with uncertainty; `compare-answer-quality` then scores the same
items END TO END under two of those rows and compares the answers, which is what separates a
retrieval-only coverage gain from an answer-quality gain -- see
[GraphRAG](../graphrag-backend.md) for all three
lanes, their measured CUDA-host evidence, and the artifact locations.

## Fusion question-type routing (`graph_fusion_router`)

`graph_fusion_router=question_type` changes `graph_weight` from one corpus-wide value into a
per-question endpoint choice: the configured share for likely multi-span questions, exactly zero
for likely single-span questions. The zero endpoint calls only the vector lane at `top_k`; it is an
exact ranking passthrough and does not query the graph store. `fixed` remains the default.

The pure policy lives in `src/llb/rag/fusion_routing.py`. A recognized sidecar label wins:
`multi-hop` and `comparative` route to graph fusion; `factoid`, `definition`, `numeric`, and
`procedural` route to vector. An absent or unknown label falls back to deterministic text signals:
a bridge term routes directly, while a long question routes only when it also names multiple
capitalized entities. `HeuristicPolicy` makes the word and entity thresholds explicit and
validated; setting the entity threshold to zero makes question length sufficient for controlled
calibration runs. The production default remains 16 words plus 2 linked entities. Conflicting
labels on duplicate question text are omitted from the sidecar map and therefore use the fallback.
Every decision records its source and signal tuple.

`FusedRetriever` accepts the router at the shared retrieval seam, while
`runner_retrieval._load_store` builds it from the configured gold-set sidecar. The setting is a
`RunConfig` field and is therefore present in every manifest and fingerprint; low-level runs can
select it with `run-eval --graph-fusion-router question_type` or YAML.

The fusion evidence command emits `routed/<strategy>@<weight>/d<depth>[/i<identity>]` rows beside
the fixed grid. `ROUTED_GRAPH_WEIGHT` controls their non-zero share; route counts are reported
overall and by question-type slice. The same label parses back into an ordinary answer-quality
`run-eval` lane, so the retrieval and answer comparisons exercise the production path rather than
a sweep-only approximation. `FUSION_HIDE_ROUTING_SIDECAR=1` exercises only the fallback in the
standard Make workflow; `FUSION_HEURISTIC_LONG_QUESTION_WORDS` and
`FUSION_HEURISTIC_MIN_LINKED_ENTITIES` select a frozen deterministic policy.

`make calibrate-fusion-routing` is the dedicated held-out workflow. It hides the sidecar from the
router while retaining each item's span count as the evaluation label, retrieves each physical
lane once per question, sweeps the declared threshold grid on `tuning`, freezes one policy, and
only then initializes and scores `final`. Its Markdown and JSON artifacts report confusion counts,
an item-id/signal ledger for routing errors, bootstrap precision/recall intervals, paired
multi-span coverage and single-span recall deltas, and an explicit recommendation gate. The gate
requires the tuning coverage interval to clear zero without a single-span interval below zero;
final never participates in selection and must pass the same gate independently before the frozen
policy can be recommended.

CI coverage is split along those seams: `tests/llb/rag/test_graph_vector_fusion.py` pins sidecar
precedence, heuristic signals, exact zero-weight passthrough, configuration fingerprints, and
runner wiring; `tests/llb/rag/test_fusion_evidence.py` pins routed replay and decision reporting;
`tests/llb/rag/test_fusion_calibration.py` pins threshold parsing, tuning-only selection, frozen
final scoring, and the no-gain refusal; `tests/llb/eval/test_answer_quality.py` pins label
round-tripping and the routing outcome summary.

```bash
make compare-graph-fusion CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> \
  ROUTED_GRAPH_WEIGHT=0.3 GRAPH_FUSION_CANDIDATES=k,50 \
  GRAPH_FUSION_SPAN_IDENTITY=exact,overlap
make calibrate-fusion-routing CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl>
make compare-graph-fusion CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> \
  SPLIT=tuning FUSION_HIDE_ROUTING_SIDECAR=1 \
  FUSION_HEURISTIC_LONG_QUESTION_WORDS=12 FUSION_HEURISTIC_MIN_LINKED_ENTITIES=0
make compare-answer-quality CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> \
  ANSWER_QUALITY_LANES=vector,routed/global_community@0.30/d50/ioverlap \
  SPLIT=final,tuning,calibration INCLUDE_DRAFTED=1
```

The CUDA result keeps the best fixed row's multi-hop retrieval gain while making every factoid
retrieval and answer an exact vector tie; see
[GraphRAG](../graphrag-backend/answer-quality-evidence.md#measured-result-question-type-routing-keeps-the-gain-and-clears-the-factoid-loss).
The held-out sidecar-free calibration recommends no threshold change; see
[GraphRAG](../graphrag-backend/answer-quality-evidence.md#sidecar-free-heuristic-calibration).
