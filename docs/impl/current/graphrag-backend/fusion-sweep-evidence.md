# Fusion Sweep Evidence

Part of the [GraphRAG backend](../graphrag-backend.md) area of the
[current implementation index](../../current.md).

## The graph-weight sweep lane

`llb compare-graph-fusion` (`make compare-graph-fusion`) is the lane that decides a graph weight.
`compare-retrieval` ranks backends over a whole gold set at ONE weight; this lane sweeps the weight
and answers the narrower question a recommendation needs -- on items whose answer requires more
than one source span, does fusing graph evidence retrieve more of that evidence, at which weight,
and at what cost elsewhere.

Three things separate it from the flat comparison:

- **A multi-span metric.** `recall@k` credits an item as soon as ANY labeled span is retrieved,
  which a two-hop item satisfies by returning only one hop. The lane reports `recall@k` beside
  `all-spans@k` (every labeled span covered) and `span coverage` (the fraction covered), all from
  [RAG core](../rag-core/retrieval-metrics.md#retrieval-metrics).
- **Uncertainty.** A multi-hop slice is tens of items, so every cell carries a paired percentile
  bootstrap interval over shared resample index sets, plus the item-level win/loss/tie ledger and
  an exact two-sided sign test. The verdict gates on the INTERVAL: a positive mean whose interval
  still fails the calibrated paired test is recorded as `inconclusive`, never as an adopt.
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

The make wrapper inherits the repo-wide `SPLIT ?= final`, so `make compare-graph-fusion` scores the
FINAL split alone unless it is cleared. Every fusion evidence run recorded here scores the whole
ledger and therefore passes `SPLIT=` explicitly; the report header's `scored items` count is the
check that the intended selection was used.

The lane also accepts the shared [paired-power
contract](../rag-core/paired-verdicts.md#paired-power-contract-for-comparison-lanes). Declare one
focus-slice row and metric before retrieval with `FUSION_POWER_REFERENCE=<earlier-comparison-json>`,
`FUSION_POWER_ROW=<fusion-row>`, `FUSION_POWER_METRIC=<metric>`, and `FUSION_MDE=<minimum-gain>`;
`FUSION_TARGET_POWER=<share>` overrides the 0.80 default. `power-plan.json` is written beside
`comparison.json` before the first retrieval, and the completed report states the realized SD,
binding variance-or-discordance floor, resolvable MDE, and resolution.

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

`NOISE_FLOOR=1` (`--noise-floor`) adds the [measurement
floor](../rag-core/retrieval-metrics.md#measurement-floor---noise-floor) per swept row, in TWO
blocks: over every scored item and over the focus slice alone, because the verdict is decided on the
focus slice and a floor measured on 95 items does not bound the band of a 35-item slice. It answers
a different question from the bootstrap intervals already in the table -- the intervals ask whether
the item SAMPLE supports a difference, the floor asks whether the rows differ for any reason other
than tie order -- and a weight recommendation needs both.

Artifacts per run: `report.md` (verdict, focus slice, overall, per-type slices, agreement table,
measurement floor when asked for, item ledger), `comparison.json`, and `run_config.json`.

## Accepted-ledger evidence, single graph weight

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

## Multi-hop slice evidence, swept graph weight

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

## Widened multi-hop review handoff

CUDA-host drafting evidence is under
`$DATA_DIR/graph-vector-fusion-multihop/20260727T-widened-draft-final/`. The
`make widen-multihop-draft` lane reused the five-document `goods-draft` extraction, drafted only
multi-hop paths with the same 12B MamayLM model at a 16,384-token context, excluded prior
evidence-span pairs before generation, and carried prior rows into one worksheet. The final
incremental pass made 40 local drafting calls (232.1 model-seconds, zero extraction calls), grounded
14 candidates, rejected five duplicates, and added nine rows to the 52-row carried ledger.

The handoff contains **61 multi-hop rows**, split 21 calibration / 20 tuning / 20 final. Every row
has exactly two distinct, exact-grounded spans and passes the Ukrainian output gate; 18 rows cross
document boundaries. The complete 61-row worksheet is `verify_sample.csv`. Its
`multihop_expansion_report.json` records `ready_for_human_review: true`, the 53 accepted-item
decision floor, and **eight rows of review headroom**. The drafting task is therefore large enough
to hand to review, but not large enough to absorb arbitrary attrition: rejecting nine or more rows
would make the downstream comparison undecidable again.

Two failed approaches are useful operational findings. Post-generation question-only dedup over
the first 120-path pass kept only 14 additions and produced 49 combined rows. Supplying the full
prior-question ledger to every prompt doubled model latency (537 to 1,140 model-seconds per
120-call pass) and produced no additions. The retained implementation instead caps novelty
guidance at 24 recent questions and uses an exact-question guard plus paired question/answer E5
similarity (0.90 / 0.95) for multi-hop rows. The final report records every rejected
candidate/nearest-prior pair, so the boundary remains auditable.

The generalized follow-on widening lane now selects paths by ordered relation pair, document mode,
and source document before drafting. It emits an explicit coverage/exhaustion artifact and derives
review headroom from the carried ledger rather than embedding this handoff's item counts in code.
The CUDA acceptance run, including the non-PDF carry-label and intra-batch duplicate fixes it
surfaced, is recorded in [Widening a multi-hop review
slice](../data-prep/ingestion-corpora.md#widening-a-multi-hop-review-slice). That bounded run
validates the reusable mechanics; the goods worksheet above remains subject to its human acceptance
task.

## The sweep re-read against its measurement floor

CUDA host, 2026-07-24; evidence under
`$DATA_DIR/graph-vector-fusion-multihop/20260724T-noise-floor/`. The same 95 drafted items, the
same weight grid, k, and seed were re-scored with `NOISE_FLOOR=1`. The vector store was REBUILT for
this run (the recorded one predates duplicate-chunk collapse and the v2 BM25 tokenizer, and its
lexical index is refused by the current build), so the run is also a reproduction check: 1139
chunks collapse to 1099 indexed, and 74 of the 102 compared metric cells are unchanged. The 28 that
moved moved by at most 0.029 (multi-hop recall on the `@0.30` rows) and 0.011 overall -- inside the
floors below. The verdict string, the winning row, and every headline number are identical.

Floors, worst lane of the sweep:

| item set | n | worst-lane fragile | floor recall@10 | floor MRR |
| --- | ---: | ---: | ---: | ---: |
| every item | 95 | 68/95 (`graph/global_community`) | +/-0.021 | +/-0.044 |
| multi-hop focus slice | 35 | 33/35 (`graph/local_khop`) | +/-0.043 | +/-0.074 |

Both floors are set by the GRAPH-ONLY rows, and the cause is the graph lane's score distribution,
not the corpus: link relevance sums a small set of link weights, so candidate lists carry long
exact-tie blocks and the rank-10 cut falls inside one for two thirds of the questions. `_rank_dedup`
(`src/llb/graph/retrieval.py`) breaks those ties deterministically on `(doc_id, char_start,
char_end)`, so the ranking reproduces across runs -- but which of many equally-scored spans lands in
the top 10 is decided by a document id rather than by relevance. Every fused row at a non-endpoint
weight reports `+/-0.000`: RRF ranks are integers, and once the vector lane contributes, the tie
block sits far below the cut.

Recorded verdicts re-read:

- **The multi-hop gain clears its floor.** `fused/global_community@0.10` gains +0.086 recall@10
  over the vector row on the multi-hop slice, against a +/-0.043 slice floor -- exactly twice the
  floor, so the gain is not tie order. Its bootstrap interval still touches zero, so the recorded
  `inconclusive` verdict is unchanged: the floor and the interval fail this row for different
  reasons, and only the SAMPLING one is still open.
- **The overall gains clear their floor.** +0.042 (`@0.10`) and +0.053 (`@0.20`) against +/-0.021.
- **The sweep does not choose between its two best weights.** `@0.20` leads `@0.10` by 0.011
  overall and by 0.000 on the multi-hop slice, against floors of +/-0.021 and +/-0.043; the report
  states in one line that the top two rows are not distinguished. Any recommendation naming one of
  those two weights over the other is reading tie order.
- **Graph-only retrieval still loses decisively.** -0.337 and -0.379 overall recall are an order of
  magnitude past the +/-0.021 floor, so that ordering is not an artifact of the tie blocks -- even
  though those same tie blocks are what make the graph rows' own recall fragile.
- **Endpoint and passthrough rows are unaffected.** `fused/*@0.00` reproduces the vector row.
