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

Measured 2026-07-21 on the RTX 4060 Ti 16 GB CUDA host. The run built a matched hybrid vector
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

Measured 2026-07-22 on the RTX 4060 Ti 16 GB CUDA host, over the `goods-draft` bundle.
A five-document, 1.15 MB converted Ukrainian goods-PDF
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
  measurement the flat comparison could not produce. `k=10` is the binding clause: the same grid
  re-scored at k=25 and k=50 lifts every row ([retrieval budget
  evidence](retrieval-budget-evidence.md#is-the-both-hops-ceiling-a-budget-or-a-query-problem)).
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

Drafted 2026-07-27 on the RTX 4060 Ti 16 GB CUDA host. The
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

Measured 2026-07-24 on the RTX 4060 Ti 16 GB CUDA host. The same 95 drafted items, the
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

## Scoring the graph lane below its node-relevance levels

The floor re-read above left one question open: the graph lanes' own recall was decided by tie
order for two thirds of the questions, and nobody had established whether those tie blocks were
REDUCIBLE or a property of the evidence. They were reducible, and the census says why.

### The tie-block census

Measured 2026-08-26 on an RTX PRO 3000 Blackwell 12 GB CUDA host -- a different host and a
different bundle from the 2026-07-24 re-read above, so the two are not directly comparable and the
reading below is the PAIRED before/after on this one. The corpus is the replayed five-document
Ukrainian goods-PDF draft bundle: 79 gold items (45 single-span, 34 labelled multi-hop by the
bundle's `item_provenance.jsonl`), a hybrid recursive vector store of 1,139 chunks collapsing to
1,099 indexed (multilingual E5 base, CUDA), and a graph store of 442 nodes, 238 edges, and 230
communities built from the same bundle's extraction. Every candidate pool is the 30-candidate pool
the floor already retrieves at k=10.

Before the change, over all 79 items:

| lane | cut is an exact tie | of those, a tie the 4-decimal published score MANUFACTURED | mean candidates sharing the cut score | distinct scores in a 30-candidate pool |
| --- | ---: | ---: | ---: | ---: |
| `graph/local_khop` | 54/79 | 0 | 11.2 | 3.6 |
| `graph/global_community` | 62/79 | 0 | 9.7 | 7.6 |

The counts are the whole finding. **Every fragile item was an exact tie** -- 54 and 62 match the
`fragile` counts the same run's floor block reported, so nothing was fragile for being merely
close. **None of those ties was rounding**: re-scoring the identical pools unrounded reproduces 54
and 62 exactly, so the published 4-decimal score never merged two relevance values that the lane
had actually separated. What the blocks are instead is one relevance value shared by many spans:
`local_khop` scores a node `1/(1 + hop distance)` and has three node values to work with at the
default depth, `global_community` gives every unmatched community member the same floor, and in
both lanes every mention of a node inherits the node's score. 74% (`local_khop`) and 81%
(`global_community`) of an average cut block was the mentions of a SINGLE node.

That also says what a finer NODE signal would have been worth: nothing. Edge weight, hop distance
as a continuous term, mention count, and community rank all score a node, and most of each tie
block was already one node. The signal had to be per SPAN, and the census says one exists: inside
the cut blocks, question-token overlap with the section title varied for 85% (`local_khop`) and 90%
(`global_community`) of blocks, and overlap with the span's own text varied for 89% and 69%.

### What the refinement changed

The lane now orders spans within a relevance level by their own question affinity, banded so it can
never cross a level ([retrieval strategies](modules-and-cli.md#retrieval-strategies)). Re-scoring
the identical stores and the identical 79 items at k=10 over the same 7-point weight grid, seed 13,
2,000 bootstrap resamples:

| reading | before | after |
| --- | ---: | ---: |
| worst-lane fragile, every item (n=79) | 62/79 | 36/79 |
| floor recall@10, every item | +/-0.025 | +/-0.006 |
| worst-lane fragile, multi-hop slice (n=34) | 31/34 | 18/34 |
| floor recall@10, multi-hop slice | +/-0.044 | +/-0.015 |
| mean candidates sharing the cut score (`local_khop` / `global_community`) | 11.2 / 9.7 | 5.2 / 5.5 |

The graph-only rows themselves, over all 79 items:

| row | recall@10 before | recall@10 after | MRR before | MRR after |
| --- | ---: | ---: | ---: | ---: |
| `graph/local_khop` | 0.367 | 0.392 | 0.133 | 0.283 |
| `graph/global_community` | 0.405 | 0.418 | 0.287 | 0.332 |

What the run establishes:

- **The measurement floor under every graph row shrank by 4.2x overall and 2.9x on the focus
  slice.** That is the deliverable: a graph-only row quoted to three decimals now earns roughly two
  of them instead of one. The floor is not zero -- 36 and 24 items still cut inside an exact tie --
  and it should not be, because those residual blocks are real: 22 of `local_khop`'s 36 and 16 of
  `global_community`'s 24 are entirely one node's mentions, none of which covers a question token
  in its text or its section title. The lane has no further evidence to separate them, so the
  remaining band is a property of the graph, not of the scoring.
- **Ranking quality moved with it, most on the lane that was worst.** `graph/local_khop` MRR rises
  from 0.133 to 0.283 overall and from 0.139 to 0.353 on the multi-hop slice -- the tie order was
  putting a relevant span outside the top ranks about as often as inside. Recall moves far less
  (+0.025 and +0.013 overall), which is the expected shape: a tie block that straddles the cut
  costs the metric only when a gold span is inside the block.
- **One recorded floor verdict changes, and it is the one this task existed to unstick.** Overall,
  `fused/global_community@0.10` leads `@0.30` by 0.013 recall@10; against the old +/-0.025 floor
  the report said the two rows were not distinguished, and against the new +/-0.006 floor it says
  the lead clears the floor at 2.0x. The multi-hop slice still does not choose between its top two
  rows (a 0.000 lead against +/-0.015), so the weight recommendation is unchanged -- the sweep can
  now separate two weights overall that it previously could only separate on tie order.
- **The sweep's headline verdict is unchanged**, which is the stability check: still `inconclusive`
  on `fused/global_community@0.30/d10` at +0.088 recall and +0.206 all-spans on the multi-hop
  slice, with the same borderline note. A tie-break refinement that had moved the verdict would
  have been evidence that the verdict rested on tie order.
- **The fused rows that read the graph lane's RANK improved too, without any change to fusion.**
  RRF consumes the graph lane's rank order, so a better-ordered lane feeds it better candidates:
  `fused/local_khop@0.20` goes from 0.823 to 0.886 recall overall and from 0.794 to 0.882 on the
  multi-hop slice, moving from below the vector row to above it. `global_community`'s fused rows
  were already dominated by the vector lane at these weights and barely move.
- **The binding MRR floor is now somewhere else.** At +/-0.048 overall and +/-0.067 on the slice it
  is set by `fused/*@0.50`, not by a graph row: at a graph weight of exactly 0.5 the two lanes
  carry equal RRF weight, so their contributions tie against each other. That is a different tie
  mechanism in the fusion arithmetic, and this task did not touch it.

What would overturn this: a corpus whose graph mentions carry no lexical relation to its questions
(the affinity term would score every span in a block identically and the floor would return to the
pre-change width), or a gold set whose spans sit in nodes with a single mention each (the tie
blocks the refinement targets would not exist to begin with). The reading is also on ONE bundle of
79 items on one host; the direction is large relative to the floor, the recall deltas are not.
