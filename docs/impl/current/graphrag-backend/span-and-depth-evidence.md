# Span Identity And Candidate Depth Evidence

Part of the [GraphRAG backend](../graphrag-backend.md) area of the
[current implementation index](../../current.md).

## Candidate depth evidence

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
can never enter the top-k (see [RAG
core](../rag-core/graph-vector-fusion.md#fusion-candidate-depth-graph_fusion_candidates) for the
argument), so only spans BOTH lanes return can be promoted by a deeper pool. Across all 93 questions
the two lanes share an exact `(doc_id, char_start, char_end)` span **twice at depth 50** (once at
depth 10) per strategy -- graph evidence spans are entity mentions and edge evidence, whose
boundaries essentially never coincide with an 800-character recursive chunk.

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
and re-running the sweep is tracked as forward work in [`plan.md`](../../plan.md)
(`multihop-ledger-human-acceptance`).

## Span-identity evidence

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
- **The multi-hop gain becomes separable from zero -- on four questions.** The best `exact` row
  gains +0.086 [+0.000, +0.200] multi-hop recall (interval touches zero, `inconclusive`); the best
  `overlap` row, `fused/global_community@0.30/d50/ioverlap`, gains +0.114 [+0.029, +0.229] with 4
  wins, 0 losses, 31 ties, and does not pay for it overall (+0.063 [-0.011, +0.137]). The lane's
  recorded verdict on this ledger was **adopt**. Under the shipped minimum-evidence gate that
  deciding row reads `insufficient_evidence` and the verdict is **inconclusive**: four differing
  items cannot reach 95% on the exact sign test the same row prints ([the
  gate](../rag-core/paired-verdicts.md#the-minimum-evidence-gate-on-a-paired-reading)). The shipped
  selection-adjusted re-read also rejects it: the focus-slice recall hypothesis has family-draw
  marginal p 0.0628 and Westfall-Young step-down adjusted p **0.2310** over the 44 eligible fused
  rows x 4 metrics. Thus the `exact`-to-`overlap` default flip would fail selection even if a larger
  accepted ledger removed the item-count gate ([selection-adjusted
  verdicts](../rag-core/paired-verdicts.md#selection-adjusted-grid-verdicts)). The interval, the
  ledger, and every number in the tables above are unchanged; what changed is what may be read off
  them.
- **What that withdrawn reading needs is recorded, not left open-ended.** At its own discordance
  rate (4 differing items of 35) the 95% level is unreachable below **53 multi-hop items** ([the
  re-decision](../rag-core/paired-verdicts.md#the-re-decision-what-a-withdrawn-reading-needs)).
  Human acceptance can only SHRINK a drafted ledger, so the accepted-ledger re-run below cannot
  settle this row at the current drafted size of 35 -- widening the multi-hop drafting is the
  prerequisite, tracked in [`plan.md`](../../plan.md).
- **Candidate depth is now a live knob, and only because of the identity rule.** Under `exact`,
  all 10 (strategy, weight) pairs are byte-identical at depth 10 and 50; under `overlap`, all 10
  differ. Depth 50 is what turns `local_khop@0.30` from +0.029 to +0.057 multi-hop recall and from
  +0.011 to +0.063 overall.
- **`all-spans@10` still does not move.** The best row carries BOTH hops for 3 of 35 two-hop
  questions (0.086), the same ceiling every earlier row hit. Fusion improves WHICH single hop is
  retrieved and how the pool ranks; it does not solve two-hop coverage at k=10. That ceiling is a
  property of the BUDGET, not of the ranking: the same rows re-scored at k=25 and k=50 move
  together ([retrieval budget
  evidence](retrieval-budget-evidence.md#is-the-both-hops-ceiling-a-budget-or-a-query-problem)).
- **The weight optimum shifted with the policy.** Under `exact` the best row was `global_community`
  at weight 0.10; under `overlap` it is weight 0.30 -- unsurprising once a graph vote reinforces a
  chunk instead of displacing it, since a graph candidate no longer costs a result seat.

Verdict: `graph_fusion_span_identity` ships **opt-in with `exact` as the default**. The recorded
adopt never moved it -- the evidence is measured on the DRAFTED multi-hop ledger (see the boundary
above), and the project's standing rule is that a drafted slice does not move a default -- so no
shipped default rested on the reading the gate has since downgraded to `inconclusive`. `overlap` with
`graph_fusion_candidates=50` is the setting to enable when multi-hop retrieval coverage is the
goal -- but the end-to-end run below measures an answer-side cost on the factoid slice, so it is
not a free upgrade (see
[the overlap answer-quality result](answer-quality-evidence.md#measured-result-the-overlap-span-identity-carries-more-evidence-and-costs-factoid-answers)).
Flipping the shipped default is gated on the accepted-ledger re-run tracked in
[`plan.md`](../../plan.md) (`multihop-ledger-human-acceptance`).

## Span merge-threshold evidence

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
98.9% of the spans that overlap at all are fully contained. An ~800-character recursive chunk with
a 120-character overlap is an order of magnitude longer than the median entity mention (43
characters; p95 165), and a mention landing in the shared tail is contained in BOTH neighbours --
so straddling a boundary needs a mention to sit almost exactly on a cut.

Read that histogram as the SHAPE of the decision surface, not as a decision count. It buckets each
span's strongest overlap in the depth-50 pool, and both qualifiers matter: at depth 10 the
containing chunk may not be retrieved, so the best host available is a partial overlap the
threshold does decide (`local_khop` at depth 10 has one span in `[0.25, 0.5)`, and 0.25 versus 0.5
does fold it into a different candidate). `threshold_decisions.py`, archived beside the
chunk-size run below, counts what the policy actually decides and is the authoritative probe; at
`chunk_size=800` it finds at most 1 differing merge decision below containment-only, and even that
one moves no metric.

Verdict: **pin `graph_fusion_span_merge_ratio=0.5`**. On Ukrainian goods PDFs at `chunk_size=800` it
is not a tuning surface and the operator should not spend a sweep on it. The value stays exposed
(see [RAG core](../rag-core/graph-vector-fusion.md#fusion-span-identity-graph_fusion_span_identity))
because the sweep needed it. The one directional signal is that containment-only never helps --
where it moves overall recall at all it loses a question -- so 0.5 is not merely arbitrary among the
insensitive settings.

### Does the pin survive a smaller chunk size?

CUDA-host evidence is under
`$DATA_DIR/graph-vector-fusion-multihop/20260722T195633Z-small-chunk/`. The insensitivity above is
a property of the CHUNK SIZE, not of the policy, so the same 95 drafted items, graph stores, grid,
and seed were re-scored against two materially smaller chunkings built from the identical corpus --
`sentence` at `size=200` (3,333 chunks, median 169 chars, no overlap) and `recursive` at
`size=200 overlap=30` (4,848 chunks, median 154 chars) -- against the pinning run's 1,139 chunks at
median 726. Only the vector store changes: a graph mention is corpus-anchored, so both runs reuse
the pinning run's graph store byte for byte. At 200 characters a chunk and a mention are finally
the same order of magnitude.

**The threshold stops being inert.** `threshold_decisions.py` counts merge decisions that differ
from the pinned 0.5, and the questions whose fused top-10 changes at weight 0.30:

| chunking | strategy/depth | r0.25 | r0.75 | r1.00 |
| --- | --- | ---: | ---: | ---: |
| recursive@800/120 | global_community d50 | 0 | 0 | 8 |
| sentence@200 | global_community d50 | 8 | 18 | 34 |
| recursive@200/30 | global_community d50 | 2 | 3 | 13 |
| sentence@200 | local_khop d50 | 6 | 13 | 23 |
| recursive@200/30 | local_khop d50 | 6 | 5 | 23 |

At `sentence@200` the strictest setting re-decides 34 merges over 26 of 95 questions and changes
the top-10 of 11; at 800 the same comparison re-decided 8 over 8 questions and changed 2. The
middle histogram buckets that were empty at 800 are populated at 200: `sentence@200` puts 31
(`local_khop`) and 43 (`global_community`) of its overlapping depth-50 spans below full
containment, roughly 9-13% of them, spread across every bucket down to `(0, 0.25)`.

**The metrics still barely move, and 0.5 is never beaten.** Across the 24 `overlap` row families:

| chunking | r0.25 families differing | r0.75 | r1.00 | largest multi-hop recall change |
| --- | ---: | ---: | ---: | --- |
| recursive@800/120 | 0 of 24 | 0 of 24 | 16 of 24 | 0.000 |
| sentence@200 | 0 of 24 | 12 of 24 | 17 of 24 | -0.029 (r0.75 and r1.00) |
| recursive@200/30 | 2 of 24 | 3 of 24 | 14 of 24 | 0.000 |

Every difference at `recursive@200/30` and almost every one at `sentence@200` is MRR-only in the
fourth decimal. Exactly one row family moves a headline metric: `fused/local_khop@0.50/d50/ioverlap`
at `sentence@200` scores +0.029 [-0.057, +0.143] multi-hop recall and +0.105 [+0.032, +0.189]
overall at 0.25 and 0.5, and +0.000 [-0.086, +0.086] / +0.095 [+0.021, +0.179] at 0.75 and 1.0 --
one multi-hop question of 35, lost by TIGHTENING. No setting anywhere beats 0.5 on any metric.

Verdict: **keep 0.5; the default does not have to become chunk-size aware.** What changes with a
smaller chunk is the REASON, and that is worth recording because it decides what an operator should
do. At `chunk_size=800` the threshold decides nothing and a sweep is wasted. At 200 it decides real
merges on a tenth of the questions, and 0.5 sits at the safe end of a flat range whose only
measured slope runs the other way: tightening toward containment-only costs a multi-hop question
and never pays. An operator chunking below ~200 characters, or with a corpus whose mentions run
longer than these (median 43, max 367), should re-run `threshold_decisions.py` rather than the
histogram -- but should expect to keep 0.5 and should not sweep it as a tuning knob. The sweep's
own verdict row is untouched by the threshold at either chunk size: `recursive@200/30` still
adopts `fused/global_community@0.30/d50/ioverlap` (+0.143 [+0.029, +0.257] multi-hop recall,
+0.105 [+0.042, +0.179] overall) at the default threshold.

One caveat the run makes visible, separate from the threshold: smaller chunks are not free. The
vector baseline's overall recall@10 falls 0.705 -> 0.611 (`sentence@200`) and 0.632
(`recursive@200/30`), while its multi-hop `all-spans@10` rises 0.057 -> 0.086 and 0.114. Neither
is what this run was built to measure, and neither moves the shipped `chunk_size=800` default on
its own.
