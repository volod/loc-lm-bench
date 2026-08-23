# Retrieval Metrics

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

`src/llb/rag/retrieval.py` computes recall@k and MRR by source-span overlap. The common gate is
`recall@10 >= 0.8`.

The same module also computes two multi-span refinements used wherever an item's answer needs
evidence from more than one span (multi-hop questions):

- `span_coverage_at_k` -- the fraction of the item's labeled spans that the top-k covers.
- `all_spans_at_k` -- 1.0 only when EVERY labeled span is covered.

`recall_at_k` credits an item as soon as ANY labeled span is retrieved, which a two-hop item
satisfies by returning only one of its hops; on single-span items all three metrics are identical.
The graph-vector fusion evidence lane reports all three side by side, which is how a multi-hop
retrieval gain is distinguished from a partial hit.

When `all_spans_at_k` is stuck and no ranking knob moves it, `probe-multihop-hops` says why: it
ranks each labeled span by the item's own question and again by the span's own text, and classifies
every item as budget-limited, query-limited, or unreachable ([retrieval budget
evidence](../graphrag-backend/retrieval-budget-evidence.md#the-per-hop-probe-lane)).

Span matching is occurrence-aware: a chunk that collapsed byte-identical copies ([duplicate chunk
collapse](retrieval-store.md#duplicate-chunk-collapse)) hits a span labeled at ANY place its text
appears, so indexing a repeated passage once neither loses nor invents a hit.

This metric is not a model-ranking axis. It answers whether the retrieval layer is able to surface
the evidence the model needs. If retrieval is poor, answer quality is capped by context quality.

## Evidence Intactness (`span_char_coverage@k` / `span_intact@k`)

`recall@k` credits an item as soon as a retrieved chunk OVERLAPS a gold span by ONE character
(`chunk_hits_span`), so a chunk that cuts a table row in half scores the identical hit as one
carrying the whole row. `src/llb/rag/retrieval.py` reports the intactness pair beside recall@k /
MRR, computed from the SAME `(retrieved, spans)` pairs every comparison lane already builds:

- `span_char_coverage_at_k` -- the share of each gold span's characters the top-k carries between
  them, averaged over the item's spans. Overlapping chunks count each character once
  (`_union_length`), so two chunks that both cover a span do not score above 1.0.
- `span_intact_at_k` -- the share of the item's spans that some SINGLE top-k chunk carries whole.
  A span reassembled from two adjacent chunks is fully COVERED but not INTACT.

The four readings answer four different questions about one top-k: was the evidence found
(`recall@k`), how early (`MRR`), how much of it arrived (`span_char_coverage@k`), and did one
chunk carry it whole (`span_intact@k`). An item labeling no span is vacuously 1.0 on both, exactly
as `span_coverage_at_k` is; a degenerate zero-character span scores 0.0 on both, which is also what
`chunk_hits_span` reports for it, so the pair never disagrees with recall about the same span.
Matching is occurrence-aware on the same terms as recall: a chunk that collapsed byte-identical
copies covers a span labeled at ANY place its text appears.

Both are pure functions of the pairs, so every lane gets them for no extra retrieval:
`evaluate_retrieval` returns them in `RetrievalMetrics` (so every per-lane and per-slice row in
`compare-retrieval` carries them, and so does a run manifest's `retrieval` block), and
`embedding_bakeoff_uncertainty.item_vectors` emits their per-item vectors, so they ride the SAME
seeded bootstrap draw as recall and MRR and appear as paired deltas with intervals, win/loss/tie
ledgers, and calibrated readings on every lane the bake-off, the reranker lane, and
`compare-retrieval` score. The ASCII report renders them as the `cover@k` / `intact@k` point
columns and as a second paired-delta block under the recall/MRR one.

What they deliberately do NOT do: they never rank the leaderboard and never decide a verdict.
`VERDICT_BARS` in `src/llb/rag/comparison/uncertainty.py` stays `(recall_at_k, mrr)` and
the bake-off's adoption `BARS` are unchanged, so the Westfall-Young selection family is the same
size it was and every recorded adopt-or-retain call reproduces. They also say nothing about
whether an ANSWER used the evidence -- that is answer-side coverage, a separate measurement.
`--noise-floor` still bands recall@k and MRR only.

The two keys are `NotRequired` on `RetrievalMetrics` for one reason: a run manifest or a paired
artifact recorded before the pair existed must still validate and still re-read. `paired_rows`
skips a metric either side is missing, and the ASCII report prints `n/a` for a column an older
artifact never recorded.

The first evidence read on this axis is the chunker re-read in [table-aware
chunking](chunking.md#the-intactness-re-read-of-the-same-three-chunkers): recall and MRR reproduce
bit-identically, `table` turns out itemwise IDENTICAL to `recursive` on both intactness columns,
and `sentence` -- flat against `recursive` on recall and MRR -- loses span characters and whole-span
delivery on intervals clear of zero. That is the pair earning its place: the only axis on which
that comparison separates anything.

Tests: `tests/llb/rag/test_retrieval.py` (span carried whole, span split across two chunks, span
partly retrieved, span missed, overlapping chunks counted once, the per-span average, the `k`
cutoff, collapsed-duplicate occurrences, the no-span case, and the aggregate) and
`tests/llb/rag/comparison/test_compare_retrieval_core.py` (two lanes that tie on recall and separate
on intactness, per-slice columns, the paired block, the unchanged selection family, and the ASCII
rendering).

## Paired lane uncertainty and verdict

`compare-retrieval` now derives both aggregate rows and per-item vectors from one retrieval pass
per item per lane (`src/llb/rag/comparison/run.py`). It reuses
`embedding_bakeoff_uncertainty.item_vectors` / `paired_rows`, so every lane carries recall@k and
MRR deltas against one named baseline, a percentile interval, win/loss/tie ledger, calibrated
randomization reading, and neighbouring-confidence stability. One seeded bootstrap index set is
shared across all lanes and both metrics. The JSON report keeps those blocks on each
`backends` row and the aligned vectors in `paired_items`, so a weight or candidate comparison can
be re-read without retrieving again.

`compare-vector-stores` rides the same lane over the vector backends -- same paired columns, same
shared draw, same verdict -- so a backend swap is decided exactly the way an embedder swap is; its
knobs, artifacts, and measured readings are in [platform
matrix](../platform-vector-matrix.md#paired-backend-evidence). Both commands resolve their
incumbent through one shared `resolve_paired_baseline` in `src/llb/cli/rag/compare_stores.py`.

Baseline selection is mode-aware: `recursive` for a chunker comparison when present, `dense` for
hybrid, `faiss` for built backend comparisons and for `compare-vector-stores`, otherwise the first
scored lane. Override it with `--baseline` / `RETRIEVAL_BASELINE=` (`VECTOR_BASELINE=` on the
backend lane); a named lane the run did not score exits 2 rather than pairing against a different
row. `--resamples`, `--confidence`, and `--seed` have matching
`RETRIEVAL_RESAMPLES=`, `RETRIEVAL_CONFIDENCE=`, and `RETRIEVAL_SEED=` make variables. With
`CONFIG=`, the make alias now leaves the config's goldset and split intact unless the operator
explicitly overrides `GOLDSET=` or `SPLIT=`.

The report keeps point ranking and inference separate. It selects the best deployable row by the
existing recall -> MRR -> label order, then applies the standard paired evidence gate plus a
Westfall-Young lane x metric family adjustment. A positive separated recall delta may adopt the
winner. MRR may adopt only when recall is identical on every paired item; it cannot hide an
unresolved recall tradeoff. Disabling resampling marks the readings `unmeasured` and can never
produce ADOPT. Otherwise the verdict retains the baseline. One retain case states more than "the
gap did not separate": when the winner's ledger has zero wins AND zero losses on EVERY scored
metric, the verdict says the two lanes are itemwise identical, so the point-estimate lead is label
order and no larger item set could separate them -- a settled comparison rather than an unresolved
one. That is the routine reading of an invariant seam such as the vector backends.
`dense+oracle-doc` and the lexical-only diagnostic row remain visible with paired columns but
cannot receive ADOPT.
Rendering is isolated in `comparison/report.py`, and the decision is isolated in
`comparison/uncertainty.py`. One rendering rule the intactness columns forced: the shared
`reading_of` vocabulary is ONE-SIDED (it asks whether the candidate separated), so a lane the
baseline beats by an interval clear of zero used to print `flat`, which reads as "nothing to see".
The report now names that case `regressed`, using the existing `paired.regresses` predicate with
its own minimum-evidence gate. It is a rendering change only -- no verdict, adoption bar, or
recorded number moves, and a lane whose interval merely TOUCHES zero still reads `flat`.
Fake-store tests cover exact point reproduction, one-pass
retrieval, persisted item ids/vectors, recall and MRR adoption rules, baseline validation, ASCII
rendering, and the CLI JSON artifact. The full RAG suite runs independently; its reranker latency
test no longer imports a fixture through a nonexistent `tests` package.

## Question-Type Slices

An aggregate recall row cannot say WHICH questions a change helped, and a chunking change is
exactly the kind that helps one slice and hurts another. `compare-retrieval` therefore reports a
per-question-type breakdown beside the aggregate: every lane is scored again on the items carrying
each label, from the SAME retrieval pass (no second retrieval, so the slices cost nothing).

A slice also carries its OWN paired reading against the baseline lane, drawn on that slice's items
with the run's resamples/confidence/seed (`_one_slice` in `src/llb/rag/comparison/run.py`). The
reason is the slice size: a 14-item slice turns on one or two questions, so a point move from 0.357
to 0.714 there is not yet a reading -- the interval and the win/loss/tie ledger beside it are what
say whether it is an item set. The JSON artifact carries all four metrics' slice deltas; the ASCII
rendering prints the INTACTNESS pair under each slice's point rows (`SLICE_PAIRED_BLOCK`), because
the aggregate table already carries all four blocks and a slice is the resolution at which an
evidence-delivery change is read. An empty slice pairs nothing rather than reporting invented zeros.

The labels are not in the gold set -- a `GoldItem` has no question type -- they live in the draft
bundle's sidecars, and `src/llb/rag/question_types.py` is the one place that knows where those sit:
`needle_items.jsonl` (ontology-assisted drafting) and `item_provenance.jsonl` (the external-draft
import lane), each looked up beside the gold set and one level up when the gold set is an accepted
ledger under `accepted/`. The two are JOINED, nearest sidecar first, so a bundle carrying either one
(or both) slices the same way; a gold set with neither reports no slices at all instead of an
invented label.

`FOCUS_SLICES` (`src/llb/rag/comparison/models.py`) -- `numeric`, `comparative`, `multi-hop`,
`procedural` -- are
always present in the JSON report even at `n=0`, so a reader can tell "this corpus labels no
numeric question" from "nobody looked". The ASCII rendering scores only the non-empty slices and
names the empty ones on one line, because printing a zero-item slice's zeros would read as a
measured result. Those four are the slices a chunking change is read on: in converted Ukrainian
PDFs the numeric and comparative answers live in tables ([table-aware
chunking](chunking.md#table-aware-chunking)), a multi-hop answer needs every span carried at
once, and a procedural answer is a multi-line step sequence -- the shape a `size` cap cuts, which
is why it is where evidence INTACTNESS is read.

## Served Context Cost (`served_chars@k`)

A lever that delivers more of a gold span by serving MORE text is a different trade from one that
delivers the same characters in fewer pieces, and the four quality columns cannot tell the two
apart. `served_chars_at_k` (`src/llb/rag/retrieval.py`) is the cost column that can: the mean total
characters of the top-k chunk TEXTS, so what an overlapping chunk pair serves twice is counted
twice, because what the model pays for is what is laid into its prompt rather than the character
union behind it.

It rides the same `evaluate_retrieval` return every lane already computes, so every per-lane and
per-slice row in `compare-retrieval` carries it for no extra retrieval, and the ASCII report renders
it as the `chars@k` point column beside `intact@k`. It is deliberately NOT in `METRICS`: it is never
resampled into a paired delta and no verdict is argued from it. A lane is adopted for what it
delivers; this column is what delivering it cost. Like the intactness pair, the key is `NotRequired`
on `RetrievalMetrics` so an artifact recorded before it existed still validates and still re-reads,
printing `n/a` in that column.

## Reading A Stitched Twin (`<row>+stitch`)

`compare-retrieval --stitch` adds an assembly-time twin per compared row: the SAME top-k with
contiguous same-document chunks merged into one block ([contiguous-chunk
stitching](context-assembly.md#contiguous-chunk-stitching-fragmented-evidence-delivery-lever)). The
twin retrieves nothing its base lane did not, which fixes how its four columns are read:

- `recall@k` and `cover@k` MUST reproduce the base lane exactly -- the retrieved character set is
  identical -- and `stitch_report` records that invariance per twin as `recall_invariant` /
  `coverage_invariant` in the report's `stitching` block, rendered as `invariance held` or
  `INVARIANCE FAILED`. A failure means the twin changed evidence rather than reflowing it and its
  reading is void.
- `intact@k` is the axis the lever moves, upward only: a merge can turn a span that arrived cut into
  one a single block carries whole, and can never do the reverse.
- `chars@k` is its price, and for stitching it can only fall or stay level, because merging counts a
  chunk-overlap once instead of twice.
- `mrr` is NOT readable: merging shortens the returned list, so the first hit can only move to an
  earlier position. The report prints that caveat on its own line beside the stitch census.

Because a stitched twin ties its base lane on both metrics a verdict is decided on,
`verdict_lanes` (`src/llb/cli/rag/compare_retrieval_lanes.py`) keeps it out of the eligible set --
it is a reported lever, not an adoption candidate, and letting it win a tie-break on a compressed
`mrr` would be the one way the report could lie about it.

## Measurement Floor (`--noise-floor`)

`recall@k` / `MRR` are reported to three decimals, and the floor under those decimals is a
property of the CORPUS, not zero by default. `src/llb/rag/noise_floor/measure.py` measures it (and
`src/llb/rag/noise_floor/report.py` renders the one ASCII and one Markdown block every lane below
shares):
`NOISE_FLOOR=1` (`--noise-floor`, `NOISE_FLOOR_REPLICATES=` / `--noise-floor-replicates` to change
the replicate count) retrieves a `3k` candidate pool once per lane, perturbs every candidate score
by `N(0, 1e-6)`, re-ranks, keeps the top k, and reports the band the metric spans over 64 seeded
replicates plus the worst-lane `floor` to read every delta against. The replicates only re-sort a
cached pool, so the whole measurement costs one extra retrieval pass per lane; the seed is stable
per lane (`crc32` of the label, never the salted `hash()`), so a report reproduces byte-identically.

The measurement is store-agnostic -- it needs a lane's candidates and their scores, nothing else --
so every comparison lane that publishes three-decimal rows reads its own floor through the one
module:

| lane | flag | where the floor lands |
| --- | --- | --- |
| `make compare-retrieval` | `NOISE_FLOOR=1` | the ASCII table and `report["noise_floor"]` |
| `make compare-embeddings` | `NOISE_FLOOR=1` | a `### Measurement floor` block in `report.md` |
| `make compare-vector-stores` | `NOISE_FLOOR=1` | the ASCII table and the `--out` JSON |
| `make compare-graph-fusion` | `NOISE_FLOOR=1` | two blocks in `report.md`: every item, and the focus slice |

Two properties the multi-lane wiring needed:

- **A recommendation is restated as clearing the floor or not.** Every floor report carries a
  `margin`: the top two lanes by recall@k (ties broken by MRR, the order every table here ranks
  by), their gap, and whether that gap exceeds the floor. It is rendered as one sentence, because
  a lane comparison names ONE winner and a winner whose lead is inside the floor has not been
  distinguished from the runner-up -- which is exactly how a bake-off's sub-item delta becomes a
  recommendation. The same margin now persists its signed `clearance` (`delta - floor`) and
  `floor_multiple` (`delta / floor`, null at a zero floor), so the binary never appears without
  its distance from the cut. This is deliberately not called `p_positive`: the floor is a
  deterministic range over score perturbations, not a paired bootstrap confidence interval.
- **A FUSED row is perturbed at its own depth.** Most lanes extend cleanly -- a dense store's top-k
  is the prefix of its top-3k -- but a fused row's ranking depends on how deep each lane was asked,
  so retrieving `3k` from it would answer for a DIFFERENT row (that is the [candidate-depth
  knob](graph-vector-fusion.md#fusion-candidate-depth-graph_fusion_candidates)). Such a row exposes
  `retrieve_candidate_pool(question, k, candidates)`: fuse exactly as at `k`, move only the cut. The
  fusion sweep also caches its lanes `3k` deep under `--noise-floor` (`build_sweep_rows(...,
  pool_depth=...)`), which widens no row's ranking because every row still asks for its own depth.

The floor a comparison quotes is the WIDEST band any lane showed, and each per-lane band is itself
a 64-replicate sample: two lanes with byte-identical rankings can report `+/-0.000` and `+/-0.005`
because they are seeded independently. Read the per-lane bands as fragility evidence and the
worst-lane `floor` as the number a delta must clear.

Why `1e-6`: two processes that built BYTE-IDENTICAL chunks on this host produced dense vectors
differing by up to 5.4e-7 per dimension -- the encoder's kernels depend on the batch shapes it
saw earlier in the process, so the lane built BEFORE this one changes its output -- which moved
the cosine scores by up to 6.0e-7 (mean 1.3e-7). Repeats WITHIN one process are byte-identical,
so a naive repeat check reports a spread of zero and never sees the drift. The default rounds the
measured maximum up and perturbs every candidate independently, so the reported floor is
deliberately conservative: a delta that clears it is not numeric noise.

Each lane also reports `fragile N/n` -- items whose rank-k and rank-(k+1) candidates sit within
the jitter, so their top-k membership is decided by noise or by the backend's arbitrary order at
an exact tie. That count explains the band's width and is the number to act on: acting on it is
exactly what [duplicate chunk collapse](retrieval-store.md#duplicate-chunk-collapse) did.

Measured floors on the chunker lane (CUDA host, pinned e5-base, k=10, `sentence` vs `recursive`;
reports under `$DATA_DIR/retrieval-noise-floor/<run>/`):

| corpus | n | chunk `size` | duplicate chunks | fragile | floor recall@10 | floor MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| converted Ukrainian goods PDFs (before collapse) | 95 | 200 | 37.7% | 25/95 | +/-0.021 | +/-0.018 |
| converted Ukrainian goods PDFs (shipped) | 95 | 200 | 0.0% | 1/95 | +/-0.000 | +/-0.000 |
| committed `ua_squad_postedited_v1` (final split) | 82 | 800 | 0.0% | 0/82 | +/-0.000 | +/-0.000 |
| accepted converted-PDF goldset | 40 | 800 | 0.5% | 1/40 | +/-0.000 | +/-0.000 |

Measured floors on the other three lanes (CUDA host, 2026-07-24, k=10; see each lane's section for
the tables the floors are read against):

| lane | corpus / item set | n | worst-lane fragile | floor recall@10 | floor MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| `compare-embeddings` (4 candidates) | accepted converted-PDF goldset, `recursive` 800/120 | 40 | 0/40 | +/-0.000 | +/-0.000 |
| `compare-vector-stores` (faiss/chroma/qdrant) | the same corpus + goldset | 40 | 0/40 | +/-0.000 | +/-0.000 |
| `compare-graph-fusion`, every item | drafted goods multi-hop bundle | 95 | 68/95 | +/-0.021 | +/-0.044 |
| `compare-graph-fusion`, multi-hop slice | the same bundle's focus slice | 35 | 33/35 | +/-0.043 | +/-0.074 |

The two dense-only lanes have nothing to arbitrate: with duplicates collapsed no item's rank-10 /
rank-11 cosine scores sit within `1e-6`, so a delta of any size in those tables is a real ranking
difference (still subject to SAMPLING uncertainty, which the floor does not answer). The fusion
sweep is the opposite case, and its cause is measured: the GRAPH lanes score by link relevance, a
sum over a small integer-ish set of link weights, so their candidate lists carry long exact-tie
blocks (`5.0077, 5.0075, 2.5042, ... , 0.001, 0.001, 0.001, ...` -- eight identical `0.001` tails
are typical), and the rank-10 cut falls inside such a block for 68 of 95 questions. `_rank_dedup`
in `src/llb/graph/retrieval.py` breaks those ties deterministically on `(doc_id, char_start,
char_end)`, so the ranking is REPRODUCIBLE -- but reproducible is not the same as retrieved: which
equally-scored span lands in the top 10 is decided by a document id, not by relevance. Every fused
row at a non-endpoint weight inherits a `+/-0.000` band, because RRF ranks are integers and the
tie block is far below the cut once the vector lane contributes.

The floor tracks DUPLICATE CHUNKS, not gold-set size, and that is why the measured floor is now zero
on every corpus: the goods corpus at `size=200` HAD 37.7% of its chunks byte-identical to another
chunk (repeated page furniture and table boilerplate in converted scanned manuals; the largest
identical group was 58 copies for `recursive` and 72 for `sentence`), identical text embedded to an
identical vector, that scored an exact tie, and the backend broke the tie by candidate order -- so a
quarter of that corpus's items had a top-10 membership no retrieval property decided. [Duplicate
chunk collapse](retrieval-store.md#duplicate-chunk-collapse) indexes each distinct passage once and
gives any surviving tie a documented `chunk_id` tie-break, which removes the mechanism; the row
above is the same corpus, goldset, k, and seed re-measured with it.

Verdicts re-read against the measured floors:

- Goods PDFs at `size=200`: with duplicates collapsed, `recursive` leads `sentence` by 0.063
  recall@10 against a +/-0.000 floor, so the recall ranking is now resolved rather than at the
  edge (before collapse it was 0.032 against a +/-0.021 floor, the two bands touching at 0.621).
  The MRR gap closed to 0.000 -- the two chunkers rank their first hit equally well here, and the
  earlier 0.003 gap was inside its own floor and meant nothing either way.
- Committed UA fixture and the accepted PDF goldset: floor 0.000, so their recorded recall/MRR
  deltas are not numeric noise. They remain subject to SAMPLING uncertainty, which is a separate
  question the paired-bootstrap lanes answer -- a 0.022 recall delta on a 44-item set is under one
  item either way.
- Embedder bake-off: the recorded `e5-base` recommendation is NOT reproduced on the accepted
  goldset that still exists, and the floor is not why -- see [the bake-off
  re-read](embedders.md#the-recommendation-re-read-against-the-floor). The paired re-read then
  showed the challenger's lead does not clear its SAMPLING interval either, and the two scored
  corpora separate different candidates -- see [the paired
  re-read](embedders.md#the-recommendation-re-read-with-paired-uncertainty).
- Vector-store backends: `faiss`, `chroma`, and `qdrant` return the identical recall@10 / MRR on
  this corpus, so the `best (recall@k)` line is label order, not a ranking -- see
  [platform matrix](../platform-vector-matrix.md#embedding-bake-off).
- Graph-vector fusion: the recorded multi-hop and overall gains clear their floors; the CHOICE
  between the two best weights does not -- see
  [GraphRAG](../graphrag-backend/fusion-sweep-evidence.md#the-sweep-re-read-against-its-measurement-floor).

A zero floor is not a permanent property of a corpus: it is measured per run, and a corpus whose
chunks tie for a reason collapse does not remove (a backend that rounds its scores, a graph lane
whose link relevance saturates, or lexical fusion producing equal RRF sums) will report a non-zero
band again.

The floor is opt-in, so every existing comparison row is unchanged when it is not asked for. Tests:
`tests/llb/rag/noise_floor/test_noise_floor.py` (zero floor on separated scores, a full 0.0-1.0 band
when the cut sits on a tie, the fragility count, per-lane seeding and reproducibility, the
unscored-lane skip, the margin reading and its MRR tie-break, the candidate-pool seam, and the ASCII
rendering), plus each lane's own wiring in `tests/llb/rag/test_embedding_bakeoff.py`,
`tests/llb/rag/test_compare_retrieval.py` (the `compare-vector-stores` CLI over injected stores),
and `tests/llb/rag/test_fusion_evidence.py` (per-row and focus-slice floors, and that the pool seam
keeps the ranking the sweep published) -- all over fake stores, no FAISS, no GPU.

The default store retrieves dense-only (cosine over the pinned E5 embedding). Measured against the
gate, dense-only passes on the committed fixture (`recall@10=0.980`) but falls short on the real
full-corpus PDF index (`recall@10=0.729`, see the quickstart note in [the retrieval
store](retrieval-store.md)), so dense-only has NOT been proven sufficient for a real Ukrainian
corpus. [Hybrid retrieval](hybrid-retrieval.md), [cross-encoder reranking](rerank-and-query.md), and
the ordered `rag/query_prep/` pipeline are the available retrieval levers.
