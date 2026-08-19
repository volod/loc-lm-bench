# Chunking Strategies

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

The `src/llb/rag/chunking/` package implements every strategy behind one seam in `dispatch.py`
(`chunk_spans -> (start, end, metadata)`), each anchored to `doc_id` + exact character offsets so
`validate-goldset` and source-span scoring work identically across strategies:

- `fixed`: character window with overlap (pure Python, zero deps);
- `sentence`: pack whole sentences up to `size` (never cuts mid-sentence; a single unit longer
  than `size` falls back to the shared cap split -- see
  [`size` is a hard cap](#size-is-a-hard-cap-on-every-strategy));
- `recursive`: pinned langchain `RecursiveCharacterTextSplitter` (offset-verified; default);
- `markdown`: one chunk per leaf section BODY (heading lines stripped), breadcrumb in
  `metadata.headers`, long sections recursively sub-split;
- `semantic`: native embedding-distance-spike splitter over sentence offsets (pinned embedder);
- `page`: PDF page/citation-aware -- chunk boundaries never cross a `*.citations.json`
  page-sidecar span (loader: `doc_page_spans` reusing `page_metadata.load_page_citations`);
  pages longer than `size` sub-split WITHIN the page; docs without a sidecar fall back to
  `recursive`, as do `parent_child` children (their page coordinates are unknown inside a parent
  slice);
- `heading`: heading-hierarchy (layout-aware) -- a whole heading subtree that fits `size` becomes
  ONE chunk with heading lines INCLUDED in the text (unlike `markdown`); oversized subtrees emit
  their own section and recurse into child headings; every chunk carries the full breadcrumb;
- `late`: late chunking (Guenther et al. 2024) -- spans are IDENTICAL to `sentence` (so a
  retrieval delta isolates the embedding effect), but vectors are mean-pooled from
  whole-document token embeddings (`src/llb/rag/late_encoding.py`; the document is processed in
  consecutive encoder windows, e5-base: 512 tokens). Needs a token-level local embedder
  (`Embedder.passage_token_offsets` / `encode_passage_tokens`); flat mode only -- `RagStore.build`
  refuses `parent_child`; a chunk no token overlapped falls back to per-chunk encoding, logged.
- `table`: markdown-table-aware -- a chunk boundary never falls inside a table ROW; a table that
  fits `size` is ONE chunk carrying its nearest heading breadcrumb, a longer one splits between
  row blocks and every block records the header row's SOURCE offsets in
  `metadata.table_header_span`; non-table text routes through `recursive`, so a `table`-vs-
  `recursive` delta isolates the table handling (see
  [table-aware chunking](#table-aware-chunking)).

Selection: `make build-index CHUNK_STRATEGY=<name> CHUNK_SIZE=<chars> CHUNK_OVERLAP=<chars>` /
`build-index --strategy <name> --size <chars> --overlap <chars>` / `RunConfig.strategy`;
chunk-only via `python -m llb.rag.chunking --strategy <name>`. `make build-index CONFIG=<yaml>`
builds into that config's own `data_dir`, which is how an experiment gets a store beside its run
artifacts instead of overwriting the default one; with `CONFIG=` the YAML owns `corpus_root`
unless `CORPUS=` is also passed on the command line. The Optuna
tuner searches the original five by default; `llb tune --extended-chunkers` adds
`page`/`heading`/`late`/`table` (`EXTENDED_STRATEGIES` in `src/llb/optimize/tuning_space.py`) --
opt-in because `late` re-embeds whole documents per trial, `page` only differs from `recursive` on
sidecar-bearing PDF corpora, and `table` only differs on corpora carrying markdown tables.

Chunker comparison: `make compare-retrieval CHUNK_STRATEGIES=page,heading,late,markdown,semantic`
(`compare-retrieval --strategies ...`) builds one flat FAISS store per strategy over the SAME corpus
+ pinned embedder (persisted under `$DATA_DIR/llb/rag/<strategy>/`) and ranks them by recall@k / MRR
on the gold set, so the best chunker is demonstrated per corpus, never assumed. Add `NOISE_FLOOR=1`
to learn how much of a chunker delta the corpus can actually resolve ([measurement
floor](retrieval-metrics.md#measurement-floor---noise-floor)); the paired delta and verdict are
always reported as described under [paired lane
uncertainty](retrieval-metrics.md#paired-lane-uncertainty-and-verdict). Tests:
`tests/llb/rag/test_chunking_strategies.py` (offset round-trips, page-boundary alignment on the
committed `samples/pdf_pages` sidecar fixture, heading packing/breadcrumbs, late pooling math and
fallbacks) plus the pre-existing `test_chunking.py`/`test_page_metadata.py` suites.

Durable evidence, full corpus (2026-07-10, chunking-comparison-full-corpus on the CUDA host,
outside quick CI): all seven strategies over the verified 44-item quickstart-PDF accepted goldset
(5 PDF documents WITH `*.citations.json` sidecars, so `page` is genuinely page-aligned here;
pinned e5-base, k=10, non-saturated):

| strategy | recall@10 | MRR |
| --- | ---: | ---: |
| `sentence` | **0.977** | **0.740** |
| `recursive` (default) | 0.955 | 0.740 |
| `heading` | 0.932 | 0.716 |
| `semantic` | 0.932 | 0.721 |
| `page` | 0.909 | 0.724 |
| `late` | 0.886 | 0.576 |
| `markdown` | 0.818 | 0.703 |

Winner for this corpus: `sentence` (+0.022 recall@10 over the `recursive` default at equal MRR)
-- apply with `make build-index CHUNK_STRATEGY=sentence`. Important comparisons are `page` vs
`recursive` at -0.046 recall --
page-aligned packing LOSES to plain recursive splitting even on a sidecar-bearing corpus (page
boundaries cut mid-topic in these scanned-manual PDFs), so `page`'s value is page-provenance
display, not retrieval quality; `late` vs `sentence` (identical spans, late document-context
pooling) is -0.091 recall / -0.164 MRR -- late pooling blurs retrieval on this corpus and its
extra whole-document embed pass costs the most wall-clock of any strategy, so it stays a
prove-it-per-corpus option, never a default. `markdown` trails badly because the docling-emitted
markdown carries few semantic heading boundaries in the big 1.1 MB manual. Two caveats on those
rows: the bake-off predates the `size` cap below, so its `sentence` / `late` / `semantic` stores
still contained oversized units, and its 44-item set puts one item at 0.023 recall -- the
`sentence` win of +0.022 is under one item, which the
[measurement floor](retrieval-metrics.md#measurement-floor---noise-floor) lane exists to make
visible.

## Paired re-read of `sentence` versus `recursive`

CUDA-host re-read (2026-07-28), pinned e5-base, k=10, size 200 / overlap 30, 2000 paired
resamples, 95% confidence, seed 13, and `NOISE_FLOOR=1`, on the 95-item drafted goods set. The
point estimates reproduce the post-collapse rows recorded below exactly: `sentence` 0.632 and
`recursive` 0.695 recall@10, both at a +/-0.000 recall floor. Report, config, stores, and per-item
vectors are under
`$DATA_DIR/retrieval-comparison-paired-uncertainty/goods-chunking/`.

Against the named `recursive` baseline, `sentence` has recall delta -0.063
[-0.137, +0.000], with a 2/8/85 win/loss/tie ledger; its calibrated reading is flat. Its MRR
delta is -0.000 [-0.062, +0.062], 12/18/65, also flat. The verdict retains `recursive`: it is the
point-estimate leader and the available item set does not separate the two chunkers under paired
sampling. This result applies to the capped goods stores; the older seven-strategy accepted-PDF
ranking remains a separate forward re-run because that exact accepted item set is unavailable.

## Table-Aware Chunking

`src/llb/rag/chunking/table.py` (`table_spans`, dispatched as `CHUNK_STRATEGY=table`) is the
strategy for corpora whose answers live in converted-PDF markdown tables. It partitions a document
into TABLE regions and everything else:

- a table is found by the GFM rule -- a row line carrying a cell separator followed by a delimiter
  line of dashes/colons/separators -- and runs to the first line that is not a row, so
  `find_tables` never claims a plain `---` horizontal rule under a pipe-bearing paragraph;
- a table that fits `size` stays ONE chunk; a longer one packs WHOLE ROWS up to `size` (reusing
  the `sentence` packer), so a boundary never falls inside a row;
- every chunk of a table carries `metadata.headers` (the breadcrumb of its nearest enclosing
  heading, reusing the `markdown`/`heading` parser) and `metadata.table_header_span` --
  the header row's `[start, end]` SOURCE offsets, so a consumer can restore the column names a
  middle row block would otherwise have lost. The header is recorded as OFFSETS, never copied into
  the text, because chunk text must stay a verbatim corpus slice for the source-span metric;
- non-table text routes through the `recursive` splitter WITHIN its region, so a `table`-versus-
  `recursive` delta isolates the table handling;
- the one row a boundary may cut is a row longer than `size`: it falls through to the shared
  [`size` cap](#size-is-a-hard-cap-on-every-strategy), exactly as an over-long sentence does under
  `sentence`;
- in `parent_child` mode a child re-chunks its PARENT'S text, so the header span it finds is
  parent-local; `shifted_metadata` moves it with the child's own offsets (`_build_children` in
  `src/llb/rag/store_build.py`), because a recorded span that did not move would point at
  unrelated text.

`table` is a first-class `RunConfig.strategy` value (`Strategy` in `src/llb/core/config_fields.py`),
so `make build-index CHUNK_STRATEGY=table`, `CONFIG=` YAML, and the tuner all reach it.

Tests: `tests/llb/rag/test_chunking_table.py` -- registration in `STRATEGIES` and
`EXTENDED_STRATEGIES`, offset round-trips and the `size` cap at both a splitting and a
non-splitting size, no boundary inside a row, row-boundary-aligned blocks, the breadcrumb of each
table's own enclosing heading, header spans that resolve to the real header row, header text never
copied into a later block, no non-whitespace character lost, prose regions carrying no table
metadata, the horizontal-rule rejection, and the over-long-row fallback.

### Row integrity: what the strategy guarantees and what `recursive` already achieved

Row-cut census (2026-08-16) over five corpus/size settings (`chunk_corpus` + `find_tables`; a row
is "cut" when some chunk boundary falls strictly inside it):

| corpus | `size` | rows | rows longer than `size` | `table` cuts | `recursive` cuts | `sentence` cuts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| goods PDFs (30% table chars) | 200 | 2641 | 147 | 147 (5.6%) | 150 (5.7%) | 318 (12.0%) |
| goods PDFs | 800 | 2641 | 2 | 2 (0.1%) | 2 (0.1%) | 98 (3.7%) |
| HR PDFs (7% table chars) | 200 | 2302 | 30 | 30 (1.3%) | 30 (1.3%) | 148 (6.4%) |
| HR PDFs | 800 | 2302 | 1 | 1 (0.0%) | 1 (0.0%) | 63 (2.7%) |
| quickstart PDF (2.7% table chars) | 800 | 61 | 8 | 8 (13.1%) | 8 (13.1%) | 12 (19.7%) |

`table` cuts EXACTLY the rows that exceed `size` and nothing else -- the guarantee holds by
construction. The finding that decides the recommendation is the `recursive` column: the pinned
splitter's separator list is paragraph -> LINE -> word -> character, and a markdown table row IS a
line, so `recursive` already lands on row boundaries and cuts at most three rows more than the
theoretical minimum on any corpus here. The strategy's row guarantee is therefore worth ~0 against
`recursive` on these corpora and 2.7-8.8 percentage points of row integrity against `sentence`.

### Retrieval evidence

CUDA host (2026-08-16), `make compare-retrieval CHUNK_STRATEGIES=table,recursive,sentence
NOISE_FLOOR=1`, pinned e5-base, k=10, 2000 paired resamples, 95% confidence, seed 13. Reports,
configs, stores, and per-item vectors under `$DATA_DIR/table-aware-chunking/<run>/`
(`20260816T-goods`, `20260816T-pdf-accepted`).

95-item drafted goods ledger (`size` 200 / overlap 30; 24 of its 95 items have a gold span inside
a table row):

| strategy | recall@10 | MRR | recall delta vs `recursive` | reading |
| --- | ---: | ---: | --- | --- |
| `recursive` (baseline) | 0.695 | 0.465 | -- | -- |
| `table` | 0.695 | 0.465 | +0.000 [+0.000, +0.000], 0/0/95 | flat |
| `sentence` | 0.632 | 0.465 | -0.063 [-0.137, +0.000], 2/8/85 | flat |

40-item accepted quickstart-PDF goldset (`size` 800 / overlap 120; a literature corpus, 2.7% table
characters):

| strategy | recall@10 | MRR | recall delta vs `recursive` | reading |
| --- | ---: | ---: | --- | --- |
| `recursive` (baseline) | 0.925 | 0.852 | -- | -- |
| `table` | 0.925 | 0.831 | +0.000 [+0.000, +0.000], 0/0/40 | flat |
| `sentence` | 0.925 | 0.808 | +0.000 [-0.075, +0.075], 1/1/38 | flat |

Per-question-type slices ([question-type
slices](retrieval-metrics.md#question-type-slices)) on the numeric and comparative rows, which is
where tables carry the answers:

| corpus | slice | n | `recursive` | `table` | `sentence` |
| --- | --- | ---: | ---: | ---: | ---: |
| goods | numeric | 4 | 0.750 / 0.583 | 0.750 / 0.583 | 0.500 / 0.500 |
| goods | comparative | 2 | 0.500 / 0.500 | 0.500 / 0.500 | 0.500 / 0.500 |
| goods | multi-hop | 35 | 0.657 / 0.415 | 0.657 / 0.413 | 0.657 / 0.477 |
| quickstart PDF | numeric | 8 | 0.875 / 0.781 | 0.875 / 0.674 | 1.000 / 0.750 |
| quickstart PDF | comparative | 2 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

Verdict: **RETAIN `recursive`; `table` ships as an opt-in strategy, adopted nowhere by default.**
`table` reproduces `recursive`'s recall to three decimals on every row and every slice of both
corpora at a +/-0.000 measurement floor, and costs one rank position on one item per corpus (MRR
-0.001 goods, -0.021 quickstart; both flat). The incumbent rows reproduce their recorded numbers
exactly (`recursive` 0.695 / `sentence` 0.632 on goods), so the comparison is against the recorded
state, not a re-tuned one. Two reasons the null result is not a null strategy: the guarantee is
structural rather than incidental (`recursive`'s row alignment is a side effect of one separator in
a pinned third-party splitter, and nothing measures it per build), and `table_header_span` is
information no other strategy produces.

Why recall could not move here, stated plainly: `recall@k` credits an item when a retrieved chunk
OVERLAPS a gold span by a single character (`chunk_hits_span`,
[retrieval metrics](retrieval-metrics.md)), so a chunk that cuts a table row mid-way still scores a
hit with half a row. Row integrity changes what the model READS, not whether the metric fires --
which is why the two corpora agree that recall is invariant while the row-cut census differs by up
to 171 rows. The re-read below asks the same question on the axis that can see it.

### The intactness re-read of the same three chunkers

CUDA host (2026-08-19), the SAME command, corpora, `size`/`overlap`, k, seed, and resample count
as the rows above, re-scored once [evidence
intactness](retrieval-metrics.md#evidence-intactness-span_char_coveragek--span_intactk) existed.
Reports, configs, stores, and per-item vectors under
`$DATA_DIR/table-aware-chunking/20260819T-goods-intactness/` and
`.../20260819T-pdf-accepted-intactness/`. Recall@10 and MRR reproduce the recorded rows
BIT-IDENTICALLY on both corpora (the metrics are additive; `recursive` 0.694737 / 0.465155 on
goods, 0.925000 / 0.852321 on the accepted PDF goldset), so this is the recorded state re-read,
not a re-measured one.

| corpus | strategy | recall@10 | MRR | cover@10 | intact@10 | coverage delta vs `recursive` | intact delta vs `recursive` |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| goods (95) | `recursive` | 0.695 | 0.465 | 0.575 | 0.516 | -- | -- |
| goods (95) | `table` | 0.695 | 0.465 | 0.575 | 0.516 | +0.000 [+0.000, +0.000], 0/0/95, flat | +0.000 [+0.000, +0.000], 0/0/95, flat |
| goods (95) | `sentence` | 0.632 | 0.465 | 0.501 | 0.426 | -0.074 [-0.140, -0.017], 5/16/74, regressed | -0.089 [-0.168, -0.021], 3/13/79, regressed |
| PDF (40) | `recursive` | 0.925 | 0.852 | 0.925 | 0.900 | -- | -- |
| PDF (40) | `table` | 0.925 | 0.831 | 0.925 | 0.900 | +0.000 [+0.000, +0.000], 0/0/40, flat | +0.000 [+0.000, +0.000], 0/0/40, flat |
| PDF (40) | `sentence` | 0.925 | 0.808 | 0.925 | 0.850 | +0.000 [-0.075, +0.075], 2/1/37, flat | -0.050 [-0.150, +0.050], 1/3/36, flat |

**Row-aligned chunking does NOT separate from `recursive` on intactness at the reached sample
size, and the reason is that the two lanes are itemwise IDENTICAL.** `table` reproduces
`recursive`'s coverage and intactness to six decimals on both corpora with a `0/0/n` win/loss/tie
ledger -- not a small difference inside a wide interval, but zero items on which the two differ.
That closes the question the recorded verdict left open: the row-cut census gap of up to 171 rows
does not reach the top-10 evidence of a single gold item, because `recursive`'s separator list
already ends its chunks on line boundaries and a markdown row IS a line
([row integrity](#row-integrity-what-the-strategy-guarantees-and-what-recursive-already-achieved)).
The RETAIN verdict above is unchanged and now rests on the axis designed to move it.

The pair is not insensitive -- it is the only axis on which this comparison separates anything at
all. On goods, `sentence` reads flat against `recursive` on recall (-0.063, interval touching
zero) and flat on MRR (-0.000), yet loses 0.074 of span characters and 0.089 of whole-span
delivery on intervals clear of zero. In plain terms: `sentence` finds the same evidence at
statistically the same rate and hands the model visibly less of it. That is the separation the
recorded chunker re-read could not state. On the 40-item PDF goldset nothing separates on any of
the four columns, which is the expected reading of a literature corpus at 2.7% table characters.

Intactness by question type on goods (`recursive`, the shipped chunker) is the other operator-
visible fact this re-read produced:

| slice | n | recall@10 | cover@10 | intact@10 |
| --- | ---: | ---: | ---: | ---: |
| factoid | 40 | 0.700 | 0.667 | 0.650 |
| procedural | 14 | 0.786 | 0.706 | 0.357 |
| multi-hop | 35 | 0.657 | 0.401 | 0.400 |
| numeric | 4 | 0.750 | 0.750 | 0.750 |
| comparative | 2 | 0.500 | 0.500 | 0.500 |

The procedural row is the one to read: recall says the evidence is found for 0.786 of those items
and coverage says 0.706 of its characters arrive, but only 0.357 of the spans arrive in ONE chunk.
Procedural answers on this corpus live in multi-line steps that `size=200` splits, so the model
reassembles half of them from fragments. No chunker in this comparison changes that, and no metric
in the repo could show it before.

## `size` Is A Hard Cap On Every Strategy

`chunk_spans` runs every strategy's own boundaries through `cap_spans`
(`src/llb/rag/chunking/cap.py`), so no chunk is ever longer than the requested `size`. A
unit-packing strategy (`sentence`, `late`, `semantic`) otherwise emits a single unit whole however
long it is, and a structure-aware strategy does the same for a whole section: on converted
Ukrainian PDFs a markdown table, page furniture, or a heading block carries no sentence
terminator, so it packs into one multi-hundred-character span and an operator who asks for small
chunks silently does not get them -- and the affected text is exactly the numeric/tabular content
the retrieval slices care most about.

An oversized span is split on the pinned recursive splitter's separators (paragraph -> line ->
word -> character), keeping the largest natural boundary that fits. Offsets stay exact: sub-spans
are resolved inside the oversized slice and shifted back to source coordinates, and each inherits
its span's metadata (breadcrumbs survive the split). The splitter's last-resort separator is
per-character, so a residual oversized span is impossible; `cap_span` raises rather than letting
one reach the index. `markdown` / `heading` / `page` now route their long-section sub-split through
the same helper instead of each calling `recursive_spans` themselves -- their spans are unchanged
(verified byte-identical against the pre-cap implementation on the goods corpus at
`size=200` and `size=800`).

`summarize` (`src/llb/rag/chunking/corpus.py`) reports the audit numbers -- `oversize`,
`oversize_share`, `oversize_char_share` -- and `make build-rag-store` prints them as the `over%` /
`overC%` columns per strategy, so the cap is verifiable on any corpus without a bespoke script.

Measured `sentence` oversize share before and after the cap (`chunk_corpus` + `summarize`;
`max` is the longest chunk in chars):

| corpus | `size` | before: over% / of chars / max | after |
| --- | ---: | ---: | ---: |
| committed `ua_squad_postedited_v1` corpus | 200 | 20.2% / 32.2% / 713 | 0% / 0% / 200 |
| committed `ua_squad_postedited_v1` corpus | 800 | 0% / 0% / 796 | unchanged |
| converted Ukrainian goods PDFs | 200 | 21.6% / 44.3% / 1776 | 0% / 0% / 200 |
| converted Ukrainian goods PDFs | 800 | 5.9% / 8.9% / 1776 | 0% / 0% / 800 |

The leak is not only a small-chunk problem: at the DEFAULT `size=800` the goods corpus still put
8.9% of its indexed characters into over-budget chunks. Capping costs chunk count -- the goods
corpus goes 3333 -> 5019 chunks at `size=200` (+51%) and 976 -> 1073 at `size=800` -- so an index
build and every query touch more vectors.

Retrieval evidence (CUDA host, `make compare-retrieval CHUNK_STRATEGIES=sentence,recursive`,
pinned e5-base, k=10, the 95-item drafted goods multi-hop ledger, `CHUNK_SIZE=200`
`CHUNK_OVERLAP=30`; artifacts under `$DATA_DIR/chunk-size-cap/<run>/{before,after}/`):

| lane | recall@10 before | after | MRR before | after |
| --- | ---: | ---: | ---: | ---: |
| `sentence` | 0.611 | 0.621 | 0.414 | 0.411 |
| `recursive` (control, chunks byte-identical) | 0.642 | 0.653 | 0.419 | 0.414 |

No recall regression, and the delta is not distinguishable from measurement noise: the `recursive`
control chunks are byte-identical across the two runs yet its recall moved by the same +0.011,
because the preceding lane's different batch shapes perturb the encoder output by ~5e-7 per
dimension and that is enough to flip one borderline item at k=10 on 95 items. Repeat runs within a
code version reproduce byte-identically, so the drift is invisible to a naive repeat check.
`compare-retrieval --noise-floor` measures that floor directly and put this corpus at +/-0.021
recall@10 while its duplicates were still indexed -- read any smaller retrieval delta on those rows
as noise. The same comparison after [duplicate chunk
collapse](retrieval-store.md#duplicate-chunk-collapse) reads 0.632 `sentence` / 0.695 `recursive` at
a +/-0.000 floor ([measurement floor](retrieval-metrics.md#measurement-floor---noise-floor)), so the
rows above are the pre-collapse state of this corpus, kept because they are what the cap verdict was
measured on.

Tests: `tests/llb/rag/test_chunking.py` covers the cap over the committed
`samples/chunking/goods_table_uk.md` fixture (a heading + markdown-table block with no sentence
terminator, 613 chars) -- every strategy stays within `size`, stays offset-exact, loses no
non-whitespace character, and the fixture itself is guarded against gaining a terminator.
