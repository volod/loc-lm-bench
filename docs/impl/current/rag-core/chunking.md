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

Selection: `make build-index CHUNK_STRATEGY=<name> CHUNK_SIZE=<chars> CHUNK_OVERLAP=<chars>` /
`build-index --strategy <name> --size <chars> --overlap <chars>` / `RunConfig.strategy`;
chunk-only via `python -m llb.rag.chunking --strategy <name>`. `make build-index CONFIG=<yaml>`
builds into that config's own `data_dir`, which is how an experiment gets a store beside its run
artifacts instead of overwriting the default one; with `CONFIG=` the YAML owns `corpus_root`
unless `CORPUS=` is also passed on the command line. The Optuna
tuner searches the original five by default; `llb tune --extended-chunkers` adds
`page`/`heading`/`late` (`EXTENDED_STRATEGIES` in `src/llb/optimize/tuner.py`) -- opt-in because
`late` re-embeds whole documents per trial and `page` only differs from `recursive` on
sidecar-bearing PDF corpora.

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
