# Retrieval Store And Lifecycle

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

## Retrieval Store

`src/llb/rag/vector_store/store.py` builds `RagStore`:

- chunks the corpus through `llb.rag.chunking`;
- embeds with the pinned multilingual E5 embedder;
- stores chunk records with exact source offsets;
- persists a vector index through the vector-store seam.

The default backend is FAISS. Chroma, Qdrant, and LanceDB use the same `VectorIndex` protocol in
`src/llb/rag/vector_store/vector_index.py`.

Chunk-to-source linkage (audited 2026-07-04 against the Ukrainian-RAG production checklist): every
chunk record in every strategy and both retrieval modes carries `doc_id`, a unique `chunk_id`, and
exact `char_start`/`char_end` offsets, so any chunk resolves to its verbatim place in the source
document.

Page/section provenance (`src/llb/rag/page_metadata.py`, shipped): after chunking, `RagStore.build`
joins each chunk's char span onto the `pdf-<digest>.citations.json` page-span sidecars that sit
beside the corpus docs, adding `metadata.pages = [first, last]` (source-PDF page numbers) and
`metadata.source_pdf` (the original PDF path) to every chunk whose span intersects a page, in every
strategy and both retrieval modes. The same pass fills `metadata.headers` -- the breadcrumb of
enclosing markdown headings located in the source -- for strategies other than `markdown` (which
already emits it) and for any doc with headings; plain `.md`/`.txt` docs get header breadcrumbs but
no page fields. The join is additive: chunk text, ids, and offsets are byte-identical before and
after. `store_meta.json` records `page_annotation_coverage` (the fraction of indexed chunks that
gained a `pages` field) and `build-index` logs it. In `parent_child` mode both the indexed children
and their parents are annotated, so the fields surface on retrieval hits either way. Retrieved hits
carry these fields, so verify cards, cited answers, miss clustering, and metadata filters can say
"file X, page N, section Y" without re-deriving the join.

Governance metadata (`src/llb/prep/corpus/governance.py`, `src/llb/rag/chunking/corpus.py`, and
`src/llb/rag/vector_store/store.py`) is joined from `corpus_manifest.json` onto every chunk as additive
`metadata.language`, `metadata.ingestion_time`, `metadata.source_system`, optional
`metadata.version`, optional `metadata.effective_date`, and optional `metadata.acl_label`.
The stored chunk text, ids, and offsets stay byte-identical. `store_meta.json` records the
`corpus_fingerprint`, the manifest filename, and the governance field list. `run-eval` compares
that fingerprint with the current corpus manifest before loading the vector store; a changed or
deleted source refuses with a refresh/rebuild message instead of silently serving stale chunks
(`llb refresh-index` applies the incremental path). Immutable store directories are the rollback
unit (see Dynamic Corpus Refresh below).

ACL scoping uses the same metadata-filter seam as page and heading filters:
`metadata_filter(acl_label=...)` rejects any chunk whose `metadata.acl_label` differs, and
`run-eval --acl <tag>` passes that predicate into retrieval before dense ranking, hybrid fusion,
or reranking. An ACL-scoped query therefore never receives an out-of-scope chunk; if no chunk is
in scope, the case is a retrieval miss before generation.

Durable evidence (2026-07-04, heavy build on the CUDA host, outside quick CI): a `markdown`/`flat`
store over the quickstart HR PDF corpus (`.data/quickstart-pdf-corpus-hr/_md`, 8 converted docs)
annotated all 2855 indexed chunks with page provenance -- `page_annotation_coverage = 1.0` in
`store_meta.json` -- every chunk carrying `metadata.pages`, `metadata.source_pdf`, and its heading
breadcrumb.

Retrieval modes:

- `flat`: index generation chunks directly;
- `parent_child`: index smaller child chunks and return deduplicated larger parent chunks;
- `hybrid`: index like `flat`, plus a lexical BM25 index fused with the dense ranking at query
  time (see Hybrid Retrieval below).

### Duplicate Chunk Collapse

Shipped (duplicate-chunk-suppression, `src/llb/rag/duplicates/collapse.py`): `RagStore.build`
indexes each DISTINCT chunk text once. Converted-PDF corpora repeat page furniture, boilerplate
instructions, and table headers verbatim -- on the measured goods corpus every one of the 494
collapse groups repeats INSIDE a single long manual, none across documents -- so the same text was
embedded, stored, and searched many times over; worse, identical text embeds to an identical vector,
which scores an EXACT tie, which the backend broke by candidate order -- so an item whose top-k cut
fell inside such a group had a metric no retrieval property decided (see [measurement
floor](retrieval-metrics.md#measurement-floor---noise-floor)).

How it works:

- Collapse is EXACT-only (byte-identical text) and keeps the FIRST copy in build order, so the
  surviving record is deterministic across rebuilds. Near-duplicate DOCUMENTS remain the corpus
  conflict lane's question ([data prep](../data-prep.md)).
- Collapse applies only where a chunk's vector is a PURE FUNCTION OF ITS TEXT, which is the
  premise the paragraph above states. The `late` strategy breaks it: its vectors are mean-pooled
  from whole-document token embeddings, so the same passage at two document positions carries two
  DIFFERENT vectors and dropping one would discard exactly the document context the strategy
  exists to add. `collapse_is_lossless` (`src/llb/rag/vector_store/build.py`) names the rule,
  `build_store_parts` downgrades `collapse_duplicates` to False for such a strategy and logs it,
  and the meta records what the build actually did -- so the incremental refresh, which reads
  `collapse_duplicates` back from the meta, follows without a second rule. The duplicate stats are
  still measured, so a `late` store still reports what its repeats cost. The refresh path already
  refused text-keyed row reuse for `late` for the same reason; this is that rule on the build path.
- Each dropped copy is kept on the survivor as additive `metadata.duplicate_occurrences` -- its
  whole chunk record minus the (identical) text, so offsets, ids, and page/governance metadata
  survive -- plus `metadata.duplicate_count`. Every surviving chunk and every recorded occurrence
  is still a verbatim corpus slice.
- `chunk_hits_span` (`src/llb/rag/retrieval.py`) matches a chunk at EVERY place its text appears,
  so a gold span labeled on any copy still counts as retrieved and a citation still resolves to
  every document that carries the passage. In `parent_child` mode a collapsed child surfaces the
  parents of all its occurrences, which is the same parent set the tied duplicate children
  returned before.
- Where duplicates remain (`build-index --keep-duplicate-chunks`, or a backend that rounds its
  scores), `order_by_score` in `src/llb/rag/vector_store/build.py` breaks an exact score tie on the
  stable `chunk_id` instead of the backend's candidate order, on both the dense and the hybrid
  path -- so a tie is a documented property of the data, reproducible across rebuilds and
  backends.
- The incremental refresh undoes the collapse before its per-document merge and re-applies it
  after (`expand_duplicate_chunks` plus `resolve_duplicates` in
  `src/llb/rag/refresh/merge.py`), so a refreshed store still equals a from-scratch
  rebuild even when the document that happened to carry a survivor is the one edited or deleted.
  A repeated passage that is already indexed costs no embedding call when a new document
  introduces it again, and -- because the leftover fresh rows are keyed on stored TEXT, not on
  chunk position (`text_row_map`) -- that holds even when the EDITED document is the one that
  carried the survivor: its re-emitted copy recovers the row an unchanged document still holds
  instead of paying the encoder for text the store already has. The refresh reports the recovered
  rows as `n_reused_by_text` (`refresh-index` prints "N recovered by text").
- `store_meta.json` records `collapse_duplicates` and the measured `duplicates` stats (`n`,
  `unique`, `collapsed`, `duplicate_chunks`, `duplicate_share`, `groups`, `largest_group`,
  `intra_document_groups`, `cross_document_groups`), and `build-index` echoes them as its
  duplicate-rate line -- measured either way, so a store built with `--keep-duplicate-chunks` still
  reports what the repeats cost. `make build-rag-store` adds `dup%` / `maxdup` columns to its
  per-strategy table. The intra/cross split says WHERE a corpus's repetition comes from: page
  furniture shared across documents (`cross_document_groups`) versus a boilerplate block a single
  manual repeats section after section (`intra_document_groups`), which is a conversion-side
  property of that one document handled at ingestion by
  [intra-document repeat handling](../data-prep/ingestion-import.md#intra-document-repeated-block-handling---repeat-blocks).
- `compare-retrieval` prints each built lane's duplicate census beneath the recall table
  (`duplicate_census` in `src/llb/rag/comparison/run.py`), so a recall row is read next to how much of
  that lane's index is repeated text and whether the repeats are intra- or cross-document; a lane
  with no build meta (a graph or fake store) simply contributes no census row.
- The occurrences travel into the run bundle's `retrieval.jsonl`, so a lane that recomputes a
  metric from that sidecar agrees with the run that wrote it (see [the persisted retrieval
  record](persistence-and-execution.md#the-persisted-retrieval-record) under Persistence).

Durable evidence (2026-07-23, CUDA host, pinned e5-base, k=10, reports under
`$DATA_DIR/retrieval-noise-floor/<run>/`; the no-collapse baseline is the recorded
[measurement-floor](retrieval-metrics.md#measurement-floor---noise-floor) run):

| corpus (`size`) | lane | indexed chunks | fragile | recall@10 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| goods PDFs (200) | `recursive` | 4848 -> 3515 | 25 -> 1 | 0.653 -> 0.695 | 0.414 -> 0.465 |
| goods PDFs (200) | `sentence` | 5019 -> 3659 | 20 -> 0 | 0.621 -> 0.632 | 0.411 -> 0.465 |
| accepted PDF goldset (800) | `recursive` | 1124 -> 1120 | 0 -> 0 | 0.925 -> 0.925 | 0.852 -> 0.852 |
| committed UA fixture (800) | both | 311 -> 311 | 0 -> 0 | 0.976 -> 0.976 | 0.838 -> 0.838 |

The goods corpus stopped spending 27% of its index on text it already held, its floor fell from
+/-0.021 to +/-0.000, and recall/MRR ROSE rather than regressing: a top-10 that no longer repeats
one passage up to 58 times carries more distinct evidence. The two corpora with essentially no
duplicates reproduce every recorded number exactly, which is the check that collapse is a no-op
where there is nothing to collapse.

Every one of those goods collapse groups repeats INSIDE one document (measured:
`intra_document_groups` = all 494, `cross_document_groups` = 0). Collapse removes the index and tie
cost of that repetition but cannot fix it at the source -- the survivor is still returned for a
question about any section that carries the block. Handling those blocks at CONVERSION time
(`--repeat-blocks drop`) removes the later copies from the source and lifts recall@10 a further
+0.022/+0.034 above the collapse baseline on the shared item set; see
[intra-document repeat handling](../data-prep/ingestion-import.md#intra-document-repeated-block-handling---repeat-blocks)
for the option and its adopt-`drop` / reject-`anchor` verdict.

Tests: `tests/llb/rag/duplicates/test_duplicates.py` (collapse, occurrence metadata,
offset-exactness against the committed `samples/corpora/duplicate_chunks_uk_v1/` fixture, span
matching at every occurrence, exact expansion, the tie-break, and the parent expansion) and
`tests/llb/rag/duplicates/test_duplicates_store.py` (index budget, retrievability of every copy's
place, the fragility drop measured through `measure_noise_floor`, refresh-equals-rebuild when the
survivor's document is deleted, and the `late` exception -- every repeat indexed, the stats still
measured, a text-only strategy on the same corpus still collapsing, and the refresh following the
recorded flag) -- fake hashed-BoW and token-level embedders, no GPU. `collapse_is_lossless` itself
is covered in `tests/llb/rag/chunking/test_chunking_strategies.py`.

### Near-Duplicate Residue And The Collapse Tiers

Shipped (near-duplicate-chunk-collapse, `src/llb/rag/duplicates/tiers.py` and
`src/llb/rag/duplicates/residue.py`): collapse takes a TIER that decides when two chunk texts count
as one passage, and a measurement that says how much repetition a store still holds after it.

Tiers, cheapest first, each strictly coarser than the one before:

| tier | two texts are one passage when | loss-free |
| --- | --- | --- |
| `exact` (default) | they are byte-identical | yes -- the survivor's text IS every copy's |
| `normalized` | they share the corpus-conflict `hash` tier's normalized token stream (`llb.rag.vector_store.lexical.tokenize`: casefold, apostrophe unification, punctuation strip, whitespace collapse) | no |
| `masked` | `normalized` plus digit-run masking (`Сторінка 3` == `Сторінка 47`) | no |

- Selection: `build-index --duplicate-tier <tier>` (`make build-index DUPLICATE_TIER=`),
  `compare-retrieval --duplicate-tier <tier>` for the stores that lane BUILDS
  (`make compare-retrieval DUPLICATE_TIER=`), or the `duplicate_tier` run-config field. The tier
  lands in `store_meta.json` and in the measured `duplicates.tier`, so the build summary names what
  it measured ("byte-identical to" / "normalization-equivalent to" / "digit-masked-equivalent to").
- Reversibility holds at every tier: where a merged copy's text differs from its survivor's, the
  copy is recorded WITH its own text, so `expand_duplicate_chunks` still reconstructs the
  pre-collapse set exactly. It hands back no reusable embedding row for such a copy, so an
  incremental refresh that promotes a differing copy to survivor re-embeds it instead of inheriting
  a vector encoded from another wording -- a refreshed store still equals a rebuild under a coarse
  tier (`tests/llb/rag/duplicates/test_duplicates_store.py`).
- A chunk whose text has no word tokens at all (a rule line, a stray bullet) falls back to its
  verbatim text, so no tier ever merges on the absence of content.
- What a coarse tier gives up: only `exact` guarantees the survivor's text is a verbatim slice of
  every occurrence's offsets. `occurrence_spans` still resolves every place the passage appears --
  span matching is by offsets -- but the indexed and quoted TEXT is one representative wording.

`llb measure-duplicate-residue --store <dir>` (`make measure-duplicate-residue STORE=<dir>
[RESIDUE_THRESHOLDS=] [RESIDUE_EXAMPLES=] [RESIDUE_OUT=<json>]`) measures the residue of a BUILT
store without loading an embedder (it reads the persisted chunks and vectors), along two axes:

- text: what each coarser tier would still collapse, with the same intra/cross census;
- embedding: chunk pairs above each cosine band (default 0.999 / 0.99 / 0.95), the share of chunks
  with such a neighbour, and how many of those pairs each TEXT tier reaches -- the cross-tab that
  says whether a cheap normalizer can take the residue at all.

It also samples what a merge would actually do: the top-cosine pairs no text tier merges, and the
pairs ONLY digit masking merges (the page footer and the rate row are the same shape to it).

Durable evidence (2026-07-24, goods PDF corpus at `size` 200/30, pinned e5-base, k=10, n=95, the
same corpus/goldset/seed as the exact-collapse evidence above).

Residue left by exact collapse, per lane:

| lane | indexed | `normalized` would collapse | `masked` would collapse | chunks with a >=0.99 neighbour | those pairs / reached by `normalized` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `recursive` | 3515 | 26 (25 groups) | 311 (67 groups, largest 147) | 727 (20.7%) | 13105 / 26 |
| `sentence` | 3659 | 55 (48 groups) | 357 (68 groups, largest 221) | 772 (21.1%) | 23142 / 54 |

Retrieval per tier (same items, same k, same seed):

| tier | `recursive` recall@10 / MRR / fragile | `sentence` recall@10 / MRR / fragile | indexed |
| --- | --- | --- | --- |
| `exact` | 0.6947 / 0.46515 / 1 | 0.6316 / 0.46487 / 0 | 3515 / 3659 |
| `normalized` | 0.6947 / 0.46565 / 0 | 0.6316 / 0.46505 / 0 | 3489 / 3604 |
| `masked` | 0.6947 / 0.46565 / 0 | 0.6316 / 0.46505 / 0 | 3204 / 3302 |

The floor is +/-0.000 on every lane and every tier, and every question-type slice is identical
across the three tiers, so the only measured movements are the MRR digits and the fragile count.

Verdict: **`exact` stays the shipped default; `normalized` is available per corpus; `masked` is
not recommended even though this goldset cannot see its cost.**

- `exact` reproduced its recorded rows to the digit under the tier refactor (0.6947/0.6316, fragile
  1/0), which is the check that adding tiers changed nothing for stores that do not ask for one.
- `normalized` clears the stated gate -- recall unchanged (well inside the +/-0.000 floor), MRR
  +0.0005, and the one fragile `recursive` item removed -- but what it buys is that single item and
  0.7-1.5% of the index, paid for with the loss-free property. Worth enabling only where the
  residue measurement shows a materially larger group count than the 25-48 groups measured here.
- `masked` scores IDENTICALLY to `normalized` on every row and slice while removing ~9% of the
  index, and that is exactly the trap: the sampled merges show it merging `Поле має обмеження в 200
  символів` with `... 2000 символів`, `в 5 символів` with `в 10 символів`, and consecutive numbered
  steps -- genuine content differences alongside the real page footers (`**ЗСУ. Конфіденційно**
  **N**`, a group of 147 copies). No item of this 95-question set happens to need a merged row, so
  the retrieval metric reports no cost; the evidence against the tier is the sampled merge list,
  not the recall. Reject it on any corpus whose facts differ by one number.
- The residue that matters is NOT text-reachable: 20.7% of the exact-collapsed `recursive` chunks
  have a neighbour at cosine >= 0.99, and the `normalized` tier merges 26 of those 13105 pairs.
  Reaching the rest needs an embedding-side merge with a measured false-merge rate, which is
  forward work, not current behavior.

Tests: `tests/llb/rag/duplicates/test_duplicate_tiers.py` (the normalizer's grouping and digit
masking, the token-free fallback, the tier ladder on the committed
`samples/corpora/near_duplicate_chunks_uk_v1/` fixture, offset-exactness and per-copy text at a
coarse tier, exact expansion with no reusable row for a differing copy, and the residue report's
bands/samples over hand-built vectors) plus the two coarse-tier store tests in
`tests/llb/rag/duplicates/test_duplicates_store.py` -- fake embedders, no GPU. The fixture's
`Застереження` block additionally pins apostrophe-variant equivalence: a U+2019 copy normalizes onto
its U+0027 twin (see [hybrid retrieval](hybrid-retrieval.md#hybrid-retrieval-dense--bm25--rrf) for
the tokenizer rule and what it moved).

## Store Lifecycle: Dynamic Corpus Refresh

Shipped (dynamic-corpus-refresh): `llb refresh-index` (`make refresh-index CORPUS=<dir>
[GOLDSET=<jsonl>] [RETUNE_THRESHOLD=] [SKIP_GRAPH=1] [GRAPH_EXTRACTION=<jsonl>]`) updates the
built stores after corpus edits in time proportional to the changed documents instead of a full
rebuild, and tells the operator when the corpus has drifted enough that the tuned configuration
should be re-searched.

Manifest diff: `store_meta.json` records `doc_fingerprints` -- per-document hashes from
`corpus_doc_fingerprints` in `src/llb/prep/corpus/governance.py` (with `corpus_manifest.json`
present, each ok item's canonical row: content sha plus governance fields; hand-built corpora
hash each committed `.md`/`.txt` file, keyed by the same relative-path `doc_id` chunking uses).
A document's PDF citation sidecar (`pdf-<digest>.citations.json`, the page-provenance source
for `metadata.pages`) hashes into its fingerprint when one exists -- in both manifest and
hand-built modes and in the aggregate `corpus_fingerprint` -- so a sidecar-only regeneration
(page spans rebuilt while the text is unchanged) counts as a modified document and the refresh
re-annotates that document's chunks; docs without a sidecar keep the plain hash, so stores built
before this stay refresh-compatible. `src/llb/rag/refresh/diff.py` classifies every document as
added / modified / deleted / unchanged; a governance-only change (for example a new `acl_label`)
counts as modified so chunk metadata propagates.

Incremental update (`src/llb/rag/refresh/store_refresh.py`): unchanged documents keep their
chunk records and embedding rows verbatim (`FaissIndex.vectors()` reconstructs the stored
matrix; the adapter backends return their persisted `vectors.npy`), added/modified documents are
re-chunked (`chunk_corpus(only_docs=...)`) and re-embedded, deleted documents drop out of the
dense, lexical, and persisted-record paths. Annotation-only fast path: a modified document whose
re-chunked `(char_start, char_end, text)` grid reproduces the stored one exactly (sidecar-driven
page-span regeneration, governance-only manifest changes) rewrites its chunk records -- carrying
the re-annotated metadata -- but reuses every embedding row and its lexical postings instead of
re-embedding (`_annotation_only_sources`); `refresh-index` reports those rows as reused, not
embedded. The fast path applies only to the diff's modified class: added documents and the
legacy no-`doc_fingerprints` full refresh always start with fresh rows, and any real text edit
(including an equal-length in-place replacement, which keeps the span grid but changes chunk
text) re-chunks the document. Text-keyed reuse then recovers every fresh row whose text the store
already holds: `text_row_map` builds a `{stored chunk text -> row}` index once (references, not
copies, so it costs one entry per stored row), and `resolve_duplicates` reuses that row for any
leftover fresh unit with the same text -- so re-emitted page furniture, an unchanged chunk of a
modified document, and unchanged documents in a legacy full refresh all reuse their stored rows
regardless of which document now carries the text. It is only applied where a chunk vector is a
pure function of its text; the `late` strategy pools document context, so it passes `text_rows`
as `None` and re-encodes each changed document instead. The merged store preserves the exact
from-scratch build order, so a refresh is identical to a rebuild on the same corpus state. The
full-extra local suite proves the equivalence per store kind (FAISS, Chroma, Qdrant, LanceDB,
hybrid BM25, parent_child, graph, and the `late` chunking strategy via a token-level fake
embedder) over add/modify/delete fixture cases in
`tests/llb/rag/refresh/test_refresh_store_core.py`,
`tests/llb/rag/refresh/test_refresh_store_metadata.py`, and
`tests/llb/graph/test_graph_refresh.py`, plus annotation-only (sidecar regeneration) cases
asserting zero embedder calls in flat, hybrid, and parent_child modes and a same-span text-edit
guard. Both split refresh-store modules carry `pytest.mark.heavy_env` directly because Pytest
module marks do not propagate from their imported helper; `make ci-github` therefore deselects
these FAISS-dependent cases in its base `[dev]`-only environment.
`tests/llb/rag/duplicates/test_duplicates_store.py` covers the text-keyed reuse: the shared block is
recovered from the store's own vectors when the edited document is the one that carried its
survivor, and the refreshed store still matches a rebuild byte for byte. The
hybrid lexical side merges incrementally (`src/llb/rag/refresh/lexical_merge.py`): the old
postings invert back to exact per-chunk term counts, so unchanged chunks are never re-tokenized
or re-lemmatized. A `late`-strategy refresh re-runs `encode_store_vectors` for the changed
documents only (whole-document token pooling per doc), so kept rows stay verbatim there too.

Comparison-store refresh (`src/llb/rag/refresh/siblings.py`): `compare-retrieval` persists its
per-strategy candidate stores under `$DATA_DIR/llb/rag/<strategy>/` (including `hybrid/`).
`refresh-index` refreshes every such sibling through the same `refresh_vector_store` path --
each sibling diffs its own recorded fingerprints, refreshes into its own
`<strategy>/generations/<utc-ts>/`, and no-ops when already current (siblings refresh even when
the main store is a no-op, since they may have been built at an older corpus state). The main
store's `generations/` child is never treated as a sibling. A `compare-retrieval` rerun after
corpus edits therefore never serves stale sibling stores.

Immutable generations (`src/llb/core/store_generations.py`): a refresh never edits the live
store. It stages the refreshed store and atomically publishes it as
`$DATA_DIR/llb/rag/generations/<utc-ts>/` (`refreshed_from` recorded in its meta).
`RagStore.load` / `GraphStore.load` resolve the live store as the candidate with the newest meta
file among the base directory and its generations (ties prefer the generation), so a later
`build-index` into the base takes over again. Rollback = delete the newest generation directory.

GraphRAG refresh (`src/llb/graph/refresh.py`): `build-graph` persists its inputs
(`extraction.jsonl`, `ontology.json`) beside the store and records per-doc sha256
`doc_fingerprints` in the graph meta. A refresh keeps unchanged documents' extractions, takes
updated rows for changed documents from `--graph-extraction <jsonl>` (deletion-only refreshes
need none; missing rows refuse with the document list), rebuilds the graph deterministically,
and publishes a generation carrying its merged inputs so the next refresh chains. Diagnostic
community summaries are not carried over; re-run `build-graph --summarize` when needed.

Drift report (`src/llb/rag/refresh/drift.py`): after a refresh the command re-runs retrieval
validation (recall@k / MRR) over the configured gold set against the old and new stores and
writes `$DATA_DIR/refresh/<run-ts>/{drift.json,report.md}` with the per-metric deltas and a
`retune_recommended` flag when either absolute delta crosses `--retune-threshold` (default
0.05). Re-tuning itself stays an operator or orchestrator decision. A store built before this
feature has no `doc_fingerprints` and refreshes once as a full re-embed into a generation
(logged); it refreshes incrementally afterwards.

Semantic prefix tree (`src/llb/conflicts/semantic_tree/refresh.py`): the corpus-conflict audit
persists a centroid tree over the store's chunk vectors, and it consumes the same `ManifestDiff`
classes. Chunks of deleted and modified documents are removed, chunks of added and modified
documents are re-inserted at their nearest leaf, and centroids and radii are recomputed only along
the affected root-to-leaf paths -- nodes off those paths keep their exact geometry, so their bounds
stay valid without being touched. A refresh answers queries identically to a rebuild on the same
corpus state (asserted in CI); once more than `REBUILD_FRACTION` of the chunks have changed it
rebuilds instead, because patching stops paying. The tree meta pins the embedder model and
dimension: centroids are only meaningful in the space that produced them, so a store re-embedded
with a different encoder rebuilds rather than patches. Full behavior in [data
prep](../data-prep/conflict-detection.md#corpus-hygiene-conflict-detection-corpus-conflict-detection).
