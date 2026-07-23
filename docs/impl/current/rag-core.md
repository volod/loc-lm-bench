# RAG Core

The RAG core evaluates one model over a verified gold split:

```text
retrieve -> generate -> classify -> score -> aggregate -> persist
```

It is intentionally backend-neutral. Backends launch differently, but the evaluator talks to an
OpenAI-compatible chat endpoint and receives normalized response classes.

## Configuration

`src/llb/core/config.py` defines `RunConfig`, the typed object that flows through retrieval,
generation, scoring, telemetry, and the manifest. YAML configs and CLI overrides share the same
validation path. Unknown keys and invalid ranges fail before work starts.

`src/llb/core/paths.py` loads `.env`, honors `DATA_DIR`, and resolves relative paths from the project
root instead of the caller's current directory.

## Command Path

```bash
llb prep-models
llb list-models
llb build-index --vector-store faiss
llb validate-retrieval --k 10
llb run-eval --model llama3.2:3b --backend ollama
llb run-eval --config samples/configs/run_config_uk.yaml
llb run-eval --split calibration --worksheet calibration.csv
llb run-eval --score-semantic
llb run-eval --resume .data/run-eval/<timestamp>-<run-id>   # continue an interrupted run
```

Make targets wrap the common path:

```bash
make prep-models
make build-index
make validate-retrieval
make run-eval MODEL=llama3.2:3b BACKEND=ollama LIMIT=20
make compare-context-strategies MODEL=<m> BACKEND=<b> GOLDSET=<gs> CORPUS=<corpus-dir>
```

The Makefile defaults `GOLDSET` and `CORPUS` to the committed fixture so smoke runs do not require
network access or data regeneration.
For the local PDF-corpus Gemma 4 quickstart, see
[`docs/guides/quickstart/quickstart-pdf-corpus.md`](../../guides/quickstart/quickstart-pdf-corpus.md).
That flow builds
`.data/quickstart-pdf-corpus-rag/llb/rag/` from 19 converted PDFs: 13,211 recursive FAISS chunks
and 768-dimensional E5 embeddings (2026-07-02 build). A 4-document quick draft
(`QUICKSTART_PDF_DRAFT_DOCS=<4 ids>`, 70 unverified items, all citation-valid) scored
`recall@10=0.729`, `MRR=0.531` against that full-corpus index -- a true needle-in-haystack check;
the misses are questions phrased broadly enough that spans from OTHER doctrine documents outrank
the gold span. The PDF draft path now annotates `needle_items.jsonl` with `retrieval_rank` from the
full-corpus store; rows with a non-null rank are the retrieval-unique subset at the configured
top-k, and `pdf_ontology_report.json` records the unique-needle fraction. The matching GraphRAG
store lives under
`.data/quickstart-pdf-corpus-graph/llb/graph/` with 290 nodes, 159 edges, and 139 communities
from the same 4-document draft bundle.

## Standalone Closed-Service Runner

`src/llb/standalone/rag_squad_goldset.py` is a stdlib-only helper for operators who need to score a
closed RAG service outside the normal `llb run-eval` backend path. It reads SQuAD-shaped JSONL,
POSTs each `question` to `RAG_SERVICE_URL`, strips reasoning `<think>` blocks, and streams the
original row plus `predicted_answer`, `error`, `service`, and `latency_s` to an output JSONL file.

```bash
python src/llb/standalone/rag_squad_goldset.py INPUT.jsonl OUTPUT.jsonl --limit 10
```

The wire-format seam is intentionally narrow: edit `build_request()` and `parse_answer()` for the
remote service shape, or set `RAG_SERVICE_URL`, `RAG_SERVICE_NAME`, `RAG_API_KEY`,
`RAG_TIMEOUT_S`, and `RAG_RETRIES` in the environment. The helper is type-checked by the standard
`make ci` mypy pass.

## External Answer Log Scoring

`llb score-external-rag` / `make score-external-rag` reviews a JSONL file that already contains
answers from an external or closed RAG system. It does not launch the local RAG backend. It reads
the gold fields plus an answer field (`llm_answer`, `predicted_answer`, `model_answer`, or
`answer`), computes the same objective answer-correctness signals as `run-eval`, and opens an
interactive human scoring loop. Record state, objective scoring, CSV rendering, and orchestration
live in `src/llb/scoring/external_rag/{records,score,worksheet,run}.py`; aggregation, Markdown
reporting, and source mapping are explicit `external_rag_*` sibling modules. The terminal loop is
in `src/llb/scoring/external_rag_session/`; coverage lives in
`tests/llb/scoring/test_external_rag_score.py`.

The JSONL answer log is the session state. Each edit atomically writes `human_score_0_1`,
`human_decision`, `human_notes`, `human_corrected_answer`, and `human_status` back into the same
file, so partial sessions resume at the first unscored row. `EXTERNAL_RAG_CLEAR=1` / `--clear`
clears those human fields after confirmation. The card shown to the reviewer includes the
question, reference answer, gold source spans, raw answer, scored answer, first returned sources,
and error field.

Final artifacts are written only after all rows have a human score plus decision:

```text
<answered>.csv
<answered>.report.md
```

The CSV is sorted by `review_priority_rank` and includes the JSONL-backed human review fields. The
report records aggregate objective estimates, human decision counts, mean human score, split
estimates, common returned sources, actual scoring parameters, and improvement commands. A trailing
source footer in the answer text is stripped before objective scoring while the raw answer is
preserved in the CSV.

```bash
make score-external-rag EXTERNAL_RAG_ANSWERS=<answered-jsonl>
llb score-external-rag --answers <answered-jsonl> --answer-field predicted_answer
make score-external-rag EXTERNAL_RAG_ANSWERS=<answered-jsonl> EXTERNAL_RAG_CLEAR=1
```

This is an external-system diagnostic, not a certified local leaderboard.

### Source-span audit (external-rag-source-mapping)

When the answer log returns only provider-namespace source records (article ids, titles, URLs),
an operator-supplied mapping sidecar joins them onto benchmark corpus spans so retrieval evidence
can be audited, not only answer text:

```bash
make score-external-rag EXTERNAL_RAG_ANSWERS=<answered-jsonl> EXTERNAL_RAG_SOURCE_MAP=<map.jsonl>
llb score-external-rag --answers <answered-jsonl> --source-map <map.jsonl>
```

The sidecar (`.json` list, `.jsonl`, or `.csv`; lives beside the answer log or under
`$DATA_DIR/external-rag/<system>/`) maps provider keys to corpus locations: each record carries
`doc_id` (required), optional `char_start`/`char_end`, and at least one of `article_id`, `url`,
`article_title` (matched in that precedence order). `src/llb/scoring/external_rag_sources.py`
implements the audit; `tests/llb/scoring/test_external_rag_sources.py` covers it.

- A mapped source WITH a char range is scored by the same source-span metric as local retrieval
  (`llb.rag.retrieval.first_hit_rank` over the returned-source order): a span overlapping the
  item's gold spans is a hit.
- A mapping with only `doc_id` (typically title-keyed) can produce at most a doc-level match,
  flagged `source_hit_weak=true` -- weak evidence, never span proof.
- A returned source with no mapping counts into `source_unmapped_count` -- an audit gap reported
  separately from mapped retrieval misses.

The CSV gains additive columns (`source_hit`, `source_first_hit_rank`, `source_hit_weak`,
`source_mapped_count`, `source_unmapped_count`; absent without `--source-map`), and the report
gains a "Source-span audit" section with span-proof `recall@3` and MRR (weak hits and unmapped
sources reported beside them, never folded in). Without a sidecar the limitation stands: external
retrieval recall needs source records resolvable to corpus `doc_id`, `char_start`, `char_end`.

## Retrieval Store

`src/llb/rag/store.py` builds `RagStore`:

- chunks the corpus through `llb.rag.chunking`;
- embeds with the pinned multilingual E5 embedder;
- stores chunk records with exact source offsets;
- persists a vector index through the vector-store seam.

The default backend is FAISS. Chroma, Qdrant, and LanceDB use the same `VectorIndex` protocol in
`src/llb/rag/vector_index.py`.

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

Governance metadata (`src/llb/prep/corpus_governance.py`, `src/llb/rag/chunking/corpus.py`, and
`src/llb/rag/store.py`) is joined from `corpus_manifest.json` onto every chunk as additive
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

## Store Lifecycle: Dynamic Corpus Refresh

Shipped (dynamic-corpus-refresh): `llb refresh-index` (`make refresh-index CORPUS=<dir>
[GOLDSET=<jsonl>] [RETUNE_THRESHOLD=] [SKIP_GRAPH=1] [GRAPH_EXTRACTION=<jsonl>]`) updates the
built stores after corpus edits in time proportional to the changed documents instead of a full
rebuild, and tells the operator when the corpus has drifted enough that the tuned configuration
should be re-searched.

Manifest diff: `store_meta.json` records `doc_fingerprints` -- per-document hashes from
`corpus_doc_fingerprints` in `src/llb/prep/corpus_governance.py` (with `corpus_manifest.json`
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
legacy no-`doc_fingerprints` full refresh always embed fresh rows, and any real text edit
(including an equal-length in-place replacement, which keeps the span grid but changes chunk
text) still re-embeds. The merged store preserves the exact from-scratch
build order, so a refresh is identical to a rebuild on the same corpus state; CI proves the
equivalence per store kind (FAISS, Chroma, Qdrant, LanceDB, hybrid BM25, parent_child, graph,
and the `late` chunking strategy via a token-level fake embedder) over add/modify/delete fixture
cases in `tests/llb/rag/test_refresh_store.py` and `tests/llb/graph/test_graph_refresh.py`,
plus annotation-only (sidecar regeneration) cases asserting zero embedder calls in flat,
hybrid, and parent_child modes and a same-span text-edit guard. The
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

Semantic prefix tree (`src/llb/conflicts/tree_refresh.py`): the corpus-conflict audit persists a
centroid tree over the store's chunk vectors, and it consumes the same `ManifestDiff` classes.
Chunks of deleted and modified documents are removed, chunks of added and modified documents are
re-inserted at their nearest leaf, and centroids and radii are recomputed only along the affected
root-to-leaf paths -- nodes off those paths keep their exact geometry, so their bounds stay valid
without being touched. A refresh answers queries identically to a rebuild on the same corpus state
(asserted in CI); once more than `REBUILD_FRACTION` of the chunks have changed it rebuilds instead,
because patching stops paying. The tree meta pins the embedder model and dimension: centroids are
only meaningful in the space that produced them, so a store re-embedded with a different encoder
rebuilds rather than patches. Full behavior in
[data prep](data-prep.md#corpus-hygiene-conflict-detection-corpus-conflict-detection).

## Hybrid Retrieval (Dense + BM25 + RRF)

Shipped (hybrid-retrieval-uk): retrieval has the full hybrid shape Ukrainian enterprise corpora
need -- dense E5 plus lexical BM25 fused with weighted reciprocal-rank fusion, plus a
chunk-metadata filter seam -- so exact surnames, article/law numbers, codes, and abbreviations
stop losing to semantic-only search.

Modules:

- `src/llb/rag/lexical.py` -- pure-Python BM25 (`LexicalIndex`, in-repo)
  over the SAME offset-exact chunks the vector index holds; Ukrainian-aware token normalization
  on the LEXICAL side only (casefold, apostrophe-variant unification U+2019/U+02BC/`'`,
  punctuation strip); opt-in lemmatization via the base dependencies `pymorphy3` +
  `pymorphy3-dicts-uk`, collapsing cases/inflection to lemmas at index AND query time -- the stored
  chunk text stays byte-identical (unit-tested); `rrf_fuse` implements the weighted RRF
  (`score = w/(60+dense_rank) + (1-w)/(60+lexical_rank)`) with deterministic tie-breaks.
  Its generalized `weighted_rrf_fuse` accepts n ranked lists and non-negative weights. A
  zero-weight lane contributes neither score nor candidate membership, fixing endpoint weights
  that previously appended disabled-lane candidates when the active lane returned fewer than k.
- `src/llb/rag/filters.py` -- the chunk-metadata filter seam: `metadata_filter(doc_ids,
  heading_contains, page_range, acl_label)` builds a predicate over `doc_id` plus the
  page-metadata join's `metadata.headers` breadcrumb, `metadata.pages` range, and governance
  `metadata.acl_label`; `RagStore.retrieve(question, k, chunk_filter=...)` applies it BEFORE
  fusion/ranking (with a filter the whole index is scanned, so the cut is exact).
- `src/llb/rag/store.py` -- `mode="hybrid"` builds the lexical index beside the vector index;
  fusion runs inside `RagStore.retrieve`, so every dense `VectorIndex` backend
  (FAISS/Chroma/Qdrant/LanceDB) gains hybrid identically. The lexical index persists as
  `lexical_index.json` beside the FAISS artifacts and joins `store_meta.json`
  (`meta.lexical = {lemmatize, n_terms}`). Loading a hybrid store whose lexical file is missing
  refuses with a rebuild message, and `run-eval --retrieval-mode hybrid` over a dense-only store
  refuses too (`_load_store`); a non-hybrid config over a hybrid store serves dense-only.

Knobs (all `RunConfig` fields, hence in the manifest and the sweep cell fingerprint):
`retrieval_mode=hybrid`, `fusion_weight` (dense share of the RRF, default 0.5; 1.0 == dense
order, 0.0 == lexical order), `fusion_candidates` (per-side candidate depth, default 50), and
`lexical_lemmas` (index-time lemmatization, recorded in the store meta).

Commands:

```bash
make build-index RETRIEVAL_MODE=hybrid LEMMATIZE=1    # build-index --retrieval-mode hybrid --lemmatize
make run-eval MODEL=<m> RETRIEVAL_MODE=hybrid FUSION_WEIGHT=0.5
make compare-retrieval HYBRID=1 GOLDSET=<goldset.jsonl>
make sweep SWEEP_RAG_GRID="top_k=3,5;fusion_weight=0.4,0.6"
llb tune ...    # the Optuna space samples retrieval_mode=hybrid + both fusion knobs
```

`compare-retrieval --hybrid` embeds the corpus ONCE and scores four rows sharing that dense
index: `dense`, `hybrid` (BM25 + weighted RRF), `hybrid+lemmas` (a second, lemmatized lexical
index), and `dense+oracle-doc` -- a diagnostic row restricting candidates to each gold item's
`source_doc_id` through the filter seam, quantifying the recall headroom a PERFECT document router
would buy (never a scoring config).

The lemma normalizer is reused by the miss analysis: `topic_of` in
`src/llb/board/miss_analysis/classify.py` lemmatizes its heuristic topic key, so Ukrainian case
forms of one topic collapse into a single cluster instead of splitting across inflections.

Fixture: `samples/goldsets/exact_terms_uk/` -- a 40-entry near-identical Ukrainian orders
registry (order numbers, DSTU codes, surnames, amounts; ~41 recursive chunks) whose 8 items ask
for exact terms; the CI regression (`tests/llb/rag/test_hybrid_store.py`) proves hybrid strictly
beats a signal-free dense ranking there. Tests: `tests/llb/rag/test_lexical.py` (normalization,
BM25 determinism and tie-breaks, lemma matching, save/load), `tests/llb/rag/test_filters.py`
(doc/heading/page/ACL predicates), `tests/llb/rag/test_hybrid_store.py` (fusion order, weight
extremes, filter-before-fusion, refusal paths, config-knob application, byte-identical text), plus
grid/tuner coverage in
`tests/llb/cli/test_cli_models.py` / `tests/llb/optimize/test_tuner.py`.

Durable evidence (2026-07-08, real e5-base stores on the dev host, outside quick CI), via
`compare-retrieval --hybrid`:

- `samples/goldsets/ip_regulation_uk` (8 items, saturated fixture), k=10: all four rows hold
  recall 1.000 / MRR 1.000 -- hybrid is equal-or-better than dense on the committed goldset (the
  gate), and the fixture is too small to discriminate further.
- `samples/goldsets/exact_terms_uk` (8 exact-term items), k=10: recall ties at 1.000 but hybrid
  MRR 0.938 vs dense 0.713; at k=3 hybrid holds recall 1.000 / MRR 0.938 vs dense 0.875 / 0.688
  -- the strict exact-term win the lexical side exists for. `hybrid+lemmas` matched plain
  `hybrid` on both fixtures (exact numbers do not inflect). The oracle-doc row equals dense on
  these single-document corpora by construction (a doc filter is a no-op with one doc).

Durable evidence, full corpus (2026-07-10, hybrid-comparison-full-corpus on the CUDA host,
outside quick CI): dense vs hybrid over the verified 44-item quickstart-PDF accepted goldset
(5 documents, 1139 chunks, inflection-rich Ukrainian questions; k=10), with the `fusion_weight`
gridded across three runs:

| row | recall@10 | MRR |
| --- | ---: | ---: |
| dense | **0.955** | 0.740 |
| dense+oracle-doc (headroom) | 0.977 | 0.753 |
| hybrid w=0.5 (default) | 0.932 | 0.742 |
| hybrid w=0.6 | 0.932 | 0.750 |
| hybrid w=0.7 | **0.955** | 0.748 |
| hybrid+lemmas w=0.5 | 0.932 | **0.762** |
| hybrid+lemmas w=0.6 | 0.932 | 0.759 |
| hybrid+lemmas w=0.7 | 0.932 | 0.753 |

Fusion-knob verdict for this corpus: dense-only STAYS the default -- at the 0.5 default the
BM25 side actively costs recall (-0.023), and only a dense-heavy `fusion_weight=0.7` climbs
back to the dense recall while adding a small MRR gain (+0.008). The measured lemmatization
delta on an inflection-rich corpus is a real but MRR-only effect: +0.020 MRR at w=0.5 with
recall unchanged (the tiny-fixture zero was a corpus artifact, as predicted). The oracle-doc
router headroom row is finally non-degenerate on this multi-document corpus: perfect document
routing would buy +0.022 recall / +0.013 MRR -- modest, so a learned router stays unattractive
here. Operators who want hybrid for exact-term robustness (see the exact-term fixture win
above) should pin `FUSION_WEIGHT=0.7`; the end-to-end cross-check
(`make sweep SWEEP_RAG_GRID="fusion_weight=0.5,0.7"`) is worth running once a model roster
decision hangs on it.

## Graph-Vector Fusion Retrieval

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

### Fusion span identity (`graph_fusion_span_identity`)

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
where 1.0 is containment-only), recorded in the manifest fingerprint, and settable through
`run-eval --graph-fusion-span-merge-ratio`,
`make sweep SWEEP_RAG_GRID="graph_fusion_span_merge_ratio=0.25,0.5"`, and the evidence lane's
`GRAPH_FUSION_SPAN_MERGE_RATIO` grid. It is dead under `exact` (there is no partial overlap to
threshold), so the sweep grid expands `overlap` rows only and a non-default value extends a row
label as `/r<ratio>`. **The measured verdict is to pin 0.5 and not sweep it**, and it holds at two
chunk scales: at `chunk_size=800` the threshold decides essentially nothing (0.25 / 0.5 / 0.75 are
byte-identical on every row, because 99% of the graph spans touching a retrieved chunk are wholly
INSIDE it), and at `size=200` -- where a chunk and an entity mention are finally the same order of
magnitude -- it re-decides merges on up to a quarter of the questions yet still moves one headline
metric in one row, in 0.5's favor. See
[GraphRAG](graphrag-backend.md#span-merge-threshold-evidence) for the grid, the agreement table,
the overlap histogram, and
[the smaller-chunk re-run](graphrag-backend.md#does-the-pin-survive-a-smaller-chunk-size).

The knob rides `RunConfig`, the manifest fingerprint, `run-eval --graph-fusion-span-identity`,
`make sweep SWEEP_RAG_GRID="graph_fusion_span_identity=exact,overlap"`, and the sweep lane's
`GRAPH_FUSION_SPAN_IDENTITY` grid. `exact` remains the default: the measured adopt verdict for
`overlap` rests on a drafted multi-hop ledger, and the end-to-end run of the same two rows finds
the extra evidence is retrieval-only and costs measurable factoid answer quality -- see
[GraphRAG](graphrag-backend.md#span-identity-evidence) for both halves.

### Fusion candidate depth (`graph_fusion_candidates`)

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
`lane_weight / k > lane_weight / r`, so at least k candidates outrank it at every graph weight.
Only a span BOTH lanes return, with at least one of its ranks below `k`, can be promoted by depth.
That makes the knob's usefulness a property of the corpus AND of the span-identity policy above:
under `exact` the measured Ukrainian goods corpus shares a candidate in 2 of 95 questions and depth
changes nothing at all, while under `overlap` it shares one in 93 of 95 and every depth row moves
(see [GraphRAG](graphrag-backend.md#candidate-depth-evidence) and
[span identity](graphrag-backend.md#span-identity-evidence) for both measured verdicts). Depth is
therefore a live knob exactly when the identity rule lets the lanes agree.

`compare-retrieval` ranks backends at ONE graph weight; `compare-graph-fusion` sweeps the weight
and decides it on the multi-hop slice with uncertainty; `compare-answer-quality` then scores the
same items END TO END under two of those rows and compares the answers, which is what separates a
retrieval-only coverage gain from an answer-quality gain -- see
[GraphRAG](graphrag-backend.md#graph-vector-fusion-evidence) for all three lanes, their measured
CUDA-host evidence, and the artifact locations.

### Fusion question-type routing (`graph_fusion_router`)

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
[GraphRAG](graphrag-backend.md#measured-result-question-type-routing-keeps-the-gain-and-clears-the-factoid-loss).
The held-out sidecar-free calibration recommends no threshold change; see
[GraphRAG](graphrag-backend.md#sidecar-free-heuristic-calibration).

## Reranking And Context Order (rerank-context-order)

Shipped: the stage between retrieval and generation is tunable -- an optional local
cross-encoder reranker (retrieve `rerank_candidates`, rerank, keep `top_k`), a context-order
policy for how the kept chunks are laid into the prompt, and a lost-in-the-middle position
probe that names the per-model ordering recommendation with measured evidence.

Modules:

- `src/llb/rag/rerank.py` -- the reranker seam. `RerankingRetriever` wraps ANY retrieval
  backend exposing `.retrieve(question, k)` (flat / parent_child / hybrid stores and the
  GraphRAG store alike): it pulls `max(rerank_candidates, k)` candidates, scores every
  (question, chunk text) pair through an injectable `RerankScorer`, and keeps the `top_k`
  best -- each kept chunk carrying `rerank_score`, its original `pre_rerank_rank`, and a fresh
  contiguous `rank`; chunk text and offsets are never altered, so source-span recall@k / MRR
  score the reranked ranking on unchanged rules. The real scorer is `CrossEncoderReranker`
  (lazy sentence-transformers CrossEncoder, the `[rag]` extra; pinned default candidate
  `BAAI/bge-reranker-v2-m3`, multilingual). `maybe_wrap_reranker` applies the config knobs in
  `_load_store` (run-eval, every backend) and the tuner's `_build_store`, so reranking rides
  every existing seam. The wrapper records per-call retrieve/rerank wall-clock
  (`stage_latency`) plus cumulative means (`mean_stage_latency`).
- `src/llb/eval/common.py` -- `order_chunks` / `format_context(chunks, order=...)`: the
  context-order policy (`rank` = best-first, the default; `reverse_rank` = best-last) applied
  ONLY when chunks are laid into the prompt; `retrieved` state stays in rank order so
  retrieval metrics are unaffected. The `[i]` labels number PROMPT positions.
- `src/llb/eval/graph.py` -- the retrieve node applies the policy and records
  `retrieve_latency_s` / `rerank_latency_s` into the case state (journaled by the
  durable-eval-runner, carried into `scores.jsonl` rows); the manifest's `metrics` gains a
  `stage_latency` object (mean retrieve / rerank / generate seconds per case), so the
  reranker's precision gain is always weighed against its measured latency cost.
- `src/llb/eval/position_probe.py` -- `llb probe-context-position` (see
  [evaluation rigor](rigor-board-judge.md) for the probe contract and artifacts).

Knobs (all `RunConfig` fields, hence in the manifest and the sweep cell fingerprint):
`reranker` (HF cross-encoder id; `None` == off, the default), `rerank_candidates` (pool depth,
default 30), `context_order` (`rank` | `reverse_rank`, applies with or without a reranker).

Commands:

```bash
make run-eval MODEL=<m> RERANKER=BAAI/bge-reranker-v2-m3 RERANK_CANDIDATES=30 CONTEXT_ORDER=rank
make compare-retrieval RERANKER=BAAI/bge-reranker-v2-m3 GOLDSET=<goldset.jsonl> [HYBRID=1]
make probe-context-position MODEL=<m> BACKEND=<b> PROBE_K=5
make sweep SWEEP_RAG_GRID="rerank_candidates=0,30"    # 0 == reranker-off cell
llb tune --reranker BAAI/bge-reranker-v2-m3 ...       # adds on/off + candidate-depth axes
```

`compare-retrieval --reranker <id>` adds a `<row>+rerank` twin per compared row (the oracle-doc
headroom row excepted), so pre/post-rerank recall@k / MRR compare through the one
`evaluate_retrieval` metric, with mean per-query retrieve/rerank latency echoed per rerank row.
In the sweep grid a `rerank_candidates=0` point is the reranker-off cell; positive depths enable
the sweep-level `--reranker` model (default `BAAI/bge-reranker-v2-m3`). The tuner samples
`use_reranker` on/off and, only when on, the candidate depth (15..60) -- dead parameters are
never sampled.

Tests: `tests/llb/rag/test_rerank.py` (fake cross-encoder: candidate flow, kept set, rank bookkeeping,
stable ties, wrapper delegation, exact context ordering per policy, stage-latency capture and
manifest aggregation, config knob validation), `tests/llb/rag/test_compare_retrieval.py` (rerank twin
rows lift MRR through the shared metric; oracle row excluded), plus grid/tuner coverage in
`tests/llb/cli/test_cli_models.py` / `tests/llb/optimize/test_tuner.py`.

Durable evidence (2026-07-08, real `BAAI/bge-reranker-v2-m3` on the CUDA host RTX 4060 Ti,
outside quick CI), via `compare-retrieval --hybrid --reranker BAAI/bge-reranker-v2-m3`, k=10,
`rerank_candidates=30`:

- `samples/goldsets/exact_terms_uk` (8 exact-term items): the reranker lifts every base row to
  MRR 1.000 -- `dense+rerank` 1.000 vs dense 0.713, `hybrid+rerank` 1.000 vs hybrid 0.938
  (recall already saturated at 1.000). The cross-encoder recovers the exact-term precision that
  dense-only loses, even without the lexical side.
- `samples/goldsets/ip_regulation_uk` (8 items, saturated fixture): every row holds
  1.000/1.000 -- post-rerank uplift-or-tie holds (a tie; the fixture cannot discriminate).
- Measured latency cost: ~150 ms/query steady-state rerank wall-clock at pool depth 30 on the
  16 GB host (~300 ms on the first store while CUDA warms; the first-row mean absorbs the one-off
  model load). Retrieval itself stays ~13 ms/query, so the reranker multiplies retrieval-stage
  cost ~12x while staying far below generation cost.

Durable evidence, full corpus (2026-07-10, rerank-order-full-cohort on the CUDA host, outside
quick CI): rerank twin rows over the verified 44-item quickstart-PDF accepted goldset (1139
chunks, k=10, non-saturated), `BAAI/bge-reranker-v2-m3`:

| row | recall@10 | MRR | rerank ms/query |
| --- | ---: | ---: | ---: |
| dense | **0.955** | 0.740 | -- |
| dense+rerank (pool 30) | 0.909 | 0.859 | 783 |
| dense+rerank (pool 60) | 0.886 | 0.845 | 1434 |
| hybrid | 0.932 | 0.742 | -- |
| hybrid+rerank (pool 30) | 0.932 | **0.871** | 684 |
| hybrid+lemmas+rerank (pool 30) | 0.909 | 0.867 | 673 |

The full-corpus answer to "does the cross-encoder recover the dense recall shortfall?" is NO --
reranking is an MRR tool, not a recall tool, here: at pool 30 it lifts MRR by +0.119..+0.129
(0.740 -> 0.859 dense, 0.742 -> 0.871 hybrid) but DEMOTES gold chunks out of the top-10 on the
dense row (recall 0.955 -> 0.909), and deepening the pool to 60 makes recall worse still (0.886)
while doubling latency -- more candidates just give the cross-encoder more distractors to
promote. Steady-state rerank cost on the full corpus is ~700-780 ms/query at pool 30 (the tiny
fixture's ~150 ms was short-chunk-flattered), a real budget item beside ~5 ms retrieval. Verdict:
keep the reranker OFF by default on this corpus; switch it on (pool 30, ideally over hybrid,
where recall is not paid) only when first-hit rank dominates the harness, e.g. small `top_k`
generation prompts.

End-to-end cross-check (2026-07-10, `make sweep SWEEP_RAG_GRID="rerank_candidates=0,30"
RERANKER=BAAI/bge-reranker-v2-m3`, `llama3.2:3b` on ollama, accepted-goldset final split n=14,
k=5): the reranker DID lift in-run retrieval at this small k (recall@5 0.857 -> 0.929, MRR
0.685 -> 0.893 -- exactly the small-`top_k` regime the retrieval-side verdict carved out for
it), yet the end-to-end objective moved the other way: 0.378 [0.194, 0.584] rerank-off vs
0.312 [0.129, 0.515] rerank-on, overlapping CIs, at +0.96 s/query rerank latency (generation
itself is ~0.56 s/query, so reranking roughly doubles per-question cost). Retrieval uplift did
not translate into answer quality for this model at n=14 -- the off-by-default verdict stands
even in the reranker's best-case retrieval regime, and flipping it on should be justified with
end-to-end (not retrieval-only) evidence on the operator's own model + corpus. Run bundles:
`$DATA_DIR/run-eval/20260710T074826*` (off) / `20260710T074854*` (on) under the
`quickstart-pdf-corpus-rag` data dir, sweep id `rerank-crosscheck`.

## Query-Side Processing (uk-query-processing)

Shipped: an opt-in query lane between the user question and retrieval that measurably helps
Ukrainian queries while NEVER touching the stored corpus text (the query-side twin of the
index-side lexical normalization above). The raw question is always preserved -- only the
retrieval query is transformed -- and every step is honest: an A/B report attributes each step's
recall@k / MRR delta before anyone turns the lane on by default. Off by default (`query_prep`
empty is an exact no-op).

The `src/llb/rag/query_prep/` package is a pure, unit-testable pipeline of NAMED steps (no store, model,
or `[rag]` extra needed -- it reuses the pure tokenizer in `llb.rag.lexical`):

- `normalize` -- matching-side casefold; apostrophe-variant unification (U+2018 / U+2019 /
  U+02BC / grave / ASCII); Latin-typed Ukrainian back to Cyrillic; and safe Latin-look-alike
  repair inside mixed Cyrillic tokens. Canonical romanization preserves existing uppercase Latin
  acronyms and inserts a minimal ASCII apostrophe separator only where greedy digraph decoding
  would otherwise collide.
- `typos` -- deterministic corpus-vocabulary typo tolerance. The token vocabulary is built from
  the indexed corpus (`VocabularyContext.build` over `store.chunks`, whose `.tokens` is the same
  set `build_vocabulary` produces); a query token ABSENT from it is corrected to a nearby
  in-vocabulary token within Damerau-Levenshtein (OSA) distance 1 (2 for tokens over 8 chars).
  Tokens shorter than three characters are protected; candidate matching cannot cross
  alphabetic/numeric kinds; a token the corpus already contains is NEVER altered; and a numeric
  token is never "corrected" into a different one. Every correction is logged. An
  opt-in morphology guard (morphology-aware-typo-guard; `RunConfig.query_prep_typo_guard`,
  `--query-prep-typo-guard`, `QUERY_PREP_TYPO_GUARD=1`) additionally skips any OOV token pymorphy3
  recognizes as a valid Ukrainian word form (`llb.rag.lexical.load_uk_word_probe`): a grammatically
  valid inflection (`настанові`, `документами`) is not a misspelling and is left for the index+query
  lemmatization lane to match, while genuine misspellings stay unknown to the probe and are still
  corrected. Off by default so the pure edit-distance behavior remains explicitly selectable.
- **Ambiguity-aware restoration** (`query_prep/restore.py`) decides WHICH near candidate the
  `typos` step may take, and whether taking one is safe at all. Normalization is lossy -- Latin
  typing drops the soft sign and apostrophes, so `sut` inverts to the out-of-vocabulary `сут`,
  one edit from both `суть` and `суд`. Four constraints apply, in this order:
  1. **Surface compatibility (hard filter).** `normalization_provenance` maps every normalized
     token back to the single noisy token that produced it plus the edit `kind`; a candidate
     survives only when re-applying that transform reproduces the typed form
     (`surface_distance <= SURFACE_MAX_DISTANCE`, i.e. exactly). `суть` romanizes back to the
     typed `sut` and is kept; `суд` romanizes to `sud` and is refused. A token whose noise
     normalization already fully explains therefore cannot be rewritten by vocabulary correction
     at all. A replacement two different noisy tokens collapsed onto carries no constraint.
  2. **Short-token length lock.** At or below `AMBIGUOUS_TOKEN_MAX_CHARS` (4) an insertion or
     deletion candidate is refused, because at that length it is a different short word rather
     than a repair (`якв` -> `кв`, `зто` -> `то`). A transliteration provenance licenses the
     length change, since a dropped soft sign is exactly what romanization is known to lose.
  3. **Morphology, then local query context (ranking).** Candidates tied on edit distance are
     ordered by whether the morphology probe knows them as real word forms, then by whether they
     preserve the token's inflectional ending (`MORPH_SUFFIX_CHARS`), then by how often they share
     a corpus chunk with the query's rarest other in-vocabulary tokens
     (`VocabularyContext.cooccurrence` over up to `CONTEXT_MAX_ANCHORS` anchors), then
     alphabetically. Context is what separates `накат` from `наказ` for a query about waves.
  4. **Refusal on an unresolved tie.** When two candidates for a short token are equal on every
     signal above, the token is left unchanged instead of being resolved alphabetically.

  The constraints are always on inside the `typos` step (they only ever refuse or reorder a
  correction, never add one) and need no new knob; the morphology signal rides on the same opt-in
  probe as the guard, and the context index is built in the same pass as the vocabulary.
- `glossary` -- alias/glossary expansion. When the query mentions a known term (or a surzhyk /
  transliterated alias) the entry's other surface forms are APPENDED (the raw query is preserved),
  so retrieval catches the spelling the corpus actually uses. Sourced from a `query_glossary.json`
  built from a draft bundle's `prompt_dictionary_candidates.jsonl` (see
  [data prep](data-prep.md) query glossary).
- `rewrite` -- an optional local-LLM query rewrite through the run's backend endpoint seam
  (`eval.rag.query_rewrite` prompt). OFF by default and NEVER present unless explicitly requested;
  records both the original and rewritten query per case.
- `hyde` -- generates a short hypothetical answer through the same local endpoint and embeds it
  on the dense lane while retaining the processed user question for BM25 and graph linking. It
  does not alter the question sent to answer generation.
- `decompose` -- parses a bounded JSON or line-list response into at most five subqueries,
  retrieves every subquery, and deduplicates exact source spans with weighted RRF. A 2x
  original-query lane stabilizes ranking when the model over-decomposes a simple question.

Wiring: `src/llb/eval/graph.py` processes the question before retrieval and hands the structured
result to `query_prep/retrieval.py`. `RagStore.retrieve_queries` accepts separate dense and
lexical text; graph, fused, and reranking wrappers preserve that contract. The raw question stays
in state for generation. `scores.jsonl` and the durability journal carry `query_processed`,
`query_corrections`, `query_hypothetical_answer`, `query_decomposition`, and
`query_subqueries`, so normal and resumed runs preserve generated-query provenance. Journal
inclusion also fixes the earlier loss of deterministic query-prep provenance on resume.

Knobs (all `RunConfig` fields, hence in the manifest fingerprint): `query_prep` (ordered list of
`normalize` | `typos` | `glossary` | `rewrite` | `hyde` | `decompose`;
unknown/duplicated steps rejected at config validation), `query_glossary_path`, and
`query_prep_typo_guard` (refused at config validation unless the `typos` step is present).

Commands:

```bash
make build-query-glossary BUNDLE=<draft dir>            # -> <bundle>/query_glossary.json
make run-eval MODEL=<m> QUERY_PREP=normalize,typos,glossary QUERY_GLOSSARY=<json>
make validate-retrieval GOLDSET=<gs> QUERY_PREP=normalize,typos,glossary QUERY_GLOSSARY=<json> QUERY_PREP_AB=1
make validate-retrieval CONFIG=<yaml> GOLDSET=<gs> QUERY_PREP=hyde,decompose \
  QUERY_PREP_MODEL=<m> QUERY_PREP_BACKEND=ollama QUERY_PREP_AB=1 \
  QUERY_PREP_OUT=<report.json>
make bench-query-robustness MODEL=<m> BACKEND=<b> GOLDSET=<gs>
```

The `validate-retrieval --query-prep-ab` A/B report scores `baseline` then each cumulative step
with per-step recall@k / MRR deltas. Model steps use `--query-prep-model` and
`--query-prep-backend`; their completions and parsed subqueries are embedded per case in the JSON
report. Endpoint generators cache a question within one cumulative run, avoiding duplicate model
calls while preserving fixed-temperature results.

Tests: `tests/llb/rag/test_query_prep.py` (apostrophe and mixed-script repair, collision-safe
romanization, Latin acronym preservation, Damerau-Levenshtein transposition, typo correction that
never touches in-vocabulary, short, or cross-kind tokens + long-token distance 2 + deterministic
tie-break, deterministic alias expansion + glossary
build/round-trip, rewrite off-by-default, exact no-op when the lane is off, pipeline ordering +
dependency validation, A/B per-step delta over a fake store, retrieve-node raw-preservation and
processed-query wiring, HyDE dense/lexical separation, decomposition parsing/bounds/RRF span
deduplication, runner resolver dependency wiring, provenance mapping including the ambiguous
same-replacement case, per-kind surface distance, refusal of an incompatible nearest neighbor,
restoration of the romanization-compatible form, the short-token length lock and the
transliteration exemption from it, unresolved-tie refusal, context-driven candidate choice, and
both morphology preferences), `tests/llb/rag/test_store.py` (split hybrid
queries), and `tests/llb/executor/test_durable_resume.py` (generated-query journal round trip),
plus config validation in `tests/llb/core/test_config.py`.

The end-to-end noise benchmark, model evidence, and model-specific default recommendation live in
[evaluation rigor](rigor-board-judge.md#ukrainian-query-robustness-benchmark).

Durable evidence (2026-07-09, `intfloat/multilingual-e5-base`, flat FAISS over
`samples/goldsets/ip_regulation_uk/corpus`, k=5):

- Clean UA goldset queries (`samples/goldsets/ip_regulation_uk`, 8 items): baseline recall@5
  1.000 / MRR 1.000; `+normalize`, `+typos`, `+glossary` all hold 1.000/1.000 (+0.000 each). The
  fixture saturates (as the base-model comparisons here do), so the deltas are honestly zero --
  the typo step also "corrects" a few valid inflected query forms to the nearest corpus form
  (crude inflection matching, not a misspelling; the shipped lemmatization is the right tool for
  inflection), which the A/B would surface as a negative delta on a non-saturated corpus.
- Latin-typed variant of the same 8 queries (each Cyrillic word romanized -- e.g.
  `na yaki dvi velyki hrupy podilyayut pravo intelektualnoyi vlasnosti?`): baseline recall@5
  0.875 / MRR 0.812; `+normalize` (transliteration) RECOVERS to 1.000 / 1.000 -- a +0.125 recall /
  +0.188 MRR uplift. This is the mechanism's honest positive-delta demonstration.

Morphology-guard A/B (2026-07-10, morphology-aware-typo-guard on the CUDA host): over the
verified 44-item quickstart-PDF accepted goldset against the full-corpus 1139-chunk e5-base
store (k=10, non-saturated) the predicted regression is real and the guard removes it:

| stage | recall@10 | MRR | d(MRR) |
| --- | ---: | ---: | ---: |
| baseline | 0.955 | 0.740 | -- |
| +normalize | 0.955 | 0.748 | +0.009 |
| +typos (unguarded) | 0.955 | 0.736 | **-0.012** |
| +typos (guarded) | 0.955 | 0.748 | +0.000 |

Unguarded, the edit-distance step "corrected" valid inflections to the corpus surface form
(`настанові` -> `настанова`) and paid -0.012 MRR; guarded, those known word forms pass through
untouched (the lemmatization lane is the right tool for them) while genuine out-of-vocabulary
typos -- including the mixed-script `wеб` (Latin `w`) -> `веб` -- are still corrected, and the
step becomes MRR-neutral. Verdict: turn the guard on whenever the `typos` step is in use.

### HyDE and decomposition evidence

CUDA-host evidence (2026-07-21): `MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M` through Ollama,
`intfloat/multilingual-e5-base`, hybrid FAISS, k=10, and the held-out final split (n=13) of the
available verified 40-item accepted set against its full 1124-chunk store:

| stage | recall@10 | MRR | d(MRR) |
| --- | ---: | ---: | ---: |
| baseline | 0.923 | 0.814 | -- |
| +hyde | 0.923 | 0.833 | +0.019 |
| +hyde +decompose | 0.923 | 0.833 | +0.000 |
| baseline (isolated decomposition run) | 0.923 | 0.814 | -- |
| +decompose | 0.923 | 0.827 | +0.013 |

An initial equal-weight decomposition run regressed MRR to 0.699. Replaying its recorded
subqueries showed that adding the original question at 2x weight changed the result from harmful
to useful; the final independent run confirms +0.013 MRR, and the cumulative lane preserves the
larger HyDE gain. Recall is unchanged. Both steps remain opt-in. Reports, including endpoint and
per-case generated text, are under
`$DATA_DIR/query-prep-hyde-decompose/<run>/query_prep_ab_improved.json` and
`decompose_ab_improved.json`.

## Chunking Strategies

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
(`compare-retrieval --strategies ...`) builds one flat FAISS store per strategy over the SAME
corpus + pinned embedder (persisted under `$DATA_DIR/llb/rag/<strategy>/`) and ranks them by
recall@k / MRR on the gold set, so the best chunker is demonstrated per corpus, never assumed.
Add `NOISE_FLOOR=1` to learn how much of a chunker delta the corpus can actually resolve
([measurement floor](#measurement-floor-compare-retrieval---noise-floor)).
Tests: `tests/llb/rag/test_chunking_strategies.py` (offset round-trips, page-boundary alignment on the
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
[measurement floor](#measurement-floor-compare-retrieval---noise-floor) lane exists to make
visible.

### `size` Is A Hard Cap On Every Strategy

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

No recall regression, and the delta is not distinguishable from measurement noise: the
`recursive` control chunks are byte-identical across the two runs yet its recall moved by the same
+0.011, because the preceding lane's different batch shapes perturb the encoder output by ~5e-7
per dimension and that is enough to flip one borderline item at k=10 on 95 items. Repeat runs
within a code version reproduce byte-identically, so the drift is invisible to a naive repeat
check. `compare-retrieval --noise-floor` measures that floor directly and puts this corpus at
+/-0.021 recall@10 -- read any smaller retrieval delta on this set as noise
([measurement floor](#measurement-floor-compare-retrieval---noise-floor)).

Tests: `tests/llb/rag/test_chunking.py` covers the cap over the committed
`samples/chunking/goods_table_uk.md` fixture (a heading + markdown-table block with no sentence
terminator, 613 chars) -- every strategy stays within `size`, stays offset-exact, loses no
non-whitespace character, and the fixture itself is guarded against gaining a terminator.

## Embedder Conventions And Bake-off

Per-family query/passage conventions (`src/llb/rag/embedding.py`): a retrieval-tuned encoder scored
with the wrong instruction silently loses recall, so `Embedder` applies each model FAMILY's
convention (`embedding_family` resolves it, `apply_query_convention` / `apply_passage_convention`
are pure + unit-tested):

- `e5` (`intfloat/multilingual-e5-*`): `query:` / `passage:` prefixes (each with a trailing space);
- `bge-m3` (`BAAI/bge-m3`): NO instruction on either side (FlagEmbedding retrieval default);
- `bge` (other BGE retrieval lines, e.g. `bge-large-en-v1.5`): a query-only instruction;
- `plain` (paraphrase/STS models like `lang-uk/ukr-paraphrase-multilingual-mpnet-base`): symmetric,
  no prefix.

`llb compare-embeddings` (`src/llb/rag/embedding_bakeoff.py`; `make compare-embeddings`) answers
"which embedder for Ukrainian?" with evidence, not assumption. It builds one store per candidate
over the SAME corpus + chunking (each under its own family convention), scores recall@k / MRR by the
model-independent source-span metric (reusing `evaluate_retrieval`), and reports embed throughput,
index size, dimension, and device -- ending in a written recommendation the operator applies via
`build-index --embedding-model <winner>` + `RunConfig.embedding_model`. Artifacts:
`$DATA_DIR/compare-embeddings/<timestamp>/report.md` plus one saved store per candidate under
`stores/<model-slug>/`. Default local candidates: `intfloat/multilingual-e5-base` (current default),
`intfloat/multilingual-e5-large`, `BAAI/bge-m3`, `lang-uk/ukr-paraphrase-multilingual-mpnet-base`.
The store builder is an injectable seam, so scoring, ranking, the consent gate, and report shaping
are fake-store unit-tested (`tests/llb/rag/test_embedding_bakeoff.py`) with no GPU/FAISS/network.

Multi-objective tune (`llb tune --objectives ...`) may sample that same shortlist as a categorical
knob; the tuner `StoreRegistry` (`src/llb/optimize/store_registry.py`) rebuilds when the embedder
or chunking fingerprint changes, prewarms the shortlist for the base chunking shape before the
Optuna loop, fans out once per new chunking fingerprint, and may reload from
`$DATA_DIR/optuna/<study>/stores/`. It never reuses a store built under a different embedder. See
[evaluation rigor](rigor-board-judge.md#multi-objective-rag-tuner).

### Context budget

`RunConfig.context_budget` is an optional token budget that couples `top_k`, `chunk_size`, and
(for vLLM) `max_model_len`. When set, `fits_context` prunes configs whose estimated retrieved
prompt exceeds the budget, and multi-objective search samples the budget from
`{2048, 4096, 8192, 16384}` then sets `max_model_len` to that value on vLLM backends. Single-objective
`llb tune` leaves the budget unset unless the operator pins it in the run config.

Store/query embedder fingerprint: `store_meta.json` records the `embedding_model` a store was built
with, and `_load_store` refuses a run whose `config.embedding_model` differs
(`store_embedder_mismatch` in `src/llb/rag/store.py`), because a store is embedded and queried by
one encoder -- a mismatch would silently score the wrong model. A non-default-embedder store runs
normally with the embedder recorded in the manifest fingerprint.

Opt-in API row (open corpora only): `--api-model cohere/embed-multilingual-v3.0`
(`src/llb/rag/api_embedder.py`) embeds the corpus through a hosted API -- full egress, so it is
bake-off EVIDENCE ONLY (never usable as `RunConfig.embedding_model` for a scored run), refused
unless `--data-classification open`, gated on an interactive consent prompt naming the corpus, and
capped by `--max-usd` (`record_embed_cost` aborts when the running cost crosses the cap). Cohere's
`input_type` (`search_query` / `search_document`) maps onto the query/passage seam. litellm is
lazily imported and the embed callable is injectable, so the consent gate + budget arithmetic are
unit-tested with a fake client, no network in CI. The drafting-side pinned-E5 seams (ontology dedup,
semantic scoring, retrieval-uniqueness annotation) are deliberately NOT switched by this task.

Durable evidence, full corpus (2026-07-10, embedding-bakeoff-full-corpus on the CUDA host,
`LLB_EMBED_DEVICE=cuda`, outside quick CI): the four local candidates over the verified 44-item
quickstart-PDF accepted goldset (5 PDF documents, ~1.2 MB markdown, 1139 chunks at 800/120 --
the recall spread is finally NON-saturated at both cutoffs):

| model | recall@10 | MRR@10 | recall@20 | MRR@20 | dim | chunks/s (GPU) | index MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `intfloat/multilingual-e5-base` | **0.955** | 0.740 | 0.977 | 0.742 | 768 | 69 | 4.99 |
| `intfloat/multilingual-e5-large` | 0.932 | **0.795** | 0.977 | 0.798 | 1024 | 38 | 6.10 |
| `BAAI/bge-m3` | 0.932 | 0.753 | 0.955 | 0.755 | 1024 | 38 | 6.10 |
| `lang-uk/ukr-paraphrase-multilingual-mpnet-base` | 0.455 | 0.307 | 0.500 | 0.311 | 768 | 122 | 4.99 |

Winner for the 16 GB host: `intfloat/multilingual-e5-base` (the current default) -- it holds the
highest recall@10 (the gate metric; the score an operator's answers are capped by), embeds ~1.8x
faster than the 1024-dim pair, and builds the smallest index. e5-large trades a small recall@10
loss (-0.023) for the best early ranking (MRR 0.795 vs 0.740) and ties e5-base at recall@20 --
pick it only when a downstream reranker or a small `top_k` makes first-hit rank the binding
constraint. bge-m3 trails e5-large on both axes at the same cost, and the paraphrase/STS
`lang-uk` model collapses to 0.455/0.500 recall on a real corpus (the tiny-fixture 1.000 was
saturation, not quality) -- the "paraphrase objective loses to retrieval-tuned encoders"
hypothesis is supported by this corpus. Embed VRAM peaked ~4 GB (sequential model loads),
so every candidate fits the 16 GB host with a co-resident judge stopped; steady-state GPU
throughput at 1139 chunks is no longer cold-load-dominated. Reports:
`$DATA_DIR/compare-embeddings/20260710T044652*/report.md` (k=20) and
`.../20260710T044914*/report.md` (k=10).

## Context Ablation: Does RAG Pay For Itself? (rag-vs-long-context-ablation)

A leaderboard row says how well a model answers WITH retrieval; it never says how much of that
score retrieval bought. `llb compare-context-strategies` (`make compare-context-strategies`)
scores ONE item set end to end under three context lanes and reports the differences.

`RunConfig.context_strategy` selects the lane and is recorded in the manifest fingerprint like
every other knob, so a lane's bundle is reproducible from its own config
(`make run-eval CONTEXT_STRATEGY=<lane>`):

- `rag` (default) -- retrieve as configured. This is the leaderboard row; the other two are
  DIAGNOSTICS and never rank a model.
- `closed_book` -- no context at all. `src/llb/eval/context_ablation/sources.py` supplies an empty
  context and swaps in the `eval.rag.closed_book` prompt, which asks the model to answer from its
  own knowledge (the RAG system prompt would push it to abstain). The empty context deliberately
  does NOT raise `retrieval_miss`: that status short-circuits generation, and a lane that never
  calls the model measures nothing.
- `long_context` -- the item's whole gold source document(s) laid into the prompt as one
  offset-exact chunk per document, with the SAME generation prompt as `rag`, so the delta is
  attributable to the context and not to prompt wording. The lane is oracle-grounded (it reads the
  item's own gold `doc_id`s), which makes it a ceiling, not a shippable retrieval policy.

Budget and skips: the lane resolves the model's usable window ONCE per run --
`resolve_model_spec` looks the served artifact up through `candidate_sources`, so an Ollama GGUF
tag resolves to its roster entry priced at the right quant -- and each item is checked with
`fits_context_chars` (`src/llb/optimize/tuning_space.py`, the same arithmetic as `fits_context`).
An item whose document does not fit terminates as `context_overflow`, a new pre-generation status
in the shared taxonomy: no model call, no truncation. A truncated document is a different and
unstated retrieval policy, so crediting its answer to "long context" would measure whichever slice
survived the cut. Without a manifest entry for the model, only an explicit `context_budget` /
`max_model_len` can bound the prompt, so an unlisted model skips nothing rather than everything.

The comparison (`src/llb/eval/context_ablation/`) is pure and file-driven: it consumes canonical
`scores.jsonl` rows, aligns them with `llb.eval.paired_cases` (shared with
`compare-answer-quality`), and reuses the fusion-evidence paired bootstrap and per-slice reporting,
so the artifact reads beside the retrieval sweep. It reports:

- `retrieval_uplift` = `rag - closed_book`, paired per item -- how much of the RAG score retrieval
  paid for.
- `long_context_delta` = `long_context - rag` -- whole-document stuffing versus chunked retrieval.
- `long_context_delta_fitting` -- the same delta over items the lane did not skip, emitted only
  when something WAS skipped. A skipped item scores zero, so the all-items delta would otherwise
  read a document that never reached the model as a long-context loss; the VERDICT reads the
  fitting delta when it exists.
- A per-item contamination flag: the closed-book answer already matches the reference (`exact` or
  `contains` is 1.0). Items the model answers with no evidence were never a retrieval problem, and
  a corpus full of them makes any uplift look small for reasons unrelated to retrieval.

Verdicts, in check order: `long_context_wins` | `rag_pays_off` | `retrieval_inconclusive` |
`no_retrieval_gain` | `no_evidence`. Every gate reads the paired INTERVAL, never the point
estimate. Artifacts: `$DATA_DIR/context-ablation/<run>/{report.md,comparison.json}`, plus one
ordinary `run-eval` bundle per (lane, split) under `$DATA_DIR/run-eval/`. CI drives all three
lanes over fake bundles and the committed fixtures
(`tests/llb/eval/test_context_ablation.py`), no backend or GPU.

### Context-ablation evidence

Durable evidence (2026-07-22, CUDA host, Ollama, committed UA fixture
`samples/goldsets/ua_squad_postedited_v1/` -- 82 verified `final` items, 250-document corpus,
311 chunks at 800/120, `top_k=5`, `DATA_DIR=.data/context-ablation-host`):

| model | closed_book | rag | long_context | retrieval uplift | long-context delta | closed-book matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MamayLM-Gemma-3-12B-IT v2.0 GGUF Q4_K_M | 0.160 | 0.501 | 0.643 | +0.340 [+0.262, +0.423] | +0.142 [+0.083, +0.206] | 10/82 (12.2%) |
| Lapa v0.1.2-instruct GGUF Q4_K_M | 0.100 | 0.496 | 0.576 | +0.396 [+0.314, +0.484] | +0.080 [+0.036, +0.133] | 12/82 (14.6%) |

Both models return `long_context_wins`, and both agree on the shape of the result:

- Retrieval pays for itself, decisively. The uplift interval is far clear of zero for both models
  (sign-test p<0.001, 50/6/26 and 59/3/20 item wins/losses/ties). RAG is not decoration on this
  corpus.
- Whole-document stuffing still beats chunked retrieval, by a smaller but separable margin. That
  is expected here and is NOT an argument to ship long context: SQuAD-derived documents are ~1.5k
  characters, the lane is oracle-grounded on the item's own gold document, and `rag` retrieval was
  already near-ceiling (`recall@5=0.951`). The measured gap is what the retrieval layer still
  loses to chunk boundaries when the right document is known for free.
- Roughly one item in eight is answered correctly with no context at all -- parametric knowledge
  or contamination of a public post-edited SQuAD set. Any uplift on this fixture is therefore
  measured against a baseline that is not zero.

Skip path, measured (same model and item set, `context_budget: 1250` to force overflow):
28/82 items skipped, and the two populations diverge exactly as designed -- all-items
`long_context_delta` reads `-0.085 [-0.188, +0.018]` (the 28 skips score zero) while
`long_context_delta_fitting` over the remaining 54 reads `+0.165 [+0.091, +0.250]`. The verdict
reads the fitting delta, and the report carries both.

Reproducibility, measured: the `rag` lane's bundle is byte-identical to a plain `run-eval` of the
same configuration (all 82 items: same answers, same per-case scores), which is the check that the
lane machinery adds nothing to the leaderboard path. The `rag` and `long_context` lanes reproduce
exactly across runs; the `closed_book` lane does NOT -- 11/82 answers differed between two
identical invocations (lane mean 0.160 vs 0.153), because an ungrounded prompt leaves a much
flatter next-token distribution for GGUF kernel nondeterminism to flip. The drift is well inside
the uplift interval half-width (~0.08) and changed no verdict, but a closed-book number is a
noisier measurement than a grounded one and should be quoted with that in mind.

Reports: `$DATA_DIR/context-ablation/20260722T142639Z/` (MamayLM),
`.../20260722T143030Z/` (Lapa), `.../20260722T143459Z/` (the budget-constrained skip run).

## Retrieval Metrics

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

This metric is not a model-ranking axis. It answers whether the retrieval layer is able to surface
the evidence the model needs. If retrieval is poor, answer quality is capped by context quality.

### Measurement Floor (`compare-retrieval --noise-floor`)

`recall@k` / `MRR` are reported to three decimals, and the floor under those decimals is a
property of the CORPUS, not zero by default. `src/llb/rag/noise_floor.py` measures it:
`make compare-retrieval ... NOISE_FLOOR=1` (`compare-retrieval --noise-floor`,
`NOISE_FLOOR_REPLICATES=` / `--noise-floor-replicates` to change the replicate count) retrieves a
`3k` candidate pool once per lane, perturbs every candidate score by `N(0, 1e-6)`, re-ranks,
keeps the top k, and reports the band the metric spans over 64 seeded replicates plus the
worst-lane `floor` to read every delta against. The replicates only re-sort a cached pool, so the
whole measurement costs one extra retrieval pass per lane; the seed is stable per lane
(`crc32` of the label, never the salted `hash()`), so a report reproduces byte-identically.

Why `1e-6`: two processes that built BYTE-IDENTICAL chunks on this host produced dense vectors
differing by up to 5.4e-7 per dimension -- the encoder's kernels depend on the batch shapes it
saw earlier in the process, so the lane built BEFORE this one changes its output -- which moved
the cosine scores by up to 6.0e-7 (mean 1.3e-7). Repeats WITHIN one process are byte-identical,
so a naive repeat check reports a spread of zero and never sees the drift. The default rounds the
measured maximum up and perturbs every candidate independently, so the reported floor is
deliberately conservative: a delta that clears it is not numeric noise.

Each lane also reports `fragile N/n` -- items whose rank-k and rank-(k+1) candidates sit within
the jitter, so their top-k membership is decided by noise or by the backend's arbitrary order at
an exact tie. That count explains the band's width and is the number to act on.

Measured floors (CUDA host, pinned e5-base, k=10, `sentence` vs `recursive`; reports under
`$DATA_DIR/retrieval-noise-floor/<run>/`):

| corpus | n | chunk `size` | duplicate chunks | fragile | floor recall@10 | floor MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| converted Ukrainian goods PDFs | 95 | 200 | 37.7% | 25/95 | +/-0.021 | +/-0.018 |
| committed `ua_squad_postedited_v1` (final split) | 82 | 800 | 0.0% | 0/82 | +/-0.000 | +/-0.000 |
| accepted converted-PDF goldset | 40 | 800 | 0.5% | 1/40 | +/-0.000 | +/-0.000 |

The floor tracks DUPLICATE CHUNKS, not gold-set size. The goods corpus at `size=200` has 37.7% of
its chunks byte-identical to another chunk (repeated page furniture and table boilerplate in
converted scanned manuals; the largest identical group is 58 copies for `recursive` and 72 for
`sentence`). Identical text embeds to an identical vector, which scores an exact tie, which the
backend breaks by candidate order -- so a quarter of that corpus's items have a top-10 membership
that no retrieval property decides. The two corpora with essentially no duplicates have a floor of
exactly zero, and their deltas can be read at face value.

Verdicts re-read against the measured floors:

- Goods PDFs at `size=200`: `recursive` leads `sentence` by 0.032 recall@10, just outside the
  +/-0.021 floor (the two bands touch at 0.621), so the recall ranking is at the edge of what
  this set resolves; the MRR gap of 0.003 is far inside the +/-0.018 floor and means nothing.
  The floor also covers the between-process drift that motivated it: the `recursive` control moved
  0.642 -> 0.653 across two processes on byte-identical chunks, and both values sit inside its
  measured 0.621-0.663 band ([the `size` cap evidence](#size-is-a-hard-cap-on-every-strategy)).
- Committed UA fixture and the accepted PDF goldset: floor 0.000, so their recorded recall/MRR
  deltas are not numeric noise. They remain subject to SAMPLING uncertainty, which is a separate
  question the paired-bootstrap lanes answer -- a 0.022 recall delta on a 44-item set is under one
  item either way.

The floor is opt-in, so every existing comparison row is unchanged when it is not asked for.
Tests: `tests/llb/rag/test_noise_floor.py` (zero floor on separated scores, a full 0.0-1.0 band
when the cut sits on a tie, the fragility count, per-lane seeding and reproducibility, the
unscored-lane skip, and the ASCII rendering) over fake stores -- no FAISS, no GPU.

The default store retrieves dense-only (cosine over the pinned E5 embedding). Measured against
the gate, dense-only passes on the committed fixture (`recall@10=0.980`) but falls short on the
real full-corpus PDF index (`recall@10=0.729`, see the quickstart note above), so dense-only has
NOT been proven sufficient for a real Ukrainian corpus. Hybrid retrieval (see Hybrid Retrieval
above), cross-encoder reranking (see Reranking And Context Order above), and the ordered
`rag/query_prep/` pipeline are the available retrieval levers.

## Generation Graph

`src/llb/eval/graph.py` builds the retrieve-generate flow. LangGraph is imported only when the
graph is built. The graph records one status per case: `ok`, `empty`, `malformed`, `refusal`,
`timeout`, `backend_error`, `retrieval_miss`, `context_overflow`, or another typed failure from the
shared taxonomy. `retrieval_miss` and `context_overflow` are the pre-generation statuses
(`eval_common.PRE_GENERATION_STATUSES`): the prompt is never sent, so the answer stays empty and
the case scores zero rather than being quietly repaired into a different prompt.

Two optional seams keep diagnostic lanes out of the retrieval path itself. `context_source`
replaces the retrieve node's store lookup with a `RagState -> RagState` closure that supplies its
own context, and `template_id` overrides the generation prompt; both are resolved from
`RunConfig.context_strategy` in `_default_runner_fn` and are how the context ablation runs without
special-casing anything downstream (see Context Ablation above).

`src/llb/backends/openai_client.py` normalizes endpoint failures. Backend launchers own process
lifecycle and readiness checks.

## Scoring

`src/llb/scoring/correctness.py` computes objective correctness using normalized token-F1 with
exact and contains helpers. `--score-semantic` records a pinned-embedder cosine signal for
paraphrases and morphology; it is kept separate from the objective unless a ranking policy
explicitly uses it.

`src/llb/scoring/judge/model.py` owns the calibration gate and outcome policy;
`src/llb/scoring/judge/scorer.py` normalizes scores and handles empty answers; and
`src/llb/scoring/judge/deepeval_adapter.py` runs the optional local DeepEval integration. The judge
enters ranking only when the caller supplies a calibration rho that clears the trust threshold.
Otherwise it is diagnostic and objective correctness ranks alone.

`src/llb/scoring/aggregate.py` produces leaderboard rows. The policy favors quality first, then
throughput, then lower VRAM when telemetry is available.

### Groundedness and citation metrics (groundedness-citation-metrics)

Shipped: four answer-side signals that go beyond reference-answer overlap, all deterministic and
additive -- they never change the headline objective (they stay separate columns until a ranking
policy explicitly adopts them). `src/llb/scoring/groundedness.py` is a pure, dependency-free scorer
(no RAGAS, no frontier judge); the calibration-gated judge's faithfulness stays the optional
secondary groundedness signal.

- Groundedness fraction (`--score-groundedness`): the share of the answer's sentence-ish claims
  SUPPORTED by any retrieved chunk via token-overlap matching (a claim is supported when
  `GROUNDEDNESS_SUPPORT_THRESHOLD`=0.6 of its content tokens appear in a chunk). A fully-supported
  answer scores 1.0; an answer whose claims are absent from the context scores near 0.0.
- Citation validity + hallucinated-citation rate (`--cited-answers`): swaps in the
  `eval.rag.cited_answer` generation prompt (requires `[i]` chunk citations, reusing the numbered
  format `format_context` emits) and validates each citation against the chunk it points at, in
  PROMPT-LAYOUT order (so `reverse_rank` renumbering is respected). A citation whose in-range chunk
  lacks the claim is flagged invalid (lowers validity); a citation whose index is out of range is
  hallucinated.
- Citation coverage (`--cited-answers`, citation-coverage-metric): the share of countable claims
  (>= `MIN_CLAIM_TOKENS` content tokens, the same rule groundedness counts by) that carry ANY
  `[i]` citation, right or wrong. Validity alone collapses two failures into one low number -- a
  model that emits NO citations and a model that cites the WRONG chunk both score 0.0 validity
  (the durable llama3.2:3b run below made that concrete). Coverage separates them: coverage 0.0 =
  an instruction-following gap (does not cite); coverage high with validity low = a grounding gap
  (cites, but points at the wrong chunks). Reported beside validity, fully independent of it.
- Insufficient-context abstention probe (`--insufficient-context-probes <n>`,
  `src/llb/eval/insufficient_context.py`): re-runs a seeded sample of gold items with every chunk
  overlapping their gold spans EXCLUDED from retrieval (through the shipped chunk-metadata filter
  seam). Correct behavior is an explicit abstention (`llb.eval.common.is_abstention` = refusal OR an
  insufficient-context marker), scored as abstention accuracy. Probe rows live in `probes.jsonl` (+
  `insufficient_context_report.md`), NEVER in `scores.jsonl`, so they cannot enter the plain
  correctness aggregates.

Per-case fields land in `scores.jsonl` (`groundedness`, `citation_validity`, `citation_coverage`,
`hallucinated_citation_rate`, `n_citations`); their means plus `abstention_accuracy` / `n_probes`
land in the manifest `metrics`, echoed as the run's `answer-side:` summary line. Config knobs
(`cited_answers`, `score_groundedness`, `insufficient_context_probes`) are recorded in the manifest
fingerprint. `RunConfig` toggles are off by default, so pre-existing bundles keep their shape.

Modules/tests: `src/llb/scoring/groundedness.py`, `src/llb/eval/insufficient_context.py`, the
`eval.rag.cited_answer` template, `ScoreOptions` in `src/llb/executor/cases.py`;
`tests/llb/scoring/test_groundedness.py` (fully/partially/unsupported groundedness with zero
cross-class leakage, valid/flagged-invalid/hallucinated citations, coverage separating no-citation
from wrong-citation at equal validity, abstention markers, cited-answer prompt wiring, per-case
scoring + context-order-aware citation numbering, manifest mean coverage) and
`tests/llb/eval/test_insufficient_context.py` (gold exclusion, seeded sampling, abstention
accuracy, transport-error exclusion).

Durable evidence (2026-07-09, `llama3.2:3b` on Ollama, `intfloat/multilingual-e5-base` flat FAISS
over `samples/goldsets/ip_regulation_uk`, final split n=4, `--cited-answers --score-groundedness
--insufficient-context-probes 4`): mean groundedness 0.625 (per-case 1.0 / 0.5 / 1.0 / 0.0);
citation validity 0.000 with hallucinated-citation rate 0.000 -- the 3B model largely IGNORED the
`[i]` citation instruction (mostly emitted no citations), so validity is dominated by "did not cite"
rather than "cited wrongly"; abstention accuracy 0.000 -- on all four probes the model FABRICATED an
answer (even citing non-existent chunks) instead of abstaining when its gold evidence was removed.
Honest, unflattering evidence that a small model's answer-side grounding discipline is weak -- exactly
the axis these metrics expose beyond a passing recall@k.

## Backends

`BackendLauncher` is the core seam:

- `OllamaLauncher` talks to a pre-existing Ollama daemon;
- `VllmLauncher` starts and stops `vllm serve`;
- `LlamaCppLauncher` starts and stops `llama-server`.

All serve through an OpenAI-compatible base URL. When a launcher owns a subprocess, startup logs are
preserved on failure.

## Persistence

`src/llb/tracking/manifest.py` writes canonical run artifacts first:

```text
$DATA_DIR/run-eval/<timestamp>-<run-id>/
  manifest.json
  scores.jsonl
  retrieval.jsonl
```

Parquet is used when `pyarrow` is available; JSONL is the portable fallback. The bundle is staged
in a hidden sibling directory and atomically renamed when canonical files are complete. MLflow
mirroring runs after canonical persistence and is best-effort.

Per-case score rows record `retrieval_hit` and `first_hit_rank`. `retrieval.jsonl` stores bounded
retrieved chunk text plus source-span coordinates for miss analysis and observability;
`src/llb/executor/cases.py` constructs both the persisted records and the in-process retrieval
pairs used by aggregate metrics and judge records.

## Executor

`src/llb/executor/runner.py` orchestrates one run. It filters unverified items, loads the selected
retrieval backend, executes cases, collects optional telemetry, writes artifacts, mirrors to MLflow,
and prints the row.

`run_eval(..., verified_only=False)` is the one documented exception to the unverified filter. It
exists so a diagnostic lane can score exactly the item set a drafted-grounded retrieval sweep
measured (a drafted multi-hop slice has no accepted counterpart until a reviewer produces one), and
it is deliberately hard to reach by accident: no default path sets it, `run-eval` itself has no
flag for it, and the resulting manifest records `config.item_grounding: drafted` so the bundle is
self-describing. The only caller is `compare-answer-quality --include-drafted`
([GraphRAG](graphrag-backend.md#answer-quality-evidence)); a leaderboard run never uses it.

Isolation and GPU safety live outside the scoring path:

- `src/llb/executor/vram.py`: basic reclaim checks;
- `src/llb/executor/contention.py`: pre-launch vLLM contention guard;
- `src/llb/executor/isolation.py`: process-per-cell sweep and cooldown primitive.

## Durability

`src/llb/executor/durability.py` makes a run survive endpoint flaps, a launcher-owned backend
crash, and host restarts, so a long campaign does not lose hours of model calls to one blip. Three
recovery layers wrap the per-case loop:

- **Per-case retry.** A transient transport failure -- the typed status `timeout` or
  `backend_error` -- retries with capped exponential backoff (`--max-case-retries`,
  `--retry-backoff-s`). A scored answer or any non-transport terminal status (`ok`, `empty`,
  `malformed`, `refusal`, `retrieval_miss`) is a real outcome and is never retried.
- **Journal + resume.** Each completed case appends its terminal state to an append-only
  `cases.progress.jsonl` (keyed by `item_id`) in the staging dir, beside a
  `cases.progress.meta.json` sidecar that pins the config-fingerprint and goldset digests.
  `llb run-eval --resume <run-dir>` (Make: `RESUME=<run-dir>`) reuses the journaled cases instead
  of re-spending their model calls and runs only the remainder; a resume whose config, goldset, or
  split no longer matches the sidecar is refused. Everything downstream of the raw terminal state
  (scoring, retrieval pairs, judge records) is recomputed deterministically, so a resumed run's
  per-case scores are identical to an uninterrupted one -- verified across a real two-process kill
  (`os._exit` mid-run, fresh process resumes) as well as the committed-fixture unit harness.
- **Backend relaunch.** When a case exhausts its per-case retries still in a transport failure and
  the launcher owns a serving process, the backend is relaunched through the existing
  `BackendLauncher.stop()/start()` seam a bounded number of times and the case gets another round.

A case that reaches a terminal state -- including a terminal transport failure after exhausting
retries and relaunches -- is journaled (done-as-is); only a hard kill mid-case leaves a case
un-journaled, so resume re-runs exactly that one. The atomic staged-rename stays the transaction
boundary: the journal and its sidecar are dropped from the staging dir just before finalize, so the
published bundle never carries them. On a graceful interrupt (`KeyboardInterrupt`) or an abrupt
kill the staging dir is preserved for `--resume`; on a genuine error a fresh run's staging is
cleaned up (a resume attempt keeps its staging for another try). Retry, relaunch, and resumed-case
counters are recorded in `manifest.durability`. Sweep cells inherit all of this unchanged because
each cell shells out to `run-eval` (the hidden `.`-prefixed staging dir does not collide with the
sweep's cell-directory diff).

## Sweep RAG-config grid

`llb sweep` runs one isolated cell per runnable model. The `--rag-grid top_k=3,5,8` flag (Make:
`SWEEP_RAG_GRID`, **defaulting to `top_k=3,5,8`**) expands each model into one cell per `top_k`, so
the sweep answers "which `(model, top_k)`" for THIS host, not just "which model". This is the
default because the best depth VARIES by model -- on the 16 GiB committed goldset MamayLM-12B peaks
at `top_k=3` (0.541, well above its 0.501 at `top_k=5`) while Mistral peaks at `top_k=8`, and
gridding flipped the host recommendation from Lapa to MamayLM-12B@top_k=3. Only QUERY-TIME knobs
are gridded -- they change retrieval against the SAME index, so no re-index is needed: `top_k`
(depth) and `fusion_weight` (the hybrid dense/lexical RRF share; a `fusion_weight` axis implies
`retrieval_mode=hybrid`, so build the index with `RETRIEVAL_MODE=hybrid` first). Axes are
`;`-separated and cross-multiplied (`top_k=3,5;fusion_weight=0.4,0.6` -> 4 cells per model).
Every grid knob is a `RunConfig` field and therefore part of the cell fingerprint, so each grid
point gets its own resume key (existing cells resume, not re-run), and `recommend`'s
best-per-model dedup then represents each model by its highest-scoring grid point. Index-time
knobs (`chunk_size`/`chunk_overlap`) are out of scope because they need rebuilt indexes. Set
`SWEEP_RAG_GRID=` (empty) to disable the grid and run one cell per model at the manifest's
single config.

```bash
make sweep SWEEP_ID=grid                              # default grid: 5 models x 3 top_k -> 15 cells
make sweep SWEEP_ID=one SWEEP_RAG_GRID=               # disable: one cell per model
make sweep SWEEP_RAG_GRID="top_k=3,5;fusion_weight=0.4,0.6"   # hybrid fusion grid (hybrid index)
make recommend                                        # ranks each model at its best grid point
```
