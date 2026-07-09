# RAG Core

The RAG core evaluates one model over a verified gold split:

```text
retrieve -> generate -> classify -> score -> aggregate -> persist
```

It is intentionally backend-neutral. Backends launch differently, but the evaluator talks to an
OpenAI-compatible chat endpoint and receives normalized response classes.

## Configuration

`src/llb/config.py` defines `RunConfig`, the typed object that flows through retrieval,
generation, scoring, telemetry, and the manifest. YAML configs and CLI overrides share the same
validation path. Unknown keys and invalid ranges fail before work starts.

`src/llb/paths.py` loads `.env`, honors `DATA_DIR`, and resolves relative paths from the project
root instead of the caller's current directory.

## Command Path

```bash
llb prep-models
llb list-models
llb build-index --vector-store faiss
llb validate-retrieval --k 10
llb run-eval --model llama3.2:3b --backend ollama
llb run-eval --config samples/run_config_uk.yaml
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
interactive human scoring loop. The scoring/report core is in `src/llb/scoring/external_rag.py`;
the terminal session loop is in `src/llb/scoring/external_rag_session.py`; coverage lives in
`tests/test_external_rag_score.py`.

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

This is an external-system diagnostic, not a certified local leaderboard. If the answer log contains
only source article ids, titles, or URLs, the command cannot compute source-span recall; external
retrieval recall needs source records with corpus `doc_id`, `char_start`, and `char_end`.

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
"file X, page N, section Y" without re-deriving the join. The metadata *filter* seam over these
fields is shipped (`src/llb/rag/filters.py`; see Hybrid Retrieval below); governance fields
(`language`, `date`/`version`, ACL) are forward task 17 in [`plan.md`](../plan.md).

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

## Hybrid Retrieval (Dense + BM25 + RRF)

Shipped (hybrid-retrieval-uk): retrieval has the full hybrid shape Ukrainian enterprise corpora
need -- dense E5 plus lexical BM25 fused with weighted reciprocal-rank fusion, plus a
chunk-metadata filter seam -- so exact surnames, article/law numbers, codes, and abbreviations
stop losing to semantic-only search.

Modules:

- `src/llb/rag/lexical.py` -- pure-Python BM25 (`LexicalIndex`, in-repo, no new required dep)
  over the SAME offset-exact chunks the vector index holds; Ukrainian-aware token normalization
  on the LEXICAL side only (casefold, apostrophe-variant unification U+2019/U+02BC/`'`,
  punctuation strip); opt-in lemmatization via `pymorphy3` + `pymorphy3-dicts-uk` (the new
  `[lex]` optional extra) collapsing cases/inflection to lemmas at index AND query time -- the
  stored chunk text stays byte-identical (unit-tested); `rrf_fuse` implements the weighted RRF
  (`score = w/(60+dense_rank) + (1-w)/(60+lexical_rank)`) with deterministic tie-breaks.
- `src/llb/rag/filters.py` -- the chunk-metadata filter seam: `metadata_filter(doc_ids,
  heading_contains, page_range)` builds a predicate over `doc_id` plus the page-metadata join's
  `metadata.headers` breadcrumb and `metadata.pages` range; `RagStore.retrieve(question, k,
  chunk_filter=...)` applies it BEFORE fusion/ranking (with a filter the whole index is scanned,
  so the cut is exact). Task 17's ACL label will apply through this same seam.
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
index; skipped with a log line when `[lex]` is absent), and `dense+oracle-doc` -- a diagnostic
row restricting candidates to each gold item's `source_doc_id` through the filter seam,
quantifying the recall headroom a PERFECT document router would buy (never a scoring config).

The lemma normalizer is reused by the miss analysis: `topic_of` in
`src/llb/board/miss_analysis.py` lemmatizes its heuristic topic key best-effort, so Ukrainian
case forms of one topic collapse into a single cluster instead of splitting across inflections
(identity fallback when `[lex]` is absent).

Fixture: `samples/goldsets/exact_terms_uk/` -- a 40-entry near-identical Ukrainian orders
registry (order numbers, DSTU codes, surnames, amounts; ~41 recursive chunks) whose 8 items ask
for exact terms; the CI regression (`tests/test_hybrid_store.py`) proves hybrid strictly beats a
signal-free dense ranking there. Tests: `tests/test_lexical.py` (normalization, BM25 determinism
and tie-breaks, lemma matching, save/load), `tests/test_filters.py` (doc/heading/page
predicates), `tests/test_hybrid_store.py` (fusion order, weight extremes, filter-before-fusion,
refusal paths, config-knob application, byte-identical text), plus grid/tuner coverage in
`tests/test_cli_models.py` / `tests/test_tuner.py`.

Durable evidence (2026-07-08, real e5-base stores on the dev host, outside quick CI), via
`compare-retrieval --hybrid`:

- `samples/goldsets/ip_regulation_uk` (8 items, saturated fixture), k=10: all four rows hold
  recall 1.000 / MRR 1.000 -- hybrid is equal-or-better than dense on the committed goldset (the
  gate), and the fixture is too small to discriminate further.
- `samples/goldsets/exact_terms_uk` (8 exact-term items), k=10: recall ties at 1.000 but hybrid
  MRR 0.938 vs dense 0.713; at k=3 hybrid holds recall 1.000 / MRR 0.938 vs dense 0.875 / 0.688
  -- the strict exact-term win the lexical side exists for. `hybrid+lemmas` matched plain
  `hybrid` on both fixtures (exact numbers do not inflect; the lemma delta needs an
  inflection-rich full corpus -- forward task `hybrid-comparison-full-corpus` in
  [`plan.md`](../plan.md)). The oracle-doc row equals dense on these single-document corpora by
  construction (a doc filter is a no-op with one doc); it becomes informative only on a
  multi-document corpus.

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

Tests: `tests/test_rerank.py` (fake cross-encoder: candidate flow, kept set, rank bookkeeping,
stable ties, wrapper delegation, exact context ordering per policy, stage-latency capture and
manifest aggregation, config knob validation), `tests/test_compare_retrieval.py` (rerank twin
rows lift MRR through the shared metric; oracle row excluded), plus grid/tuner coverage in
`tests/test_cli_models.py` / `tests/test_tuner.py`.

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

## Query-Side Processing (uk-query-processing)

Shipped: an opt-in query lane between the user question and retrieval that measurably helps
Ukrainian queries while NEVER touching the stored corpus text (the query-side twin of the
index-side lexical normalization above). The raw question is always preserved -- only the
retrieval query is transformed -- and every step is honest: an A/B report attributes each step's
recall@k / MRR delta before anyone turns the lane on by default. Off by default (`query_prep`
empty is an exact no-op).

`src/llb/rag/query_prep.py` is a pure, unit-testable pipeline of NAMED steps (no store, model,
or `[rag]` extra needed -- it reuses the pure tokenizer in `llb.rag.lexical`):

- `normalize` -- matching-side casefold, apostrophe-variant unification (U+2019 / U+02BC / `'`),
  and a small transliteration table that maps Latin-typed Ukrainian tokens back to Cyrillic
  (`zakon` -> `закон`). The romanization map is injective, so the Latin->Cyrillic inverse is
  longest-match deterministic.
- `typos` -- deterministic corpus-vocabulary typo tolerance. The token vocabulary is built from
  the indexed corpus (`build_vocabulary` over `store.chunks`); a query token ABSENT from it is
  corrected to its nearest in-vocabulary token within Damerau-Levenshtein (OSA) distance 1 (2 for
  tokens over 8 chars). A token the corpus already contains is NEVER altered, and a purely numeric
  token (article/law number, code) is never "corrected" into a different one. Every correction is
  logged.
- `glossary` -- alias/glossary expansion. When the query mentions a known term (or a surzhyk /
  transliterated alias) the entry's other surface forms are APPENDED (the raw query is preserved),
  so retrieval catches the spelling the corpus actually uses. Sourced from a `query_glossary.json`
  built from a draft bundle's `prompt_dictionary_candidates.jsonl` (see
  [data prep](data-prep.md) query glossary).
- `rewrite` -- an optional local-LLM query rewrite through the run's backend endpoint seam
  (`eval.rag.query_rewrite` prompt). OFF by default and NEVER present unless explicitly requested;
  records both the original and rewritten query per case.

Wiring: `src/llb/eval/graph.py`'s retrieve node processes the question BEFORE `store.retrieve`
(the raw question stays in state for generation) and records `query_processed` /
`query_corrections` into the case state, carried into `scores.jsonl` rows so both query forms are
recoverable per case. `src/llb/executor/runner.py` `build_query_prep` resolves each step's
dependency (vocabulary from the loaded store, glossary from `query_glossary_path`, rewriter from
the launcher) and raises a clear message on a missing one.

Knobs (both `RunConfig` fields, hence in the manifest fingerprint): `query_prep` (ordered list of
`normalize` | `typos` | `glossary` | `rewrite`; unknown/duplicated steps rejected at config
validation) and `query_glossary_path`.

Commands:

```bash
make build-query-glossary BUNDLE=<draft dir>            # -> <bundle>/query_glossary.json
make run-eval MODEL=<m> QUERY_PREP=normalize,typos,glossary QUERY_GLOSSARY=<json>
make validate-retrieval GOLDSET=<gs> QUERY_PREP=normalize,typos,glossary QUERY_GLOSSARY=<json> QUERY_PREP_AB=1
```

The `validate-retrieval --query-prep-ab` A/B report scores `baseline` then each cumulative step
(`+normalize`, `+typos`, `+glossary`) with per-step recall@k / MRR deltas, so each step's marginal
retrieval effect is attributable (the `rewrite` step needs a model, so it runs only in `run-eval`,
not the A/B). `query_prep_ab_report` is pure over the `.retrieve` seam.

Tests: `tests/test_query_prep.py` (apostrophe unification, transliteration-table round-trips,
Damerau-Levenshtein transposition, typo correction that never touches in-vocabulary or numeric
tokens + long-token distance 2 + deterministic tie-break, deterministic alias expansion + glossary
build/round-trip, rewrite off-by-default, exact no-op when the lane is off, pipeline ordering +
dependency validation, A/B per-step delta over a fake store, retrieve-node raw-preservation and
processed-query wiring, runner resolver dependency wiring), plus config validation in
`tests/test_config.py`.

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

## Chunking Strategies

`src/llb/rag/chunking.py` implements every strategy behind one seam
(`chunk_spans -> (start, end, metadata)`), each anchored to `doc_id` + exact character offsets so
`validate-goldset` and source-span scoring work identically across strategies:

- `fixed`: character window with overlap (pure Python, zero deps);
- `sentence`: pack whole sentences up to `size` (never cuts mid-sentence);
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

Selection: `make build-index CHUNK_STRATEGY=<name>` / `build-index --strategy <name>` /
`RunConfig.strategy`; chunk-only via `python -m llb.rag.chunking --strategy <name>`. The Optuna
tuner searches the original five by default; `llb tune --extended-chunkers` adds
`page`/`heading`/`late` (`EXTENDED_STRATEGIES` in `src/llb/optimize/tuner.py`) -- opt-in because
`late` re-embeds whole documents per trial and `page` only differs from `recursive` on
sidecar-bearing PDF corpora.

Chunker comparison: `make compare-retrieval CHUNK_STRATEGIES=page,heading,late,markdown,semantic`
(`compare-retrieval --strategies ...`) builds one flat FAISS store per strategy over the SAME
corpus + pinned embedder (persisted under `$DATA_DIR/llb/rag/<strategy>/`) and ranks them by
recall@k / MRR on the gold set, so the best chunker is demonstrated per corpus, never assumed.
Tests: `tests/test_chunking_strategies.py` (offset round-trips, page-boundary alignment on the
committed `samples/pdf_pages` sidecar fixture, heading packing/breadcrumbs, late pooling math and
fallbacks) plus the pre-existing `test_chunking.py`/`test_page_metadata.py` suites.

Durable evidence (2026-07-08, CUDA host, outside quick CI): on the committed
`samples/goldsets/ip_regulation_uk` fixture (8 items, single `.md` corpus, no PDF sidecars --
`page` therefore equals `recursive` there), recall@10 SATURATES at 1.000 for all seven compared
strategies; at k=3 `heading`/`markdown`/`page`/`recursive`/`semantic`/`sentence` all hold
1.000 recall / 1.000 MRR while `late` drops to 0.875 / 0.750 -- on this tiny corpus late pooling
blurs, not sharpens, retrieval; it must prove itself per corpus before adoption. Like the
embedder bake-off, the fixture is too small to discriminate the winners -- the discriminating
run is forward task `chunking-comparison-full-corpus` in [`plan.md`](../plan.md).

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
are fake-store unit-tested (`tests/test_embedding_bakeoff.py`) with no GPU/FAISS/network.

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

Durable evidence (2026-07-04, heavy build on the CUDA host, outside quick CI): the four local
candidates over the committed `samples/goldsets/ip_regulation_uk` fixture (8 items, 10 chunks,
`k=10`):

| model | recall@10 | MRR | dim |
| --- | ---: | ---: | ---: |
| `intfloat/multilingual-e5-base` | 1.000 | 1.000 | 768 |
| `intfloat/multilingual-e5-large` | 1.000 | 1.000 | 1024 |
| `BAAI/bge-m3` | 1.000 | 1.000 | 1024 |
| `lang-uk/ukr-paraphrase-multilingual-mpnet-base` | 1.000 | 0.917 | 768 |

Winner for the 16 GB host: `intfloat/multilingual-e5-base` (the current default) -- the three
retrieval-tuned encoders all saturate recall@10 and MRR on this tiny fixture, so the paraphrase/STS
`lang-uk` model is the only one that drops MRR (0.917), confirming the hypothesis that a paraphrase
objective can lose to retrieval-tuned encoders; among the tied three, e5-base wins the throughput
tie-break at the smallest index. Caveat: recall saturates on a 10-chunk fixture and the reported
`chunks/s` is load-dominated (cold SentenceTransformer load over 10 chunks), so re-run the bake-off
on a real full corpus to separate steady-state throughput and to let recall@k discriminate.

## Retrieval Metrics

`src/llb/rag/retrieval.py` computes recall@k and MRR by source-span overlap. The common gate is
`recall@10 >= 0.8`.

This metric is not a model-ranking axis. It answers whether the retrieval layer is able to surface
the evidence the model needs. If retrieval is poor, answer quality is capped by context quality.

The default store retrieves dense-only (cosine over the pinned E5 embedding). Measured against
the gate, dense-only passes on the committed fixture (`recall@10=0.980`) but falls short on the
real full-corpus PDF index (`recall@10=0.729`, see the quickstart note above), so dense-only has
NOT been proven sufficient for a real Ukrainian corpus. Hybrid retrieval (see Hybrid Retrieval
above) and cross-encoder reranking (see Reranking And Context Order above) are the shipped
levers; query processing is forward task 15 in [`plan.md`](../plan.md).

## Generation Graph

`src/llb/eval/graph.py` builds the retrieve-generate flow. LangGraph is imported only when the
graph is built. The graph records one status per case: `ok`, `empty`, `malformed`, `refusal`,
`timeout`, `backend_error`, `retrieval_miss`, or another typed failure from the shared taxonomy.

`src/llb/backends/openai_client.py` normalizes endpoint failures. Backend launchers own process
lifecycle and readiness checks.

## Scoring

`src/llb/scoring/correctness.py` computes objective correctness using normalized token-F1 with
exact and contains helpers. `--score-semantic` records a pinned-embedder cosine signal for
paraphrases and morphology; it is kept separate from the objective unless a ranking policy
explicitly uses it.

`src/llb/scoring/judge.py` runs the local judge only when configured. The judge enters ranking only
when the caller supplies a calibration rho that clears the trust threshold. Otherwise it is
diagnostic and objective correctness ranks alone.

`src/llb/scoring/aggregate.py` produces leaderboard rows. The policy favors quality first, then
throughput, then lower VRAM when telemetry is available.

### Groundedness and citation metrics (groundedness-citation-metrics)

Shipped: three answer-side signals that go beyond reference-answer overlap, all deterministic and
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
- Insufficient-context abstention probe (`--insufficient-context-probes <n>`,
  `src/llb/eval/insufficient_context.py`): re-runs a seeded sample of gold items with every chunk
  overlapping their gold spans EXCLUDED from retrieval (through the shipped chunk-metadata filter
  seam). Correct behavior is an explicit abstention (`llb.eval.common.is_abstention` = refusal OR an
  insufficient-context marker), scored as abstention accuracy. Probe rows live in `probes.jsonl` (+
  `insufficient_context_report.md`), NEVER in `scores.jsonl`, so they cannot enter the plain
  correctness aggregates.

Per-case fields land in `scores.jsonl` (`groundedness`, `citation_validity`,
`hallucinated_citation_rate`, `n_citations`); their means plus `abstention_accuracy` / `n_probes`
land in the manifest `metrics`, echoed as the run's `answer-side:` summary line. Config knobs
(`cited_answers`, `score_groundedness`, `insufficient_context_probes`) are recorded in the manifest
fingerprint. `RunConfig` toggles are off by default, so pre-existing bundles keep their shape.

Modules/tests: `src/llb/scoring/groundedness.py`, `src/llb/eval/insufficient_context.py`, the
`eval.rag.cited_answer` template, `ScoreOptions` in `src/llb/executor/cases.py`;
`tests/test_groundedness.py` (fully/partially/unsupported groundedness with zero cross-class leakage,
valid/flagged-invalid/hallucinated citations, abstention markers, cited-answer prompt wiring, per-case
scoring + context-order-aware citation numbering) and `tests/test_insufficient_context.py` (gold
exclusion, seeded sampling, abstention accuracy, transport-error exclusion).

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
```

Parquet is used when `pyarrow` is available; JSONL is the portable fallback. The bundle is staged
in a hidden sibling directory and atomically renamed when canonical files are complete. MLflow
mirroring runs after canonical persistence and is best-effort.

Per-case rows record `retrieval_hit` and `first_hit_rank`, but the retrieved chunk records
themselves are not persisted in the bundle -- `retrieval_pairs` stay in-process
(`src/llb/executor/cases.py`) for aggregate retrieval metrics and judge records. Adding an
additive per-case retrieved-spans record is part of forward task 6 (`miss-analysis-recommendations`
in [`plan.md`](../plan.md)), which needs it for span-overlap miss classification.

## Executor

`src/llb/executor/runner.py` orchestrates one run. It filters unverified items, loads the selected
retrieval backend, executes cases, collects optional telemetry, writes artifacts, mirrors to MLflow,
and prints the row.

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
