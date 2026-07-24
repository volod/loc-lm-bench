# Data Prep

Data prep turns source documents into benchmarkable, verified records. Keep the distinction clear:
automated drafting and cross-checking can prepare evidence, but only reviewed `verified: true`
items score models.

## Gold Item Contract

`src/llb/goldset/schema.py` defines `GoldItem` and `SourceSpan`.

```text
id
lang
question
reference_answer
source_doc_id
source_spans[{doc_id, char_start, char_end, text}]
provenance
verified
split
```

The important design choice is span identity. Labels use document-relative character offsets rather
than chunk ids, because chunking strategy is an experiment variable. `load_goldset` and
`dump_goldset` read and write UTF-8 JSONL.

## Splits And Validation

`src/llb/goldset/splits.py` assigns deterministic disjoint
`calibration`, `tuning`, and `final` splits. `src/llb/goldset/validate.py` checks ids, split
labels, corpus references, and exact span text.

```bash
make validate-goldset
```

The default target validates `samples/goldsets/ua_squad_postedited_v1/goldset.jsonl` against its
sibling `corpus/`.

## Committed Fixture

`samples/goldsets/ua_squad_postedited_v1/` is the default development fixture:

- `goldset.jsonl`: 250 verified Ukrainian QA items;
- `corpus/`: one exact source document per item;
- `source.json`: upstream identity, revision, source digest, selection rule, and license context.

The fixture is intentionally small and stable. It is suitable for smoke checks, retrieval
comparisons, and development regressions. It is not a substitute for evaluating a private target
corpus.

## Ingestion

`src/llb/prep/ingest_squad.py` maps SQuAD-like rows to `GoldItem` records. It accepts local JSON,
Hugging Face rows, flattened rows, nested article rows, and rows whose `answers` value is encoded
as a dict string.

```bash
make ingest-uk-squad GOLDSET_MODE=development
make ingest-uk-squad GOLDSET_MODE=skeleton
make ingest-uk-squad GOLDSET_MODE=draft CORPUS=<dir>
make ingest-squad SQUAD_JSON=path.json
python -m llb.prep.ingest_squad --hf-dataset <id> --hf-split train
```

Draft imports start with `verified=false`. A verification ledger can adopt matching canonical rows
by id. Adoption replaces the whole canonical item and corpus file, which prevents a reused id from
certifying changed content.

`src/llb/prep/goldset_skeleton.py` writes an editable from-scratch SQuAD template under
`$DATA_DIR/goldset-skeleton/<timestamp>/`.

For **open** corpora, drafts can also be authored with an external AI provider service (Claude
Projects, Gemini/NotebookLM, ChatGPT Projects) and imported either as SQuAD-shaped context docs
(`make ingest-squad`, Artifact A) or, for full-document needle realism, as corpus-grounded JSONL
(`make import-external-draft`, Artifact B). Restricted or private corpora stay on the local ontology
pipeline -- egress is never the default. The workflow, per-service setup, copy-paste prompts, and the
exact artifact shapes are in
[`docs/guides/data-prep/external-ai-service-artifacts.md`](../../guides/data-prep/external-ai-service-artifacts.md),
[`docs/guides/data-prep/external-service-prompts/`](../../guides/data-prep/external-service-prompts/README.md),
and the [external-service draft contract](../../design/external-draft-contract.md).

### Grounded-JSONL import (Artifact B -> draft bundle)

`make import-external-draft` / `llb import-external-draft` (`src/llb/prep/external_draft.py`) turns a
grounded-JSONL export (contract Artifact B: `quote` + `source_doc_id` rows) into a canonical draft
bundle for the usual `validate-goldset` -> `cross-check-goldset` -> `verify-*` chain. Unlike
`ingest-squad` -- which stamps `provenance: public-reused`, hashes each context into its own doc
(losing full-document needle realism), and cannot read grounded JSONL -- the import re-grounds
against the FULL original corpus doc:

- egress gate FIRST: the required `external_provenance.json` sidecar must be present and declare
  `data_classification: "open"`; a missing or non-open sidecar aborts before any bundle is written
  (uploading a corpus to a provider publishes it -- restricted data never leaves the box);
- re-grounding: each `quote` is located in `<corpus-root>/<source_doc_id>` via
  `frontier.ground_span` (exact, then casefold/whitespace-normalized-but-exact); a non-verbatim row
  is dropped and counted, never mis-grounded, and a near-verbatim quote is re-snapped to the exact
  corpus text with exact `source_spans` computed from the match;
- canonical bundle: `goldset.jsonl` (`provenance: frontier-drafted`, `verified: false`), a
  byte-identical verbatim `corpus/` copy of the referenced docs, `provenance.json` recording the
  external service / model / export date / `data_classification`, and `item_provenance.jsonl`
  carrying each item's `question_type`/`difficulty` (honored from the row when valid, else
  classified via `ontology.question_types`) WITHOUT changing the `GoldItem` schema;
- multi-service merge: `llb curate-drafts --kind grounded` merges/dedups/filters many Artifact B
  exports (re-grounding quotes, dropping non-verbatim/flabby rows, unique-id rewrite) into ONE JSONL
  before import, exactly like the other curation kinds;
- needle parity (external-import-needle-parity): an optional
  `--retrieval-index-dir <index>` / `--retrieval-k <k>` annotates each imported item with its
  gold-span retrieval rank against the full-corpus index (the shipped
  `ontology.needles.annotate_needle_retrieval`), recorded as additive `retrieval_rank` /
  `retrieval_k` fields in `item_provenance.jsonl` -- the same per-item confidence-ordering +
  retrieval-uniqueness signal local drafts carry into the verify worksheet (which already reads
  that file). `--drop-nonretrievable-needles` (explicit opt-in, requires the index) drops
  rank-less items with the reason counted in the import report; `provenance.json` gains the
  `needle_retrieval` summary. Without an index the lane is an exact no-op.

Committed fixture + unit coverage (no network): `samples/external-drafts/claude-projects-open/`
(one open-data artifact + sidecar), `tests/llb/prep/test_external_draft.py` (including the needle-rank
annotation over an injected fake retriever), and the grounded cases in
`tests/llb/prep/test_curate_drafts.py`.

```bash
llb curate-drafts <svc-a>.jsonl <svc-b>.jsonl --kind grounded \
  --corpus-root <corpus> --out grounded.jsonl
make import-external-draft ARTIFACT=grounded.jsonl CORPUS=<corpus> \
  SIDECAR=<external_provenance.json> RETRIEVAL_INDEX_DIR=<rag-index> RETRIEVAL_K=10
```

### External-draft curation (merge / dedup / filter)

`make curate-drafts` / `llb curate-drafts` (`src/llb/prep/curation/`) turns the pile of
per-service, per-batch external exports into ONE importable artifact per kind -- the mechanism
behind multi-service best-of-N drafting (run the same prompts in Claude and Gemini, merge the
union). Kinds: `squad` (Artifact A -> `make ingest-squad`), `grounded` (Artifact B ->
`make import-external-draft`), `security` (Artifact C -> `make bench-security`), `chains`
(Artifact D, review-only), `inventory` (merged coverage plan for the drafting prompts). Behavior:

- lenient loading: whole JSON files, raw replies with fenced code blocks, or JSONL;
- inventory batch arrays: `CURATE_KIND=inventory` also accepts one top-level JSON array of complete
  prompt-01 response objects, so NotebookLM "continue" sessions can be saved as
  `[{response 1}, {response 2}, ...]` in a single file;
- coverage source rendering: `make coverage-plan-text` / `llb coverage-plan-text`
  (`src/llb/prep/curation/coverage_text.py`) converts a per-document prompt-01 inventory slice
  into a NotebookLM-friendly `.txt` source using the shared curation JSON loader and atomic writer;
- verbatim repair via `frontier.ground_span`: near-verbatim answers/contexts/grounding quotes are
  re-snapped to exact corpus text when `CURATE_CORPUS=<staged-dir>` is set, and a wrong SQuAD
  `title` is corrected to the document where the context was found;
- invalid filters: answers not in context, contexts not in corpus, schema-invalid security cases
  (closed families via `SecurityCase.from_record`, benign-vs-expect_refusal conflicts, leak
  probes without markers), structurally broken chains (step counts, missing quotes, reused
  passages);
- flabby filters: circular questions (reuses `ontology.refine.is_circular`), vague stubs,
  document-structure references ("у цьому документі"), whole-paragraph answer spans, chains whose
  final answer is findable from the step-1 passage;
- dedup: exact normalized-question dedup, then greedy pinned-E5 near-dup (threshold 0.9, same
  meaning as ontology drafting dedup) with bias pairs / cross-language groups protected as
  intentional twins and orphaned bias-pair variants dropped whole;
  `CURATE_DEDUP_AGAINST=<bundle>` suppresses re-drafts of prior accepted bundles' questions;
- id collision rewrite across services and a `*.curation_report.json` sidecar with per-source,
  per-reason counts.

Unit coverage: `tests/llb/prep/test_curate_drafts.py` (fake hashed-BoW embedder; no model downloads).

`make external-squad-rag` is the single-command prompt-02 SQuAD path for a directory or explicit
list of external exports. It accepts `SQUAD_DRAFT_INPUT_DIR=<exports-dir>` or
`SQUAD_DRAFT_INPUTS="<file> [<file> ...]"`, requires `SQUAD_DRAFT_CORPUS=<staged-corpus-dir>`, and
writes the curated export, canonical `llb/goldset/<name>`, imported `llb/corpus`, and `llb/rag`
index under `SQUAD_DRAFT_OUT_DIR` (default `$DATA_DIR/external-squad-rag`). The target runs
curation, SQuAD ingest, structural validation, and `build-index` in order. It sources the project
`.env` before curation so `HF_TOKEN` is exported for semantic deduplication and embedding
downloads.

Already-answered external RAG logs use the RAG-core diagnostic command rather than `run-eval`:
`make score-external-rag EXTERNAL_RAG_ANSWERS=<answered-jsonl>` opens an interactive human scoring
session over rows carrying gold fields plus `llm_answer` or `predicted_answer`. Human scores,
decisions, notes, and corrected answers are saved back into the JSONL after each edit; final CSV and
Markdown report artifacts are written only after all rows are scored. The CSV keeps raw answers and
first-source columns, while objective scoring uses the same reference-correctness functions as
local RAG runs. See [RAG core](rag-core.md) external answer log scoring and
[`docs/guides/data-prep/goldset-from-scratch.md`](../../guides/data-prep/goldset-from-scratch.md).

NotebookLM inventory-array coverage is implemented in `src/llb/prep/curation/inventory.py` and
covered by `test_inventory_accepts_array_of_response_objects`. The goods quickstart NotebookLM
inventory export was curated with:

```bash
make curate-drafts CURATE_KIND=inventory \
  CURATE_INPUTS="$DATA_DIR/quickstart-pdf-corpus-md/nlm-inventory.json" \
  CURATE_OUT="$DATA_DIR/quickstart-pdf-corpus-md/nlm-inventory.curated.json" \
  CURATE_CORPUS="$DATA_DIR/quickstart-pdf-corpus-md"
```

Output:
`$DATA_DIR/quickstart-pdf-corpus-md/nlm-inventory.curated.json` and
`$DATA_DIR/quickstart-pdf-corpus-md/nlm-inventory.curated.curation_report.json`.
The run loaded inventory document entries, kept staged documents, and retained topics,
entities, relations, numeric facts, sensitive-topic labels, and cross-document
links. The report recorded repairs and invalid quote-grounding failures, all from quotes
that were not exact substrings of the staged markdown corpus.

Prompt 02 (`docs/guides/data-prep/external-service-prompts/02-goldset-draft.md`) documents how to
map a large curated inventory into a drafting prompt: extract a per-document JSON slice with `jq`,
convert that slice to text for NotebookLM with `make coverage-plan-text`, upload the text as a
NotebookLM source, and reference the source file name in `COVERAGE PLAN`. Non-NotebookLM services
can still receive a compact pasted JSON slice, and bounded array windows remain useful for
section-like batches when a single document's inventory is too large for one chat turn. NotebookLM
draft replies are capped at 15 requested items.

`make pdf-to-markdown`, `llb pdf-to-markdown`, and `llb ingest-pdf-corpus` extract local PDF
directories into the canonical `.md` corpus shape used by RAG, ontology drafting, prompt-system
packages, and GraphRAG. The default `PDF_PARSER=auto` path uses PyMuPDF4LLM with OCR disabled for
born-digital PDFs, and Docling with Tesseract CLI OCR (`ukr+eng`) for image-only PDFs when the
`pdf-quality` extra and OCR apt packages are installed. Marker, Unstructured, and MarkItDown remain
available as explicit `PDF_PARSER=<tool>` probes, but they are not default full-corpus candidates.
The converter writes stable ASCII `pdf-<digest>.md` ids, preserves the source PDF path in a manifest,
and skips PDFs only when the selected parser output stays below `--min-chars`.

Conversion is incremental: each manifest item records `source_sha256`, and a rerun reuses the
existing `.md` plus citation sidecar when the source fingerprint, requested parser, and min-chars
still match and the outputs exist (`reused: true` in the manifest; `[pdf-corpus] reuse ...` in the
log). `--refresh` (make: `PDF_REFRESH=1`) forces a full reconversion. This makes quickstart reruns
skip the docling/OCR pass entirely for an unchanged corpus.

```bash
make pdf-to-markdown
make pdf-to-markdown PDF_DIR=<pdf-dir> PDF_OUT_DIR=<out-dir> PDF_MIN_CHARS=500 PDF_PARSER=auto
make pdf-to-markdown PDF_REFRESH=1
llb ingest-pdf-corpus --pdf-root <pdf-dir> --out-dir <out-dir> --min-chars 500 --parser auto
```

The converter strips PDF page furniture line-by-line while it renders (`strip_page_furniture` in
`src/llb/prep/pdf/furniture.py`): a short line that recurs on many pages -- a running header or
footer, a bare page number -- is dropped so a passage crossing a page break grounds contiguously.
That pass is per-line and cross-page; the block-level intra-document handling below is a separate,
opt-in step for whole blocks a single document repeats.

#### Intra-document repeated-block handling (`--repeat-blocks`)

A converted manual also repeats whole BLOCKS inside the one document -- a boilerplate procedure
step restated in section after section, a note repeated under every table -- which the per-line
furniture pass cannot see and which index-time [duplicate chunk
collapse](rag-core.md#duplicate-chunk-collapse) can only hide, not fix at the source: collapse
indexes the block once but still returns that one copy for a question about any section that
carries it, and the document's own chunk ordinals stop tracking its reading order. Measured on the
goods corpus every one of the 494 exact chunk-collapse groups is intra-document (0 cross-document),
the largest block repeating 335 times in the single 637 KB manual -- so on this corpus the
repetition is entirely a conversion-side property of one document, not shared page furniture.

`llb.prep.pdf.repeats` measures and, optionally, rewrites it. `ingest-pdf-corpus` /
`pdf-to-markdown` / `ingest-corpus` take `--repeat-blocks keep|drop|anchor` (the mode is recorded
per manifest item and is part of the reuse key, so switching it reconverts):

- `keep` (default) -- unchanged; the rendered document is byte-identical to before.
- `drop` -- index the FIRST occurrence of a repeated block and remove the rest. Loss-free (every
  removed copy is byte-identical to the survivor) and it shrinks the source, so the freed top-k
  slots carry other evidence.
- `anchor` -- keep every occurrence and prefix each with its enclosing-heading breadcrumb (glued
  with no blank line, so every blank-line splitter keeps anchor and block in one chunk), so copies
  under different sections stop being identical and each is retrievable in its own section.

A block counts as repeated at `--min-repeats` occurrences (default 3) INSIDE one document; repeated
markdown headings and table-header/`|`-rows are never rewritten, because they carry structure the
tables and sections under them depend on. Both rewriting modes are offset-exact: every edit is a
recorded `TextEdit` and `remap_span` moves a surviving offset (a dropped copy resolves onto the
survivor of its identical text), so page-citation sidecars and gold spans follow the rewrite; a
span that straddles a rewrite has no single image and is refused rather than moved.

`make strip-corpus-repeats` (`llb strip-corpus-repeats`) runs the same census or rewrite on an
ALREADY-converted `_md` corpus -- the common case, since the corpus outlives its source PDFs. It
never edits in place: `REPEAT_MODE=keep` (default) reports only, `drop`/`anchor` write a NEW root
under `REPEAT_OUT=` with the citation sidecars remapped, and `GOLDSET=` remaps a gold set's span
offsets onto the rewritten corpus (dropping and naming any item whose evidence straddles a rewrite)
so the same questions stay scoreable.

```bash
make strip-corpus-repeats CORPUS=<md-corpus>                       # census only, writes nothing
make strip-corpus-repeats CORPUS=<md-corpus> REPEAT_MODE=drop REPEAT_OUT=<new-root> GOLDSET=<gs>
llb ingest-pdf-corpus --pdf-root <pdf-dir> --repeat-blocks drop    # at conversion time
```

Retrieval verdict (CUDA host, pinned e5-base, `sentence`/`recursive` at `size=200`, k=10, seed 13,
exact collapse ON in every lane; the 89 goods items whose gold spans survive both rewrites, so the
three lanes score one item set; floor `+/-0.000` throughout; reports under
`$DATA_DIR/retrieval-noise-floor/20260723T-intra-repeats/`):

| lane | recursive recall@10 | sentence recall@10 | dup% (recursive) | corpus chars |
| --- | ---: | ---: | ---: | ---: |
| `keep` (baseline) | 0.708 | 0.640 | 37.7% | 681627 |
| `drop` | 0.730 | 0.674 | 24.8% | 531011 |
| `anchor` | 0.685 | 0.685 | 34.9% | 755943 |

Verdict: ADOPT `drop` as an available conversion-side option, KEEP `keep` as the default, REJECT
`anchor`. `drop` lifts recall@10 by +0.022 (`recursive`) / +0.034 (`sentence`) -- both clear of the
`+/-0.000` floor -- while cutting the intra-document duplicate share the index carries from 37.7% to
24.8% and shrinking the source 22%. The gain is not about ties (exact collapse already drove the
floor to zero): a top-10 that no longer must re-list one boilerplate block carries more distinct
evidence, and unlike collapse the survivor now sits in its first section only. `anchor` helps
`sentence` (+0.045) but regresses `recursive` (-0.023) and, by making copies textually distinct,
defeats the cheaper exact collapse and inflates the index -- so it is not a default, only a probe
for a corpus whose repeated blocks genuinely belong to several sections at once. `drop` stays
opt-in because it is not loss-free at the QUESTION level: on the goods corpus it removes 5 of 95
items from the scored set (their gold span straddled a removed block), 3 of which the baseline
could retrieve -- the per-question audit below quantifies exactly that cost, which is the operator's
call to make per corpus.

##### Per-question yield audit (`audit-repeat-yield`)

The pooled recall verdict above is measured on the items that SURVIVE the rewrite, so it cannot
show what `drop` cost the questions it moved. `make audit-repeat-yield` (`llb audit-repeat-yield`,
`src/llb/prep/pdf/repeat_yield.py`) measures that directly: it runs the `drop` strip, indexes the
keep and drop corpora identically, retrieves each item on its own corpus (baseline against the
original spans, drop against the remapped spans), and classifies every item -- `held` (hit both
sides), `lost` (hit -> miss), `recovered` (miss -> hit), `dropped_from_set` (evidence straddled a
rewrite, item removed). The goldset remap tags each item's change as `unmoved`, `rehomed` (its
evidence moved onto a survivor), or `dropped`, so the report separates a re-homing from a
corpus-wide ranking side-effect. It ends in an ADOPT/HOLD verdict naming any question the strip
cost that retrieval could previously answer.

```bash
make audit-repeat-yield CORPUS=<md-corpus> GOLDSET=<gs> CHUNK_STRATEGY=sentence CHUNK_SIZE=200
```

Measured on the goods corpus (CUDA host, pinned e5-base, `size=200`, k=10, all 95 items; reports
under `$DATA_DIR/retrieval-noise-floor/20260723T-repeat-yield-<strategy>/`):

| strategy | kept recall@10 keep -> drop | held | recovered | lost (re-home) | dropped-from-set | answerable lost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `sentence` | 0.633 -> 0.667 (+0.033) | 56 | 4 | 1 (unmoved flip) | 5 | 4 |
| `recursive` | 0.700 -> 0.722 (+0.022) | 63 | 2 | 0 | 5 | 3 |

Both strategies return HOLD, and the audit resolves exactly where the pooled gain and the hidden
cost each come from:

- Re-homing itself is harmless to retrieval. NO item whose evidence moved onto a survivor became a
  miss under either strategy -- the survivor is byte-identical text and retrieval still reaches it.
  The one `sentence` `lost` item was `unmoved`: a corpus-wide ranking side-effect of removing other
  blocks, not the re-homing.
- The pooled gain is real and comes from `recovered` items (4 `sentence` / 2 `recursive`): a top-10
  no longer padded with repeated boilerplate surfaces evidence the baseline missed.
- The genuine cost is the 5 items `drop` removes from the scored set entirely -- their gold span
  STRADDLED a removed block boundary, so `remap_span` cannot map it to one contiguous image. 3 of
  those 5 the baseline could retrieve. This is the concrete question-level cost the survivor-only
  pooled number hid, and `--recover-straddle` below removes it.

###### Straddle recovery (`--recover-straddle`)

A straddling gold span is `<tail of a removed copy> + <head of the block after it>`; the removed
copy's text still exists on the survivor and the following block stays in place, so the span is not
truly lost -- it just maps to two non-contiguous images. `remap_span_split` (`--recover-straddle`
on `strip-corpus-repeats` and `audit-repeat-yield`, `REPEAT_RECOVER=1`) splits the span at every
edit boundary it crosses, re-anchors each piece (the removed part onto the survivor, the kept part
by shift), and keeps the item with several spans instead of dropping it. Because `recall_at_k`
credits an item when ANY of its spans is covered, the split preserves the original retrieval
semantics, and each piece is verified against the stripped text so an off-by-one remap fails loudly.

Re-run with recovery on (same corpus, k, splits; reports under
`$DATA_DIR/retrieval-noise-floor/20260723T-straddle-recover-<strategy>/`):

| strategy | kept recall@10 keep -> drop | dropped-from-set | answerable lost | verdict |
| --- | --- | ---: | ---: | --- |
| `sentence` | 0.632 -> 0.663 (+0.032) | 0 (was 5) | 1 (was 4) | HOLD |
| `recursive` | 0.695 -> 0.716 (+0.021) | 0 (was 5) | 0 (was 3) | ADOPT |

Recovery does exactly what its design predicts: all 5 straddlers re-enter the scored set, every one
of the 3 previously answerable-lost items becomes `held` (retrieval reaches the recovered survivor
piece), and the pooled kept-recall is unchanged within the `+/-0.000` floor. `recursive` flips to
ADOPT -- the strip now costs zero answerable questions. `sentence` still returns HOLD, but for a
reason unrelated to the strip's rewrites: its one remaining `lost` item (`...-onto-81`) is `unmoved`
and was `lost` in the no-recovery audit too -- a corpus-wide ranking side-effect of removing
boilerplate distractors, which no straddle handling can address. So with `--recover-straddle` the
straddle cost of `drop` is fully recovered, and what remains is only the ordinary ranking noise any
index edit produces.

Each successful document gets a `pdf-<digest>.citations.json` sidecar with source PDF, parser, PDF
diagnostics, page numbers, generated-corpus character spans, and page-local block spans when the
parser exposes them.
The same directory also contains `pdf_corpus_manifest.json` and `pdf_corpus_quality.json`; the
quality report records parser attempts, diagnostics, page coverage, citation coverage, structure
markers, and the selection score.

At `build-index` time these sidecars are joined onto every chunk (`src/llb/rag/page_metadata.py`):
a chunk whose char span intersects a page span gains `metadata.pages`/`metadata.source_pdf`, and
`store_meta.json` records the resulting `page_annotation_coverage`. See the
[RAG core](rag-core.md) retrieval-store section for the join and its guarantees.

### Mixed txt/md/pdf ingestion

`make ingest-corpus` / `llb ingest-corpus` turns ONE mixed `txt`/`md`/`pdf` directory into the
canonical corpus in a single command (`src/llb/prep/corpus_ingest.py`). PDFs route through the
`ingest_pdf_corpus` converter above (same `pdf-<digest>.md` ids and citation sidecars); `.md`/`.txt`
files pass through verbatim under their relative path so offsets stay exact. Both lanes share the
PDF manifest contract: a per-source `source_sha256`, incremental reuse when the source is unchanged
(`reused: true`), and skip diagnostics for short/failed documents. A unified `corpus_manifest.json`
records every source with its `kind` (`pdf`|`text`), status, and reuse flag, so a rerun over an
unchanged mixed corpus reports `reused: true` for every document. The staged corpus walk excludes
the output subtree, so the default `<root>/_md` output is never re-ingested as new input.

Governance metadata is part of the same manifest contract. Every manifest item records
`language`, `ingestion_time`, `source_system`, optional `version`, optional `effective_date`, and
optional `acl_label`. Text sources can provide per-document values in `<source>.metadata.json` or
markdown front matter; otherwise `--default-language` is used, then a cheap deterministic detector.
`--source-system` and `--acl-label` set defaults for sources that do not provide their own values.
PDF rows inherit any conversion-manifest governance fields when present and otherwise use the same
operator defaults. Re-ingesting an unchanged source keeps the previous `ingestion_time` when its
non-time governance fields are unchanged.

Deletion propagation is explicit: a source removed from the input root is removed from the next
`corpus_manifest.json`, its staged output file is deleted from the canonical corpus, and the
manifest records `removed_sources` plus `n_removed_sources`. Changed PDF ids also clean up stale
old staged outputs. The rollback unit is the immutable store directory built from a manifest
fingerprint (`llb refresh-index` publishes each refresh as a new
`$DATA_DIR/llb/rag/generations/<utc-ts>/` generation; deleting the newest one rolls back).

Manifest-diff contract (dynamic-corpus-refresh): `corpus_doc_fingerprints` in
`src/llb/prep/corpus_governance.py` maps `doc_id -> fingerprint` from the same two sources as
`corpus_fingerprint` -- the canonical per-item row (content sha256 plus governance fields) when
`corpus_manifest.json` exists, else the sha256 of each committed `.md`/`.txt` file keyed by its
corpus-relative path. In both modes a document's PDF citation sidecar
(`pdf-<digest>.citations.json`) hash is folded into its fingerprint when one exists, so a
sidecar-only regeneration (page spans rebuilt, text unchanged) reads as a modified document;
sidecar-less docs keep the plain hash, so older stores stay refresh-compatible. `build-index`
records the map in `store_meta.json` as `doc_fingerprints`; `llb refresh-index` diffs it against
the current corpus to re-chunk/re-embed only added or modified documents and to drop deleted
ones (details in [RAG core](rag-core.md#store-lifecycle-dynamic-corpus-refresh)).

```bash
make ingest-corpus CORPUS_ROOT=<mixed-dir> CORPUS_OUT_DIR=<out-dir> CORPUS_MIN_CHARS=500
make ingest-corpus CORPUS_ROOT=<mixed-dir> CORPUS_DEFAULT_LANGUAGE=uk CORPUS_ACL_LABEL=<tag>
make ingest-corpus CORPUS_ROOT=<mixed-dir> CORPUS_REFRESH=1
llb ingest-corpus --root <mixed-dir> --out-dir <out-dir> --min-chars 500 --parser auto \
  --default-language uk --acl-label <tag>
```

`make quickstart-corpus CORPUS_SRC=<dir>` (script target `corpus`) generalizes the PDF quickstart
stages to a mixed corpus: `ingest-corpus` -> full-corpus index -> ontology draft -> graph ->
validate, logging each stage under `$DATA_DIR/llb/logs/quickstart/`. It reuses the PDF quickstart's
model selection, workload estimate, and confirmation gate, and drafts directly over the converted
corpus (passthrough text has no citation sidecar, so no per-doc staging step is needed). The mixed
fixture `samples/corpus/` (`.md` + `.txt`) backs the ingestion unit tests.

Ontology draft bundles preserve that PDF evidence. When a source document has a matching
`*.citations.json` sidecar, `prepare-goldset-draft` copies it into the bundle `corpus/` directory
and writes these review artifacts beside `goldset.jsonl`:

- `pdf_ontology_report.json`: parse rate, elapsed seconds, grounded entity/event/claim/fact counts,
  page-span citation coverage, citation-valid needle count, dictionary-term yield, needle-retrieval
  metrics when enabled, and quality gates with a `passed` roll-up (grounded extractions of any kind
  + a non-empty gold set, plus a citation-valid needle for PDF corpora).
- `prompt_dictionary_candidates.jsonl`: source-backed entity and relation terms with supporting
  spans and PDF page references when sidecars exist. This artifact also seeds the query-side
  glossary (see Query Glossary below).
- `needle_items.jsonl`: drafted gold items whose source spans map back to PDF page sidecars. Each
  row carries its `question_type` (closed taxonomy: factoid, definition, procedural, numeric,
  comparative, multi-hop) and `difficulty` label. When
  `prepare-goldset-draft --retrieval-index-dir <full-rag-index>` is set, each row also carries
  `retrieval_rank` and `retrieval_k`; `retrieval_rank: null` marks a citation-valid needle whose
  gold span was not retrieved from the full corpus within top-k, and the report adds
  `retrieval_unique_needle_fraction_by_question_type`.

The artifacts are diagnostics for review and construction. Drafted rows still remain
`verified=false` until the human verification gate emits an accepted ledger.

The retrieval-uniqueness check is opt-in for generic drafts and enabled by the PDF quickstart after
the full-corpus RAG store exists. Use `DRAFT_RETRIEVAL_INDEX_DIR=<data>/llb/rag` and
`DRAFT_RETRIEVAL_K=<k>` with `make prepare-goldset-draft`; add
`DRAFT_DROP_NONRETRIEVABLE_NEEDLES=1` only when the review artifact should omit misses instead of
flagging them. The report records `needle_retrieval`, `retrieval_unique_needle_items`,
`retrieval_unique_needle_fraction`, and `needle_items_written`. `has_retrieval_unique_needles` is
informational in `gates`; the existing `passed` roll-up still gates on citation-valid needles so
operators can inspect broad-but-grounded misses.

The ontology-assisted seed sampler uses entities, subject-relation-object facts, grounded claims,
and grounded events as draft targets. Seeds carry document, section, difficulty, and semantic-type
coverage strata, so a full-corpus draft can spread questions across manuals, dictionaries, and
after-action-style documents even when a document has few SRO facts.

Three opt-in yield-max knobs raise the meaningful-question yield of a draft: `DRAFT_COVERAGE_TARGET=N`
drafts up to N seeds per stratum bucket (with a `coverage_matrix` exhaustion report) instead of a
flat `DRAFT_MAX_ITEMS` cap; `DRAFT_MULTI_HOP=1` adds multi-span chain questions walked from 2-hop
knowledge-graph paths (each carrying >= 2 grounded spans); and `DRAFT_DEDUP_AGAINST=<bundle[,bundle]>`
drops questions that are pinned-E5 near-duplicates of prior bundles. Every drafted item is tagged
with a `question_type` and `difficulty` label reviewers and the miss analyzer can filter on. See
[robust backends and ontology drafting](robustness-ontology-backends.md) for the module map, report
fields, and command reference.

`DRAFT_MULTI_HOP=1` alone walks strict directed `A -r1-> B -r2-> C` chains, which extracted
Ukrainian PDF graphs rarely supply: the 625-node, 213-edge public-literature graph yields exactly
one such path, so a multi-hop question slice cannot be measured from it.
`DRAFT_MULTI_HOP_BRIDGE_FILL=1` (`--multi-hop-bridge-fill`) keeps directed paths first and then
fills the path budget with the same shared-bridge fact pairs the chain lane uses -- two distinct
facts incident on one entity, cited from two distinct spans. The drafted item still has to name the
bridge or end entity in its reference answer and still has to re-ground >= 2 distinct exact spans,
so the multi-span contract is unchanged; only the supply of candidate paths widens. The strict walk
remains the default, because a directed chain is the stronger multi-hop claim.

### Yield-max empirical acceptance

The 2026-07-12 local acceptance preparation compares coverage-target sampling with the flat
180-seed cap on the public one-document PDF quickstart corpus. Both lanes use `gemma4:e4b`, seed 13,
the same 55-window extraction journal, a 16,384-token context, and the pinned-E5 store at
`$DATA_DIR/llb/rag`; this holds extraction and retrieval constant while changing seed selection.
Artifacts live under `$DATA_DIR/draft-yield-quality-max/20260712T102120Z/`.

| Lane | Raw seeds | Grounded needles | Retrieval-unique needles | Unique fraction |
| --- | ---: | ---: | ---: | ---: |
| Coverage target 6, 240 safety ceiling | 240 | 215 | 194 | 0.9023 |
| Flat cap | 180 | 165 | 149 | 0.9030 |

Both bundles have parse rate 1.0, pass the PDF calibration gates, and pass `validate-goldset`.
Coverage-target sampling therefore prepared 50 more citation-valid needles and 45 more
retrieval-unique needles. Both deterministic 40-row human samples accepted 40/40 items, for equal
1.0 accept rates and 0.0 reject rates at tolerance 0.05. The coverage-target lane therefore passes
the "more citation-valid needles at an equal-or-better accept rate" gate. Both accepted ledgers
pass `validate-goldset` and live under each bundle's `accepted/goldset.jsonl`.

| Question type | Coverage-target unique fraction | Flat-cap unique fraction |
| --- | ---: | ---: |
| Comparative | 0.8889 | 0.8571 |
| Definition | 1.0000 | 1.0000 |
| Factoid | 0.9000 | 0.9123 |
| Numeric | 0.8696 | 0.8500 |
| Procedural | 0.9231 | 0.8889 |
| All types | 0.9023 | 0.9030 |

The drafting contract is Ukrainian-only for user-facing text, including bilingual source corpora:
`question` and `reference_answer` must be Ukrainian, while `answer_span` remains an exact quote in
the source language so evidence offsets stay verifiable. `prep.ontology.draft` and
`prep.ontology.multi_hop` state that foreign evidence must be translated rather than copied into
the reference answer. `src/llb/prep/ontology/language.py` applies a deterministic
Ukrainian-marked, Cyrillic-dominant gate in both flat and multi-hop refinement. The current bundles,
worksheets, and accepted ledgers have zero question/answer violations under that gate; the flat and
coverage runs rejected one and two model outputs respectively for failing it.

The final deterministic 40-row review worksheets are
`$DATA_DIR/draft-yield-quality-max/20260712T102120Z/coverage-target/verify_sample.csv` and
`$DATA_DIR/draft-yield-quality-max/20260712T102120Z/flat-cap-180/verify_sample.csv`. Their acceptance
commands emitted 40-item verified ledgers under the corresponding `accepted/` directories.

### Chain-of-questions artifacts

`src/llb/goldset/chains.py` defines canonical `ChainItem` / `ChainStep` rows for ordered
2-4-step chain-of-questions fixtures. Each step carries a question, reference answer,
dependency note, and exact `SourceSpan` list; `validate_chains` checks duplicate ids, step order,
span offsets, span reuse within a chain, and final-answer leakage from the first step's passage.

`make prepare-goldset-draft DRAFT_CHAINS=1` passes `--chains` to the ontology pipeline. The
pipeline walks the same 2-hop knowledge-graph paths as multi-hop drafting, builds ordered chain
rows in `src/llb/prep/ontology/chains.py`, records `stages.chains` in `provenance.json`, and writes
`<bundle>/chains.jsonl` beside `goldset.jsonl`.

Chain generation keeps strict directed `A -> B -> C` paths first. If that topology does not fill
the requested path budget, it adds exact-grounded pairs of facts incident on the same topic node.
This gives chain review enough candidates on sparse directed graphs without weakening the strict
directed semantics used by flat multi-hop questions. Generated questions and dependency notes are
Ukrainian, matching the chain artifact's `lang=uk` contract.

The five-document PDF chain bundle contains 214 grounded facts. Its strict directed topology yields
9 paths; the shared-topic fallback fills the configured 80-path budget, producing 80 unverified
chains with no dropped rows. `make validate-goldset` passes all 80 chains, calibration gates pass,
and every generated question uses Ukrainian wording. Human verification remains the authority for
whether a shared-topic sequence provides useful progressive context.

The public single-PDF literature corpus under `$DATA_DIR/quickstart-pdf-corpus` converts to one
626,093-character Markdown document with a page-citation sidecar. A local Ollama `gemma4:e4b`
chain draft with a 16,384-token context extracted 301 entities and 213 grounded facts, then wrote
20 flat drafts and 32 unverified chains under
`$DATA_DIR/prepare-goldset/chain-goldset-public-literature`. The extraction parse rate and PDF page
citation coverage are both 1.0, every calibration gate passes, and `make validate-goldset` passes
all 32 chains. The deterministic chain worksheet samples 20 of the 32 candidates at
`$DATA_DIR/prepare-goldset/chain-goldset-public-literature/verify_chains.csv`; its manifest records
`kind=chains`, seed 13, and a single source-document stratum. The converted source and bundle do
not contain the prior restricted corpus markers.

Human review accepted all 20 sampled chains. `make chain-goldset-finalize` enforced the minimum
10-chain gate, required every row to carry `verified=true`, validated the accepted ledger, and
promoted `samples/goldsets/chain_context_uk_v1`. The committed fixture contains 20 chains and a
compact 36-span corpus rather than the complete copyrighted source publication; promotion remaps
every span offset and validates the result before making the destination visible.

#### Complete chain-goldset workflow

Use shell variables once so every later command is short and paste-safe. Select a new bundle path
for each draft run and a destination that does not already exist:

```bash
export DATA_DIR="${DATA_DIR:-$PWD/.data}"
export CHAIN_CORPUS="$DATA_DIR/quickstart-pdf-corpus-md"
export CHAIN_BUNDLE="$DATA_DIR/prepare-goldset/<run-name>"
export CHAIN_WS="$CHAIN_BUNDLE/verify_chains.csv"
export CHAIN_FIXTURE="$PWD/samples/goldsets/<fixture-name>"
```

1. Convert source PDFs to a normalized Markdown corpus. Skip this operation when the input is
   already `.md` or `.txt`:

   ```bash
   make pdf-to-markdown \
     PDF_DIR="$DATA_DIR/quickstart-pdf-corpus" \
     PDF_OUT_DIR="$CHAIN_CORPUS" \
     PDF_PARSER=auto
   ```

2. Run the non-human pipeline shortcut. It drafts chains, requires calibration to pass, validates
   every generated chain against the copied corpus, requires at least `CHAIN_MIN_ACCEPTED`
   candidates, and writes a deterministic review worksheet. A failed stage stops the target
   immediately:

   ```bash
   make chain-goldset-pipeline \
     CHAIN_CORPUS="$CHAIN_CORPUS" \
     CHAIN_BUNDLE="$CHAIN_BUNDLE" \
     CHAIN_WS="$CHAIN_WS" \
     CHAIN_VERIFY_N=20 \
     CHAIN_MAX_PATHS=80 \
     CHAIN_MIN_ACCEPTED=10 \
     DRAFT_MODEL=gemma4:e4b \
     DRAFT_BACKEND=ollama \
     DRAFT_MAX_ITEMS=20 \
     DRAFT_NO_THINK=1 \
     DRAFT_NUM_CTX=16384 \
     DRAFT_TIMEOUT=900
   ```

3. Review the worksheet interactively:

   ```bash
   make verify-review VERIFY_WS="$CHAIN_WS"
   ```

   For every step, compare `A` with `SOURCE`, confirm that `Q` is answered, and confirm that later
   steps use useful context from earlier steps. Reject a chain when its final answer is already
   available from the first step, a cited span does not support its answer, or the dependency is
   artificial. Press `y` to accept or `x` to reject; use `x <code>` for an explicit rejection code,
   `o` for a note, `b`/`u`/`j<N>` to navigate, and `q` to save and quit. Re-running the same command
   resumes at the first undecided row.

4. Emit the accepted ledger after every worksheet row has a decision:

   ```bash
   make verify-accept \
     BUNDLE="$CHAIN_BUNDLE" \
     VERIFY_WS="$CHAIN_WS"
   ```

5. Run the final pipeline shortcut. This replaces any inline Python count check and manual copy:

   ```bash
   make chain-goldset-finalize \
     CHAIN_BUNDLE="$CHAIN_BUNDLE" \
     CHAIN_FIXTURE="$CHAIN_FIXTURE" \
     CHAIN_MIN_ACCEPTED=10
   ```

   Finalization refuses a missing accepted ledger, fewer than the required chain count, any
   `verified=false` row, a structural or span validation error, or an existing destination. On
   success it creates `chains.jsonl`, `corpus/`, and `fixture_manifest.json`, then runs
   `validate-goldset` once more against the promoted fixture.

For an interrupted extraction, resume only the draft stage, then run validation and sampling again:

```bash
make prepare-goldset-draft \
  DRAFT_RESUME="$CHAIN_BUNDLE" \
  DRAFT_NO_THINK=1 \
  DRAFT_NUM_CTX=16384
make validate-goldset \
  CHAINS="$CHAIN_BUNDLE/chains.jsonl" \
  CORPUS="$CHAIN_BUNDLE/corpus"
make verify-sample \
  BUNDLE="$CHAIN_BUNDLE" \
  VERIFY_KIND=chains \
  VERIFY_N=20 \
  VERIFY_WS="$CHAIN_WS"
```

The standard human verification target handles chains without a new command. `VERIFY_KIND=auto`
selects `chains.jsonl` when present; use `VERIFY_KIND=goldset` or `VERIFY_KIND=chains` to force a
mode. Each chain review card starts with 64 `+` characters and renders each step densely as
single-line `Q`, `A`, `SOURCE`, optional `DEPENDENCY`, and truncated `CONTEXT` fields. Questions,
answers, sources, and dependencies use distinct ANSI colors on an interactive TTY; redirected and
test output stays uncolored, and `NO_COLOR` disables color explicitly. The reviewer compares `A`
with `SOURCE`, then checks that `Q` is answered and later steps add context. The same navigation and
note shortcuts remain (`Enter`/`n`, `b`, `u`, `j<N>`, `o`, `?`, `q`). Chain answer edits are
blocked; reject and note the chain when a step needs a different span. `make verify-accept` writes
accepted chain ledgers under `<bundle>/accepted/chains.jsonl` with copied corpus files and
`verified=true`. `src/llb/goldset/promote_chains.py` implements the final acceptance-count,
verification, compaction, offset-remapping, and atomic-promotion gate exposed by
`make chain-goldset-finalize`.

```bash
make verify-sample BUNDLE=<bundle> VERIFY_KIND=chains VERIFY_N=<n>
make verify-review VERIFY_WS=<bundle>/verify_sample.csv VERIFY_ORDER=confidence
make verify-accept VERIFY_WS=<bundle>/verify_sample.csv BUNDLE=<bundle>
```

Unit coverage: `tests/llb/goldset/test_goldset_verify.py` (schema validation, chain worksheet
cards, edit blocking, accepted chain ledger) and `tests/llb/prep/ontology/test_ontology_yield.py`
(graph-path chain construction and draft-bundle emission). Promotion failure modes and compact
corpus offset remapping are covered by `tests/llb/goldset/test_promote_chains.py`.

The local `$DATA_DIR/quickstart-pdf-corpus` corpus run produced 19 markdown files, 19 citation
sidecars, and zero skips under `.data/quickstart-pdf-corpus-md`. Sixteen born-digital PDFs used
PyMuPDF4LLM. The three PDFs that had zero embedded text were recovered by Docling OCR:

| Doc id | Pages | OCR chars | Citation pages |
| --- | ---: | ---: | ---: |
| `pdf-3c3a452a8e9c.md` | 24 | 4,641 | 24 |
| `pdf-3bc34dd5f5c2.md` | 61 | 14,670 | 55 |
| `pdf-3db280e14095.md` | 59 | 11,296 | 58 |

The PDF quickstart validation flow is documented in
[`docs/guides/quickstart/quickstart-pdf-corpus.md`](../../guides/quickstart/quickstart-pdf-corpus.md).
The source PDFs are
under `.data/quickstart-pdf-corpus/`, the full converted markdown corpus is under
`.data/quickstart-pdf-corpus-md/`, and the reviewable draft bundle is under
`.data/quickstart-pdf-corpus-draft/`. The grouped quickstart wrapper is
`make quickstart-pdf-corpus`; it logs conversion, indexing, drafting, graph build, and validation
steps under `$DATA_DIR/llb/logs/quickstart/`. The PDF wrapper passes `QUICKSTART_SKIP_APT` through to
the `pdf-quality` venv step, so hosts that cannot use apt can run with the default
`QUICKSTART_SKIP_APT=1` when the required OCR binaries are already available or the corpus is mostly
born-digital.

`quickstart-pdf-corpus-draft` is the full-quality path, not a small subset. It defaults to
`QUICKSTART_PDF_DRAFT_DOCS=all`, `QUICKSTART_DRAFT_MODEL=auto`,
`QUICKSTART_DRAFT_MAX_ITEMS=180`, `QUICKSTART_DRAFT_VERIFY_N=40`, and
`QUICKSTART_DRAFT_NUM_CTX=16384`. The default `QUICKSTART_MODEL_SELECTION=auto` resolves the
most capable Gemma 4 target from the CUDA serving-tier manifest, filtering out vLLM rows whose
configured `max_model_len` is below `QUICKSTART_DRAFT_NUM_CTX`. On 12 GB hosts the PDF drafter
uses the offloaded vLLM target `google/gemma-4-12B-it-qat-w4a16-ct` with `max_model_len=16384`,
`gpu_memory_utilization=0.90`, `cpu_offload_gb=16`, and `kv_offloading_size_gb=32`. On 16 GB hosts
the same 12B target uses `gpu_memory_utilization=0.85` plus the same context, CPU-offload, and
KV-offload settings. `benchmark`, `choose`, and `frontier` are explicit operator modes when the
host-fit Gemma 4 default is not appropriate. A vLLM pick sets
`QUICKSTART_DRAFT_BACKEND=vllm`; `prepare-goldset-draft` starts `VllmLauncher`, points the local
draft endpoint at `http://localhost:<port>/v1`, and records `endpoint.backend` plus
`endpoint.base_url` in provenance. `--no-think` still works for reasoning models: Ollama uses
native `/api/chat` `think=false`, while vLLM uses OpenAI-compatible `extra_body`
(`chat_template_kwargs.enable_thinking=false`, `include_reasoning=false`,
`reasoning_effort=none`). Fresh non-resume draft runs clear prior extraction journal state in the
output directory before the first model call; only `--resume` reuses journaled windows. The draft
step prints an estimated hour count (character-based, `wc -m`, since Cyrillic UTF-8 bytes would
double it) and requires confirmation before the full ontology/goldset generation starts. The logged
make wrapper cannot prompt inside the tee'd child
process, so unattended full-draft runs require `QUICKSTART_ASSUME_YES=1`; the non-interactive error
prints the exact rerun command. The PDF and mixed-corpus
quickstart wrappers pass `DRAFT_REQUIRE_PASSED_GATES=1`, so a zero-item or ungrounded draft writes
its inspection bundle and then exits non-zero instead of continuing to graph/validation. The wrapper
passes the full PDF RAG store at
`$QUICKSTART_PDF_RAG_DATA/llb/rag` into the needle retrieval-rank annotator. Model scoring remains
gated on `verify-review` and `verify-accept`.

The host selector reads the curated serving manifest for the detected 12/16/24/32 GiB tier. It
considers `gemma-4` and `gemma-4-*` entries, ranks CUDA/vLLM rows ahead of Ollama/offload rows, then
chooses the largest parameter count in that backend class. Long-context callers pass a minimum
context requirement so short-context vLLM cells cannot be selected for PDF drafting. The 12/16 GiB
tiers therefore use the extra `gemma-4-12b-vllm` target with CPU weight/KV offload, while 24/32 GiB
tiers use the primary 31B vLLM target. The complete selection contract and override precedence are
in [the inference configuration guide](../../inference/config-example.md#automatic-cuda-host-draft-model-selection).

The accepted ledger emitted by `verify-accept` contains only the rows a human explicitly accepted
in the worksheet; the complete drafted set (all `goldset.jsonl` rows and the citation-valid
`needle_items.jsonl` subset) stays in the draft bundle at `verified=false`. To enlarge the
verified ledger later, re-draw a bigger worksheet with `make verify-sample VERIFY_N=<n>` and review
it -- no re-draft needed.

Measured on 2026-07-02 (16 GB RTX 4060 Ti host, drafter `batiai/qwen3.6-35b:iq3` via Ollama with
`num_ctx=16384`): a bounded 4-document quick run
(`QUICKSTART_PDF_DRAFT_DOCS="pdf-2ff96d2db393 pdf-3c3a452a8e9c pdf-b117ebb25eb7 pdf-d2e2499d3d06"
QUICKSTART_DRAFT_MAX_ITEMS=80 QUICKSTART_DRAFT_VERIFY_N=20`) drafted 274k chars in 24 minutes:
26 extraction windows at ~48 s each, 80 draft calls at ~3.2 s each, 100 percent extraction parse
rate, 132 entities / 159 facts / 86 claims / 75 events grounded, a 452-seed pool, 70 of 80 drafts
kept (2 circular, 3 duplicate, 5 ungroundable), all 70 citation-valid needles, gates passed. The
full 19-document corpus is 8.0M chars (668 windows), so a `QUICKSTART_DRAFT_MAX_ITEMS=400` full
draft projects to roughly 9-10 hours on this host and about 350 kept items from a roughly
2,000-seed pool.

### Resumable extraction (interrupt-safe drafting)

Because a full-corpus draft is a multi-hour extraction stage, the bundle carries a per-document,
per-window extraction journal (`src/llb/prep/ontology/journal.py`). Each completed window appends
one line to `extraction_journal.jsonl` (keyed by `(doc_id, window_index)`, deterministic from
`split_document`); a settings sidecar `extraction_journal.meta.json` is written at the start of
extraction and pins the determinism-critical settings (corpus, seed, `max_items`, window size and
overlap, retrieval options) plus the endpoint identity.

`llb prepare-goldset-draft --resume <bundle>` (make: `DRAFT_RESUME=<bundle>`;
`make quickstart-corpus QUICKSTART_CORPUS_RESUME=<bundle>`) re-enters an interrupted bundle: it
reads the meta, reuses journaled windows instead of re-calling the model, re-extracts only the
missing windows, and replays the deterministic seed/draft/emit stages. The result is byte-identical
to an uninterrupted run (same seeds, same kept items). Only parsed JSON objects are journaled;
failed calls remain resumable, while parsed empty objects carry the required `parsed=true` marker.
A row without that marker is malformed and is regenerated. A missing meta aborts the resume with a clear
message. The make target also rejects an explicitly empty command-line `DRAFT_RESUME`, preventing
an unset shell variable from silently starting a fresh default draft.

Resume option restoration lives in `src/llb/cli/prep/draft_resume.py`.
`DraftResumeBuilder` starts from the parsed `DraftRequest`, applies the journal's authoritative
corpus and phase routes, preserves explicit CLI overrides, and returns one resolved request. The
execution path consumes that object directly, so request validation has one entry point.

### Frontier ontology draft lane

The ontology pipeline has explicit extraction and drafting endpoint routes. Immutable route data
and phase telemetry live in `src/llb/prep/ontology/endpoint_config.py`; validation and construction
live in `endpoint_builder.py`; local and Litellm transports live in `endpoint.py`; spend/call
enforcement lives in `src/llb/prep/frontier_telemetry.py`.
`prepare-goldset-draft` keeps both phases local by default. A frontier run requires all of:

- `DRAFT_ENDPOINT=frontier` (or `--endpoint frontier`);
- an interactive confirmation that names the corpus path and Litellm destination;
- at least one guard (`DRAFT_MAX_USD` or `DRAFT_MAX_CALLS`); the CLI supplies a 100-call guard
  when neither is given explicitly.

The default frontier route covers both phases. `DRAFT_FRONTIER_STAGE=extraction` or `drafting`
routes only that phase off-box and requires `DRAFT_LOCAL_MODEL` for the other phase. Example:

```bash
make prepare-goldset-draft \
  DRAFT_CORPUS=<corpus-dir> \
  DRAFT_ENDPOINT=frontier \
  DRAFT_FRONTIER_MODEL=<litellm-model-id> \
  DRAFT_FRONTIER_STAGE=both \
  DRAFT_MAX_USD=<usd-cap> \
  DRAFT_MAX_CALLS=<call-cap>
```

`provenance.json` records the configured extraction/drafting routes plus aggregate and per-phase
call counts, measured cost, latency, and per-call telemetry. A call cap is checked before dispatch.
Provider cost is only knowable after a response, so the spend guard stops immediately after the
call that crosses the cap. A budget stop exits nonzero but leaves `provenance.json` with
`status: aborted`, the reason, telemetry, and the extraction journal so the bundle remains
inspectable and resumable.

`make draft-compare` (`src/llb/prep/ontology/compare.py`) runs local extraction once, selects a
bounded deterministic seed set, and drafts those exact seed objects through the local and frontier
routes. It writes self-contained lane bundles, verification worksheets, and
`$DATA_DIR/draft-compare/<timestamp>/comparison.json`. The report includes seed fingerprints,
parse rate, kept yield, calibration gates, and separate rankings for kept yield and verify-sample
accept rate. Accept rate is `pending-human-review` until reviewed worksheets are supplied with
`make draft-compare-report`; that report-only target updates `comparison.json` without calling or
spending on either model again.

The bounded `frontier-ua-draft-lane` probe reuses the committed, repo-authored synthetic
two-document corpus at `samples/text_analysis_bundle_uk/corpus`. Its adjacent `provenance.json`
records the data classification and document hashes. `make frontier-ua-draft-probe` pins this
input, requires an explicit Litellm model, USD cap, call cap, and output root, then presents the
normal corpus-and-destination egress prompt before any provider call.

Human comparison decisions use the same resumable terminal cards as gold-set verification:
`make draft-compare-review DRAFT_COMPARE_OUT_DIR=<comparison-root>` walks the local worksheet and
then the frontier worksheet, saves every decision atomically, and resumes at the first undecided
row. Cross-check/model verdicts remain hidden. `make draft-compare-finalize` refreshes reviewed
accept rates without model calls, records a `finalization` block in `comparison.json`, prints one
`[ok]` or `[fail]` line per acceptance gate, and exits nonzero unless all worksheets, calibration,
parse-rate, ranking, call-cap, and spend-cap checks pass. The lower-level
`make draft-compare-report` remains useful when worksheets live outside their generated paths.

```bash
make draft-compare \
  DRAFT_COMPARE_CORPUS=<corpus-dir> \
  DRAFT_COMPARE_SEEDS=<n> \
  DRAFT_COMPARE_LOCAL_MODEL=<local-model> \
  DRAFT_COMPARE_FRONTIER_MODEL=<litellm-model-id> \
  DRAFT_COMPARE_MAX_USD=<usd-cap>

make draft-compare-report \
  DRAFT_COMPARE_OUT_DIR=<comparison-root> \
  DRAFT_COMPARE_LOCAL_VERIFICATION=<reviewed-local-csv> \
  DRAFT_COMPARE_FRONTIER_VERIFICATION=<reviewed-frontier-csv>

make draft-compare-review DRAFT_COMPARE_OUT_DIR=<comparison-root>
make draft-compare-finalize DRAFT_COMPARE_OUT_DIR=<comparison-root>
```

Deterministic coverage is in
`tests/llb/prep/ontology/test_frontier_lane.py`: refusal occurs before completer construction,
fake-provider spend aborts preserve provenance, mixed phase routes use distinct callables, and the
comparison report fingerprints the shared seeds. No test needs network access or a provider key.

### Sequential local Qwen/Gemma draft comparison

`make local-ua-draft-probe` runs an exact-seed local comparison without data egress. The adaptive
selection policy in `src/llb/prep/ontology/local_compare_models.py` uses the benchmark GPU-tier
detector and selects Qwen/Gemma Ollama pairs for 12, 16, 24, and 32 GiB hosts. Explicit model flags
remain available, but the selected tags must already be installed. The resolved tier, GPU name,
models, context, and override/profile source are recorded under
`comparison.json.execution.resource_selection`.

`src/llb/prep/ontology/local_compare.py` enforces sequential residency: unload all Ollama models,
run the Qwen baseline extraction and drafts, unload Qwen and wait until absent, run Gemma over the
same seed objects, then unload again. `ollama_lifecycle.py` turns unload failure into an error rather
than permitting overlapping model residency. The comparison uses `baseline` and `probe` lane names;
it does not reuse the frontier names or provenance.

The reviewed 16 GiB comparison selected `qwen3:14b` then `gemma4:e4b` with an 8192-token
context. Both parsed 12/12 drafts and passed calibration. Qwen kept 7/12 items (58.3%) in 71.9
seconds of drafting; Gemma kept 5/12 (41.7%) in 31.1 seconds. Gemma matched parse rate, reduced
kept yield by 16.7 percentage points, and was about 2.3 times faster in drafting latency. Human
review accepted every kept item: 7/7 Qwen items and 5/5 Gemma items, both 100%. Finalization passed
all worksheet, calibration, ranking, sequential-execution, and unload checks. This supports Qwen as
the higher-yield baseline and Gemma as the faster probe on this small fixture; reviewed accept rate
does not distinguish them. `ollama ps` was empty after the run, confirming final unload. Runtime
artifacts follow `<comparison-root>/{comparison.json,baseline/,probe/}`.

`make draft-compare-analyze DRAFT_COMPARE_OUT_DIR=<comparison-root>` reads `comparison.json`, both
lane provenance files, and the live worksheets. It prints model order, shared-seed counts, parsed
and kept ratios, calibration, calls, latency, review progress, human accept rates, and lane deltas.
`COMPARE_ANALYZE_JSON=1` emits normalized JSON and `COMPARE_REQUIRE_GATES=1` exits nonzero when a
calibration gate fails. `make draft-compare-review` and `make draft-compare-finalize` accept either
the frontier lane schema or the local `baseline`/`probe` schema.

The dedicated operator aliases keep this workflow on one variable:

```bash
make local-ua-draft-probe LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root>
make local-ua-draft-complete LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root>
make local-ua-draft-analyze LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root> \
  COMPARE_ANALYZE_JSON=1
```

`local-ua-draft-complete` runs the resumable human review, finalization, and final table in order.
The individual `local-ua-draft-review` and `local-ua-draft-finalize` aliases remain available for
operators who prefer one explicit gate at a time.

## Verification Gate

The verification path has a mechanical half and a human half.

```bash
make cross-check-goldset BUNDLE=<draft> CROSS_CHECK_MODEL=<model>
make verify-sample BUNDLE=<draft> VERIFY_N=<n>
make verify-review VERIFY_WS=<worksheet>
make verify-accept VERIFY_WS=<worksheet> BUNDLE=<draft>
```

`src/llb/prep/cross_check.py` checks grounding and non-circularity before calling an injectable
second verifier for support and answerability. A pass means the item is reviewable, not verified.

`src/llb/goldset/verify_sampling/` handles stratification, reviewer context, confidence ordering,
row construction, and worksheet persistence. `verify_base.py`, `verify_acceptance.py`, and
`verify_refcheck.py` own the schema/I/O, acceptance/ledger, and reference-checking seams;
`src/llb/goldset/verify/cli.py` orchestrates the commands. `verify_session/` owns the interactive
terminal loop.
The review session keeps command parsing, navigation, row edits, clear confirmation, and
persistence in small helpers so the loop reads as worksheet orchestration.
The accepted ledger writes copied corpus files plus canonical `verified=true` rows; chain samples
write `accepted/chains.jsonl`, while flat goldset samples write `accepted/goldset.jsonl`.
`prepare-goldset-draft` can also write the first worksheet in the same run with
`--verification-sample-size <n>`; the make wrapper exposes this as `DRAFT_VERIFY_N=<n>`.

The rationale is anti-anchoring and auditability: automated cross-check context can be shown to a
reviewer, but it is hidden by default; the accepted ledger is a new reviewed artifact rather than an
in-place mutation of the draft.

### Reviewer throughput tooling

The review CLI carries the throughput features from the `verify-cli-throughput` task, complete
with the measured human review-pass evidence recorded at the end of this section:

- **Confidence-ordered queue** -- `make verify-review VERIFY_WS=<ws> VERIFY_ORDER=confidence`
  reviews least-confident items first: each cross-check verdict flag contributes +1/-1 and a
  needle `retrieval_rank` contributes `1/rank` (`row_confidence`/`confidence_order` in
  `verify_sampling/confidence.py`). Only the session queue is reordered; the CSV row order never
  changes.
- **On-card evidence** -- worksheet rows (new optional columns `retrieval_rank`,
  `page_citation`) carry the item's needle retrieval rank (joined from `needle_items.jsonl` /
  `item_provenance.jsonl`) and a `<source.pdf> p.N[-M]` citation resolved through the PDF lane's
  `*.citations.json` sidecars, so the reviewer can check the original page without leaving the
  terminal.
- **Ambiguous-evidence guard** -- an optional `span_occurrences` column carries how many times an
  item's primary gold span text appears verbatim across the whole corpus (`span_occurrences.py`).
  An item whose span repeats is ambiguous by construction: the answer text exists in several
  places, the retrieval metric credits any of them, and the reviewer could not otherwise tell that
  the span they are accepting is not unique. The count comes from the draft-time
  `span_occurrences.jsonl` sidecar when present, else a direct corpus scan of the sampled items;
  the review card adds an `== ambiguous evidence: this span text appears in N places ...` line so
  the reviewer decides whether the question is uniquely answerable. The guard fires above one
  occurrence (`OCCURRENCE_FLAG_THRESHOLD`) and only annotates -- it never rejects an item or
  changes the retrieval metric. The column and sidecar are BOTH absent when every sampled span is
  unique, so an all-unique bundle keeps its worksheet byte-for-byte.
- **Accept-with-edit re-grounding** -- the `e` command captures an edited reference answer and
  re-grounds it IMMEDIATELY against the bundle corpus (resolved via `sample_manifest.json`); an
  edit that is not a verbatim corpus span is refused on the spot, an accept over a stale edit is
  blocked, and `emit_accepted_ledger` re-checks authoritatively (raises) so a hand-edited CSV can
  never certify an un-grounded answer. Accepted edits flow into the ledger with the primary span
  replaced by the re-grounded offsets.
- **Additive sample enlargement** -- `make verify-sample BUNDLE=<draft> VERIFY_N=<n>
  VERIFY_MERGE=1` (`merge_sample_worksheet`) enlarges an existing worksheet to ~`n` rows by
  appending only item ids not already present: decided rows are preserved byte-for-byte, never
  re-drawn or re-shown, and re-running the merge is idempotent.
- **Session throughput stats** -- each decision prints a pace line (decided count, items/hour,
  ETA for the remaining rows); the end-of-session summary repeats it and every sitting appends a
  record to `verify_session_stats.json` beside the worksheet (the durable items-per-hour
  evidence the throughput task cites).
- **Coded rejection reasons** -- `x` infers a code from the first failed check
  (`ungrounded`/`circular`/`wrong_reference`/`label_mismatch`), `x <code>` sets one explicitly
  (also `bad_question`, `other`); `make verify-accept` exports the aggregate to
  `rejection_reasons.json` beside the accepted ledger, and the drafting pipeline reads it back
  (draft-feedback-rejection-reasons, below) to tighten its prompts on a re-draft.

All of it is unit-tested with injected input/output/clock in `tests/llb/goldset/test_goldset_verify.py`
(golden-path session tests included). The loader supplies the shared verification columns and
preserves profile-specific columns used by translation and adjudication, so one review engine can
serve each current worksheet profile without separate parsers.

The review card and controls are unified with the external-RAG human review session
(`llb.scoring.external_rag_session`, the origin interface): a `=====` banner, `== field:` labels,
a blank line before `== question:` delimiting consecutive cards, indented multi-line evidence
blocks, a two-line grouped prompt hint, and the shared `o` (note) / `w` (edit answer) keys. The
decision keys intentionally differ (`y`/`x` here) because `a`/`r`/`p` mark the verification
checks. The card layout and key aliases are documented behavior, not test surface -- session
tests cover decisions, re-grounding, merge, and stats, not print formatting.

Measured throughput evidence (2026-07-10, quickstart PDF corpus draft, 69 drafted items): a
single-sitting human pass with `VERIFY_ORDER=confidence` decided all 46 sampled rows in 9.9
minutes -- **279.5 items/h** -- recorded in `verify_session_stats.json` beside the worksheet.
`verify-accept` passed at 44 accepted / 2 rejected (reject rate 0.043 vs tolerance 0.05); both
rejects concentrated in the corpus's one long-manual document, one of them a TOC-mined
page-number question whose row also carried no `retrieval_rank` (the signal the confidence queue
sorts on). Two advisory per-stratum FAIL warnings were small-sample artifacts: at tolerance 0.05
a stratum needs >= 20 decided rows to absorb a single reject, and the flagged cells held 7 and 5.
A bare `x` reject with no failed checks exports `code: other`; marking the failing check first (or
using `x <code>`) keeps `rejection_reasons.json` actionable.

`draw_stratified_sample` allocates through `stratum_quotas`: a floor of one per non-empty stratum
(largest strata first when `n` cannot cover them all) plus a deterministic largest-remainder
top-up, each stratum
capped at its own size -- so `verify-sample VERIFY_N=<n>` draws exactly `min(n, population)`
rows at every seed while staying seeded-reproducible. The sibling `sample_manifest.json`
records the final per-stratum allocation. The allocation invariants are covered in
`tests/llb/goldset/test_goldset_verify.py`.

### Rejection feedback into re-drafting

Rejection feedback can directly guide a new draft:

```bash
make prepare-goldset-draft DRAFT_CORPUS=<dir> \
  DRAFT_REJECTION_FEEDBACK=<bundle>/accepted/rejection_reasons.json
```

`llb prepare-goldset-draft --rejection-feedback <file>` maps each dominant reject code to a
deterministic Ukrainian draft-prompt hint (`src/llb/prep/ontology/feedback.py`; the mapping
covers exactly the closed reject-code set, ordered by rejection count, and each hint carries the
first rejected item's note as an example -- e.g. a `circular`-heavy summary adds an explicit
non-circularity instruction). The combined hint block is appended to the ontology-constraint
line of every draft prompt; an empty summary is a no-op. `provenance.json` gains an
`applied_feedback` block (source path, sha256 digest, applied hint codes + counts), the setting
is pinned in the journal meta so `--resume` replays it, and
`settings.rejection_feedback` names the file. Unit tests:
`tests/llb/prep/ontology/test_draft_feedback.py` (per-code mapping, dominant ordering, no-op
summary, prompt + provenance round-trip over a fake endpoint).

### Multi-annotator gate and adjudication

The verification gate supports more than one annotator plus configurable acceptance arithmetic
(`src/llb/goldset/verify_multi/` + policy extensions in `verify_acceptance.py`; tests in
`tests/llb/goldset/test_verify_adjudication.py`):

```bash
make verify-sample BUNDLE=<draft> VERIFY_N=<n> VERIFY_ANNOTATORS=<k>
make verify-review VERIFY_WS=<bundle>/verify_sample.r1.csv   # each reviewer, own sheet
make verify-adjudicate BUNDLE=<draft>
make verify-review VERIFY_WS=<bundle>/adjudication.csv
make verify-accept VERIFY_WS=<bundle>/verify_sample.csv BUNDLE=<draft> \
  VERIFY_ACCEPT_POLICY=<global|per-stratum|weighted>
```

- **Multi-reviewer sampling** -- `VERIFY_ANNOTATORS=<k>` draws ONE stratified sample and writes
  it as `k` identical per-reviewer worksheets (`verify_sample.r<i>.csv`), each row stamped with
  a `reviewer_id` column, left blank for single-reviewer worksheets.
  The manifest records the reviewer worksheets and intentionally omits the single-`worksheet`
  key, so a multi-reviewer bundle can only stamp `--data-verified` through its accepted ledger.
- **Agreement report** -- `verify-adjudicate` writes `agreement.json` beside the worksheets:
  observed agreement plus Cohen's kappa (2 reviewers) or Fleiss' kappa (3+) over the jointly
  decided rows, per-reviewer decided/accept/reject counts, and the disagreement item ids. A
  unanimous accept whose accept-with-edit answers differ counts as a disagreement (the edit
  changes what the ledger would certify). Metric arithmetic is isolated in
  `verify_multi/agreement_metrics.py`; `AgreementReportBuilder` in `agreement_report.py` indexes
  the worksheets once and builds the report sections.
- **Adjudication pass** -- disagreements are drawn into `adjudication.csv` (exactly those rows),
  human columns blank for an independent decision, prior verdicts carried forward in a read-only
  `prior_decisions` column (`r1=reject:bad_question;r2=accept`) shown on the review card.
  Rebuilding preserves adjudicator decisions already made. The ordinary `verify-review` session
  reviews it unchanged.
- **Consensus acceptance** -- `verify-accept` on a multi-reviewer bundle scores the consensus:
  unanimous decisions stand, adjudicated decisions override disagreements, and anything else
  (a reviewer still undecided, an unadjudicated disagreement) stays undecided and blocks
  acceptance. `ConsensusBuilder` in `verify_multi/consensus.py` owns adjudication preference,
  unanimity checks, and clearing blocked human fields. The accepted ledger and
  adoption-through-ledger invariant are unchanged.
- **Acceptance policies** -- `--policy` (make: `VERIFY_ACCEPT_POLICY=`) selects the arithmetic:
  `global` (the original single-tolerance rule, still the default), `per-stratum` (EVERY stratum
  within its own tolerance; `VERIFY_STRATUM_TOLERANCES="<stratum>=<tol> ..."` overrides cells),
  and `weighted` (confidence-weighted reject rate where a decided row weighs
  `1 + max(row_confidence, 0)` -- a reject on a row the automated signals rated confident counts
  more, because it means those signals mispredict).

Agreement statistics are unit-tested against hand-computed kappa fixtures; the adjudication
draw, each acceptance policy, and the reused-id adoption invariant are covered by synthetic
reviewed fixtures.

## Judge Calibration

Judge calibration is a separate human-rating problem. The code measures whether a local judge
tracks human ratings on the calibration split. The trust gate is Spearman rho `>= 0.6`; below that,
the judge remains diagnostic.

Modules:

- `src/llb/judge/calibration.py`: worksheet IO, Spearman rho, bootstrap CI, trust decision;
- `src/llb/judge/rate/`: command parsing, worksheet state, presentation, and the interactive rater;
- `src/llb/scoring/judge/model.py`: runtime trust gate and judge outcome policy;
- `src/llb/scoring/judge/scorer.py`: score normalization and empty-answer handling.

```bash
make calibration-run
make calibration-rate
make calibration-score
make run-eval JUDGE_RHO=<rho> JUDGE_MODEL=<model> JUDGE_BASE_URL=<url>
```

`calibration-run` pre-fills model answers and optional ungated judge ratings.
`calibration-rate` hides judge ratings by default so the human rating is independent.
It stores only human-owned worksheet columns, supports resume/review navigation, and exits without
editing when the start-fresh clear prompt is not confirmed. The rating session uses the same
parser/navigation/edit-helper shape as verification review.
`calibration-score` computes rho and confidence interval from the filled worksheet.

Tracked calibration worksheets live in `calibration/`. Generated worksheets for temporary corpora
live under `$DATA_DIR/llb/calibration/` unless deliberately promoted.

## Chunking

The `src/llb/rag/chunking/` package keeps every chunk offset-exact. Strategies:

- `fixed`: dependency-free fixed windows;
- `sentence`: dependency-free sentence-aware chunks;
- `recursive`: LangChain recursive splitter when available, pure fallback otherwise;
- `markdown`: heading-aware chunks with breadcrumb metadata;
- `semantic`: pinned-embedder breakpoints while preserving source offsets;
- `page`: PDF page/citation-aware boundaries that never cross a page-sidecar span;
- `heading`: heading-hierarchy packing with heading lines kept in the chunk text;
- `late`: sentence spans embedded by whole-document token pooling (late chunking).

The `page`/`heading`/`late` details, comparison command, and durable evidence live in the
[RAG core](rag-core.md) chunking-strategies section.

```bash
make build-rag-store
python -m llb.rag.chunking --corpus-root <dir> --out-dir .data/llb/rag \
  --strategy markdown --size 800 --overlap 120 --embed
```

Production RAG indexes are built through `llb build-index` or `make build-index`.

## Query Glossary (uk-query-processing)

`llb build-query-glossary --bundle <draft>` (or `make build-query-glossary BUNDLE=<draft>`) turns a
draft bundle's `prompt_dictionary_candidates.jsonl` into a `query_glossary.json` for the query-side
`glossary` step (`src/llb/rag/query_prep/glossary.py` `build_glossary_from_candidates`). Each candidate
`term` becomes a canonical entry carrying its recorded aliases plus a romanized Latin variant
(`--no-transliterations` disables the seeding); entries are sorted by canonical term for a
deterministic artifact. Hand-add more surzhyk / transliteration aliases by editing the emitted JSON
-- the `glossary` retrieval step appends every surface form of any entry the query triggers, never
mutating the stored corpus. A committed fixture lives at `samples/query-prep/` (dictionary
candidates + the generated `query_glossary.json`). The lane's retrieval behavior, A/B report, and
durable deltas live in the [RAG core](rag-core.md) query-side-processing section.

## Corpus Hygiene: Conflict Detection (corpus-conflict-detection)

This lane owns DOCUMENT-level duplication and contradiction, which needs a human decision. The
chunk-level counterpart is automatic and separate: exact-duplicate chunk text inside one index is
collapsed at build time
([duplicate chunk collapse](rag-core.md#duplicate-chunk-collapse)), which changes no corpus byte
and reports its rate in `store_meta.json`.

`llb audit-corpus-conflicts` (`make audit-corpus-conflicts CORPUS=<dir> EFFORT=<tier>`) reports
duplicated, stale, and mutually inconsistent knowledge in a corpus. It is **detection only**: no
tier edits, deletes, or reorders a corpus byte, and a CI test asserts the corpus is unchanged after
a run. Implementation lives in `src/llb/conflicts/`, Typer wiring in `src/llb/cli/prep/conflicts.py`,
Make orchestration in `make/data-prep/corpus.mk`.

### Effort tiers

Four cumulative `--effort` tiers; each settles what it can so the next has less to look at.

| tier | mechanism | needs | cost on the 8-doc / 2578-chunk HR corpus |
| --- | --- | --- | --- |
| `hash` | content sha, raw and Ukrainian-normalized | nothing | 0.26 s |
| `lexical` | word 5-gram shingles: Jaccard + containment | nothing | 0.51 s |
| `semantic` | chunk-vector pair search over a built store | a store | 1.5 s |
| `claim` | local-model adjudication of surviving pairs | a store + a model | 77 s / 11 pairs |

`hash` splits duplicates into `raw` (byte-identical) and `normalized` (identical after casefold,
whitespace, punctuation, apostrophe unification, and front-matter removal) -- the second is the
re-ingested-edition case. Content hashing is deliberately **not** `corpus_doc_fingerprints`, which
folds the governance contract into each document's hash: that is right for refresh and wrong here,
since two byte-identical documents carrying different `effective_date` values must still read as
duplicates.

Duplicate groups are transitive, so the tier reports `n-1` chained pairs for a group of `n` but
marks the group's **full pair closure** as settled. Without that split the later tiers re-derive
(and re-report) the pairs the chaining left implicit.

### Relation vocabulary

Relations are assigned per **claim pair**, never per document: `duplicate`, `subsumes` /
`subsumed_by`, `contradicts`, `superseded_by`, `complementary`. That is what makes partial
supersession representable -- a revision that changes one fact while restating another produces a
`superseded_by` for the first and a `duplicate` for the second, from one document pair.

`superseded_by` is **derived, never asked for**. The adjudication prompt shows the model two
passages and no provenance, so it cannot rationalize a verdict from dates; a `contradicts` verdict
is promoted to `superseded_by` only when the governance fields order the two sides, with side `a`
always the deprecated claim. An undated contradiction stays an honest `contradicts` for a human.

Every finding carries exact character offsets on both sides. The claim tier narrows a finding to
the span the model quoted (via `ground_span`); a quote that cannot be located falls back to the
enclosing chunk and is marked `offsets_exact: false` rather than pointing at text that is not
there. On the HR evidence run 10 of 11 findings narrowed exactly.

### What the semantic tier excludes, and why

Three classes of chunk never pair, all learned from real corpora rather than anticipated:

- **Front matter** -- every ingested document's governance block shares the same keys, so an
  archiving instruction and an appeals regulation match at cosine 0.9 on their `version:` and
  `language:` lines alone.
- **Low-content chunks** (`--min-claim-tokens`, default 25 content tokens, HTML comments stripped)
  -- a converted PDF corpus is full of `<!-- source_pdf ... -->` markers, bare page numbers, and
  stub headings. On the HR corpus these were the single largest source of findings before the
  filter: the top-ranked "conflict" was one page marker against another.
- **Repeated structured metadata blocks** -- `semantic_filter.py` groups claim-sized body chunks
  by their normalized deepest Markdown heading, requires that heading at most once per document,
  then confirms each cross-document pair from corpus-derived shared-token coverage and numeric
  field density. The detector has no language- or publisher-specific vocabulary. Repeated claim
  prose under a shared heading stays comparable when it lacks that record-like structure.

The current HR store excludes 90 of 2578 chunks: 88 low-content chunks plus two publication
records. The goods store excludes 87 low-content chunks and no repeated metadata. `summary.json`
retains `excluded_chunks` and breaks it down as `excluded_front_matter_chunks`,
`excluded_low_content_chunks`, and `excluded_metadata_block_chunks`. The same filtered ordinal set
feeds both null-distribution calibration and semantic candidate generation.

### Encoder anisotropy and centering (measured)

Sentence-encoder spaces are strongly anisotropic, and it changes what a threshold means. Measured
over 2578 real multilingual-E5 chunk vectors:

| | random unrelated pair | findings at cosine 0.9 |
| --- | --- | --- |
| raw E5 space | 0.83 (p95 0.88) | 5185 |
| mean-centered | -0.02 (p95 0.26) | 0 cross-document |

A 0.9 "near-duplicate" threshold in raw E5 space sits barely above the similarity of two completely
unrelated chunks, which is why the uncentered run produced an unusable report. `--center-vectors`
(default on) removes the corpus mean direction first. It is skipped automatically below
`MIN_CENTERING_VECTORS` (50) chunks, where the "mean" is an accident of which few documents are
present rather than an estimate -- the audit logs when it does so.

Because centering rescales similarity, `--cos-threshold` is calibrated for the centered space. On
the HR corpus the useful operating point is ~0.6, not the 0.9 that suits raw-space question dedup.

### Corpus-calibrated cosine threshold (`--max-candidate-pairs`)

Even in the centered space a fixed cosine is not portable: the same row budget lands at ~0.60 on
the HR corpus and ~0.46 on the goods corpus. `src/llb/conflicts/null_distribution.py` (the record),
`null_sampling.py` (how it is measured), and `null_calibration.py` (which knob wins) derive the
cutoff from the distribution of the corpus's own comparable cross-document chunk pairs instead of
asking the operator to sweep for it.

`--max-candidate-pairs N` resolves the per-pair quantile `1 - N/total_pairs`, which over an
exhaustive distribution cuts at the N-th largest similarity. A bare `--cos-quantile` is the wrong
dial to expose because it is a per-PAIR rate, so the rows it admits grow with the pair space: at
the 99.9th percentile over the goods corpus's 74,586 comparable pairs it returned 84 rows, and the
same quantile on a 100k-chunk corpus would return millions. `--cos-quantile` remains as the
low-level escape hatch.

Precedence is `--cos-threshold` > `--cos-quantile` > `--max-candidate-pairs` > the fixed default:
an operator who names a cosine has usually swept for it and is never silently overridden.
Calibration is opt-in; with no knob the fixed `DEFAULT_COSINE_THRESHOLD` still applies.

The distribution is **enumerated exactly** whenever the comparable pair space fits
`MAX_EXHAUSTIVE_PAIRS` (5M); sampling is only the fallback above that. That is not an
optimization. Sampling puts a `1/N` floor under the estimable tail, and the HR corpus lands below
it: against 2.4M comparable pairs, a 200k sample has just one pair above cosine 0.6, so the
estimated tail rate stops moving and the threshold silently pins to the sample maximum (measured:
0.7257 instead of 0.5959). A sampled estimate records `resolvable_quantile` and warns when the
requested tail is finer than it can express. Enumerating 2.4M pairs costs ~2 s.

`summary.json` records the basis under the semantic tier: `cos_threshold`, `cos_threshold_source`,
and a `null_distribution` block with pair counts, the resolved quantile, the `selected_rank`, and
the 0.5/0.9/0.99/0.999/0.9999 tail. `report.md` renders a **Semantic threshold** section, so a
calibrated run is comparable against a swept one by absolute cosine.

**Measured, both quickstart corpora** (recursive/800/120 multilingual-E5 stores, centered space,
exhaustive distributions; the HR store is rebuilt with
`DATA_DIR=<scratch> make build-index CORPUS=<hr-md-dir>` so the goods store survives):

| budget | corpus | comparable pairs | resolved cosine | findings | vs. swept baseline |
| --- | --- | --- | --- | --- | --- |
| 12 | HR | 2,434,651 | 0.5790 | 12 | recovers **8/8** filtered swept-0.6 pairs, adds 4 |
| 50 | HR | 2,434,651 | 0.5349 | 50 | superset of the filtered swept-0.6 pairs |
| 12 | goods | 74,586 | 0.4617 | 12 | swept 0.6 found 0 |
| 50 | goods | 74,586 | 0.3948 | 50 | swept 0.6 found 0 |

What this buys: the knob **bounds output size on any corpus** while resolving a different absolute
cosine per corpus, so the old failure mode -- 5185 rows on HR at a fixed 0.9 in raw space -- cannot
recur, and an operator can size the candidate list to the claim-tier adjudication they can afford.

### Known limitation: there is no independent null

The knob above is a **rank selector, not a statistical guarantee**, and the distinction was
measured rather than assumed. It is documented here because the naming of an earlier iteration
(`--max-false-flags`, framed as a false-positive budget) was wrong, and the same mistake is easy to
make again.

The intent was to model "what do UNRELATED chunk pairs score on this corpus?" and flag pairs above
that tail. The implementation samples random comparable cross-document pairs -- but that population
*contains whatever genuine duplicates the corpus has*. It is therefore not an independent model of
"unrelated"; it is the observed distribution itself. Once the pair space is enumerated exactly, the
null and the observed population are literally the same set, and the consequences are exact:

- Empirical FDR (`expected false / observed`) is **identically 1.000** at every threshold on both
  corpora -- 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8 -- because the numerator and denominator are the
  same count. The statistic carries no information.
- A budget of `N` therefore returns **exactly `N`** pairs on every corpus and every budget
  (verified in `test_candidate_budget_selects_exactly_n_pairs_over_an_exhaustive_distribution`).
  That is a useful, predictable contract; it is just a rank cutoff, and no claim about how many of
  those `N` pairs are real.

The sampled fallback does not escape this: a random subsample of the same population has the same
contamination, only noisier.

Consequences for reading a report:

- No threshold on this corpus geometry can be justified as "statistically significant". The
  semantic tier is a recall-oriented **candidate generator** for the claim tier, and the claim
  tier is what establishes whether a candidate pair is a real conflict. In the original HR run,
  three of 11 pairs touched publication-metadata blocks and four other claim-bearing pairs were
  `complementary`. The structural filter removes the three metadata pairs; it does not relabel the
  four honest non-conflicts as metadata.
- Two swept operating points that look comparable are not. HR's useful 0.6 corresponds to a rank
  cutoff of ~12 pairs; goods' 0.6 corresponds to a rank cutoff of 0. No single budget reproduces
  both, and none can, because the budget is a rank and the two corpora have different amounts of
  real duplication at every rank.

Getting a real false-positive rate needs an independent null -- pairs known a priori to be
unrelated -- which the corpus alone cannot supply. That is open research, tracked as
`conflict-null-model-research` in [plan.md](../plan.md).

### Semantic prefix tree

`src/llb/conflicts/tree.py` builds a centroid tree over chunk vectors by deterministic bisecting
2-means for angular vectors and axis-aligned median splits for projected Euclidean vectors. The
angular tree retains the exact centroid/radius triangle-inequality path used by refresh and
inspection. Select the large-corpus path with `--project-dims` (Make: `PROJECT_DIMS=32`); its PCA
and persistence implementation lives in `projection.py` and `projected_index.py`.

The blocker is exact. Store vectors are unit length, so cosine cutoff `c` is Euclidean distance
`sqrt(2 - 2c)`. PCA is an orthogonal projection and can only shrink pairwise distance. A projected
distance above that cutoff therefore proves that the full-space pair cannot match. Surviving pairs
are confirmed against the original vectors in bounded NumPy batches. Projected rows are
deliberately not L2-normalized: normalization changes pairwise distance and invalidates the
lower-bound proof. A regression test locks that behavior down.

SciPy `cKDTree.query_pairs(..., eps=0)` performs the exact radius traversal in reduced space; this
is not an approximate ANN index. The persisted `SemanticPrefixTree` supplies the same exact query
as a dependency-light fallback. Its Euclidean nodes carry axis-aligned bounds, and leaves are
checked in projected space before full-space confirmation. CI asserts that projected candidates
contain every true match and that confirmed pair identities equal the unprojected blocked scan
across several projection dimensions.

`summary.json` reports `project_dims`, `projected_backend`, `projected_candidate_pairs`,
`projected_pruned_pairs`, `projected_pruning_fraction`, and `full_space_comparisons`. A matching
projection/tree is reused. The source, encoder, centering mode, dimensions, leaf size, and
projection fingerprint control reuse, so incompatible store generations rebuild rather than
querying foreign geometry.

### Needle ambiguity lane

With `--goldset`, the audit adds a second, independent signal: for each gold item it locates the
chunks overlapping the item's gold spans and asks whether any **other** document carries a
near-duplicate of them. A needle answerable from two places is ambiguous -- retrieval has two
defensible answers and whichever it ranks first is luck. The report gives
`non_unique_needle_fraction`. This is derived from the gold set rather than from corpus geometry,
so agreement with the tree's findings is corroboration rather than a restatement of one
measurement.

### Artifacts

`$DATA_DIR/corpus-conflicts/<run>/` holds `findings.jsonl` (one JSON object per claim pair, both
sides with exact offsets -- the machine-readable input a resolution lane consumes), `report.md`
(actionable relations first), `summary.json` (per-tier counts, timings, and parameters), and
`tree_meta.json` (tree geometry plus the embedder fingerprint that pins reuse, since centroids are
only meaningful in the space that produced them). With projected blocking, the resolved store
generation also holds `semantic_tree/projection.json`, `semantic_tree/tree.json`, and
`semantic_tree/tree_meta.json`. The projection JSON carries its own SHA-256 fingerprint.

### Evidence run

CUDA host, RTX 4060 Ti, real multilingual-E5 store, MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M for the
claim tier.

- **HR corpus** (8 docs, 2.77 MB, 2578 chunks): no duplicate or near-duplicate documents at any
  document tier. The original 11-pair claim run at `--cos-threshold 0.6` labelled 1 `duplicate`,
  3 `subsumed_by`, and 7 `complementary` in 77 s. Three of those complementary pairs touched two
  publication-record chunks; the other four were claim-bearing non-conflicts about personnel
  authority, medical leave, and software error handling. With the structural filter, the same
  threshold returns eight claim-bearing pairs and removes exactly those three metadata pairs.
  All eight surviving pairs occur in both the 12- and 50-candidate calibrated runs. The substantive
  findings remain: three documents cover the same "mass-edit personnel cards" procedure, and a
  2008-versus-2022 military-service statute pair is ordered by specificity.
- **Goods corpus** (5 docs, 1139 chunks) with its 19-item gold set: 0 cross-document duplicates and
  `non_unique_needle_fraction` 0.0. The two independent signals agree.

The committed fixture at `samples/corpora/conflicts_uk_v1/` plants one instance of every relation
(byte-identical copy, reformatted reissue, absorbed note, changed deadline, restated section, vague
restatement, unrelated control), plus repeated publication records and a single-occurrence prose
control, so each tier and semantic exclusion reason is asserted against a known answer in CI.

Post-filter CUDA-host evidence (RTX 4060 Ti, multilingual-E5 stores) is under
`$DATA_DIR/corpus-conflicts/20260720T-semantic-metadata-filter-*`. The HR swept, budget-12, and
budget-50 runs and the goods budget-12 and budget-50 runs are the source for the measurements
above.

## Corpus Conflict Resolution (corpus-conflict-resolution)

`llb resolve-corpus-conflicts` and the `make resolve-corpus-conflicts` alias turn an audit
`findings.jsonl` into `plan.json`, `conflict_overlay.json`, `resolution_review.jsonl`, and
`effect.md`. The implementation is split across `src/llb/conflicts/resolution_policy.py`,
`resolution_io.py`, `overlay.py`, and `resolution_effect.py`; Typer wiring lives in
`src/llb/cli/prep/conflict_resolution.py`.

The policy is deliberately narrower than the detector:

- hash, lexical, and claim-adjudicated duplicates may use `drop_duplicate`;
- `prefer-newer` may suppress an older `superseded_by` claim only when the recorded governance
  pair orders the editions;
- contradictions, unknown relations, conservative supersession, and every semantic-tier
  duplicate candidate become `escalate` records;
- complementary and subsumption findings remain `keep_both` annotations.

The semantic guard is important: the semantic tier is a recall-oriented candidate generator, not
deletion authority. Its rank-selected goods candidates coexist with the claim-level finding that
the corpus has no confirmed cross-document duplicate. Automatic suppression at that tier would
convert similarity rank into destructive policy.

### Overlay and rollback contract

Applying a plan validates every document, offset, and exact quote against the current corpus, then
atomically installs `.llb/conflict_overlay.json` below the corpus root. A stale audit is rejected
before any directive is installed. Source `.md` and `.txt` bytes are never edited.

`chunk_corpus` consumes the control file. Whole-document duplicate directives omit that document;
claim-level directives omit chunks overlapping the accepted span; keep/escalate records add
`conflict_resolutions` metadata. `corpus_doc_fingerprints` folds each document's directive into
only that document's fingerprint. The existing `refresh_vector_store` path therefore publishes a
normal immutable generation and can reuse vectors when an overlay changes annotations only.
Removing the control file and running with `ROLLBACK=1` publishes the inverse generation and
restores the previous ranking.

The resolver can refresh and measure in one invocation:

```bash
make resolve-corpus-conflicts FINDINGS=<findings-jsonl> CORPUS=<corpus-dir> \
  POLICY=conservative APPLY=1 STORE=<store-dir> GOLDSET=<goldset-jsonl>
```

Pass `BEFORE_RUN=<run-dir>` and `AFTER_RUN=<run-dir>` to add their manifest objective scores to
`effect.md`. Retrieval measurements persist in `effect.json`, so a later report update retains the
same recall/MRR comparison. Roll back with:

```bash
make resolve-corpus-conflicts ROLLBACK=1 CORPUS=<corpus-dir> \
  STORE=<store-dir> GOLDSET=<goldset-jsonl>
```

### CUDA-host resolution evidence

The goods quickstart evidence bundle is under
`$DATA_DIR/corpus-conflicts/20260720T-resolution-goods/`. It used the 1,139-chunk hybrid
multilingual-E5 store, the 19-item flat retrieval set, and the 20 human-accepted chain set with
MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M on the RTX 4060 Ti. The fixed history-policy objective run had
40 scored steps and reliability 1.000 before and after.

All 12 semantic candidates escalated to review; none received suppression authority. Applying the
annotation-only overlay reused all 1,139 vectors. Recall@10 stayed 0.8421, MRR stayed 0.5342, and
the verified final-chain objective stayed 0.6163 (all deltas 0.0000). One-command rollback and
re-apply each published a fresh generation and reproduced the same metrics exactly. The report
verdict is `REVERT` because 12 review decisions remain; unchanged metrics do not override an open
human gate. The failed endpoint probe under `before/` is excluded; the reliable baseline and
post-overlay manifests are under `before-valid/` and `after-valid/`.

Tests in `tests/llb/conflicts/test_resolution*.py` cover policy, semantic escalation, stale-source
rejection, artifact/CLI output, per-document fingerprint changes, source-byte preservation,
review-ledger decisions, all-keep ranking identity, rollback identity, objective loading, and
verdict gating. The current `make ci` passes with 1,687 tests, one skipped, and 42 slow tests
deselected.

### Large-corpus blocking evidence

On 2026-07-20, the 32-dimensional exact blocker was compared with `VectorSet.pairs_above` on both
real multilingual-E5 quickstart stores at cosine 0.9. Findings were byte-identical:

- HR: 2,578 chunks and 3,321,753 possible pairs; 4,642 reached full-space confirmation, so the
  projection pruned 99.8603%. The reused projected search took 0.150 s versus 0.073 s for the
  all-pairs matrix scan. The small corpus remains below the crossover where tree setup pays.
- Goods: 1,139 chunks and 648,091 possible pairs; 1,461 reached confirmation, so the projection
  pruned 99.7746%. Search took 0.116 s versus 0.007 s for the small all-pairs scan.

The required large run used 50,000 deterministic synthetic unit-vector chunks (64 source
dimensions, 32 projected dimensions) on the RTX 4060 Ti CUDA host. It covered 1,249,975,000 pairs,
sent 51 to confirmation, pruned 99.999996%, and returned the same zero matches as the actual
all-pairs baseline. The cold path (PCA fit, persisted-tree build, exact query, confirmation) took
9.306 s; the reusable search path took 7.571 s; the full blocked matrix baseline took 13.988 s.
This evidence includes construction cost rather than reporting query time alone.

Run the delivered path with:

```bash
make audit-corpus-conflicts CORPUS=<corpus-dir> STORE=<store-dir> \
  EFFORT=semantic PROJECT_DIMS=32
```
