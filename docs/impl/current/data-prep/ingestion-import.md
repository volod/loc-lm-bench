# Ingestion

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md).

An external-draft import writes its sidecars through registered contracts
(`llb.external-draft-provenance`, `llb.external-draft-item`); the operator-supplied
`external_provenance.json` stays unregistered because an external service authors it. See
[data-prep contracts](../artifact-contracts/data-prep-contracts.md).

`src/llb/prep/squad/ingest.py` maps SQuAD-like rows to `GoldItem` records. It accepts local JSON,
Hugging Face rows, flattened rows, nested article rows, and rows whose `answers` value is encoded
as a dict string.

```bash
make ingest-uk-squad GOLDSET_MODE=development
make ingest-uk-squad GOLDSET_MODE=skeleton
make ingest-uk-squad GOLDSET_MODE=draft CORPUS=<dir>
make ingest-squad SQUAD_JSON=path.json
python -m llb.prep.squad.ingest --hf-dataset <id> --hf-split train
```

Draft imports start with `verified=false`. A verification ledger can adopt matching canonical rows
by id. Adoption replaces the whole canonical item and corpus file, which prevents a reused id from
certifying changed content.

`src/llb/prep/goldset/skeleton.py` writes an editable from-scratch SQuAD template under
`$DATA_DIR/goldset-skeleton/<timestamp>/`.

For **open** corpora, drafts can also be authored with an external AI provider service (Claude
Projects, Gemini/NotebookLM, ChatGPT Projects) and imported either as SQuAD-shaped context docs
(`make ingest-squad`, Artifact A) or, for full-document needle realism, as corpus-grounded JSONL
(`make import-external-draft`, Artifact B). Restricted or private corpora stay on the local ontology
pipeline -- egress is never the default. The workflow, per-service setup, copy-paste prompts, and the
exact artifact shapes are in
[`docs/guides/data-prep/external-ai-service-artifacts.md`](../../../guides/data-prep/external-ai-service-artifacts.md),
[`docs/guides/data-prep/external-service-prompts/`](../../../guides/data-prep/external-service-prompts/README.md),
and the [external-service draft contract](../../../design/external-draft-contract.md).

## Grounded-JSONL import (Artifact B -> draft bundle)

`make import-external-draft` / `llb import-external-draft`
(`src/llb/prep/goldset/external_draft.py`) turns a grounded-JSONL export (contract Artifact B:
`quote` + `source_doc_id` rows) into a canonical draft bundle for the usual `validate-goldset` ->
`cross-check-goldset` -> `verify-*` chain. Unlike `ingest-squad` -- which stamps `provenance:
public-reused`, hashes each context into its own doc (losing full-document needle realism), and
cannot read grounded JSONL -- the import re-grounds against the FULL original corpus doc:

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

## External-draft curation (merge / dedup / filter)

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
local RAG runs. See [RAG core](../rag-core.md) external answer log scoring and
[`docs/guides/data-prep/goldset-from-scratch.md`](../../../guides/data-prep/goldset-from-scratch.md).

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

### Intra-document repeated-block handling (`--repeat-blocks`)

A converted manual also repeats whole BLOCKS inside the one document -- a boilerplate procedure step
restated in section after section, a note repeated under every table -- which the per-line furniture
pass cannot see and which index-time [duplicate chunk
collapse](../rag-core/retrieval-store.md#duplicate-chunk-collapse) can only hide, not fix at the
source: collapse indexes the block once but still returns that one copy for a question about any
section that carries it, and the document's own chunk ordinals stop tracking its reading order.
Measured on the goods corpus every one of the 494 exact chunk-collapse groups is intra-document (0
cross-document), the largest block repeating 335 times in the single 637 KB manual -- so on this
corpus the repetition is entirely a conversion-side property of one document, not shared page
furniture.

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
three lanes score one item set; floor `+/-0.000` throughout):

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

#### Per-question yield audit (`audit-repeat-yield`)

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

Measured on the goods corpus (CUDA host, pinned e5-base, `size=200`, k=10, all 95 items):

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

##### Straddle recovery (`--recover-straddle`)

A straddling gold span is `<tail of a removed copy> + <head of the block after it>`; the removed
copy's text still exists on the survivor and the following block stays in place, so the span is not
truly lost -- it just maps to two non-contiguous images. `remap_span_split` (`--recover-straddle`
on `strip-corpus-repeats` and `audit-repeat-yield`, `REPEAT_RECOVER=1`) splits the span at every
edit boundary it crosses, re-anchors each piece (the removed part onto the survivor, the kept part
by shift), and keeps the item with several spans instead of dropping it. Because `recall_at_k`
credits an item when ANY of its spans is covered, the split preserves the original retrieval
semantics, and each piece is verified against the stripped text so an off-by-one remap fails loudly.

Re-run with recovery on (same corpus, k, splits):

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
[RAG core](../rag-core.md) retrieval-store section for the join and its guarantees.
