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

The make alias defaults `PDF_DIR` to `$DATA_DIR/quickstart-pdf-corpus`. When `out-dir` is omitted,
the default is `<pdf-dir>/_md`, for example `.data/quickstart-pdf-corpus/_md`. Each successful
document gets a `pdf-<digest>.citations.json` sidecar with source PDF, parser, PDF diagnostics, page
numbers, generated-corpus character spans, and page-local block spans when the parser exposes them.
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
fingerprint; keep a previous `$DATA_DIR/llb/rag` directory to roll back an index.

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
`QUICKSTART_DRAFT_NUM_CTX=16384`. The default `QUICKSTART_MODEL_SELECTION=gemma4` resolves the
most capable Gemma 4 target from the CUDA serving-tier manifest, filtering out vLLM rows whose
configured `max_model_len` is below `QUICKSTART_DRAFT_NUM_CTX`. On 12 GB hosts the PDF drafter now
uses the offloaded vLLM target `google/gemma-4-12B-it-qat-w4a16-ct` with `max_model_len=16384`,
`gpu_memory_utilization=0.90`, `cpu_offload_gb=16`, and `kv_offloading_size_gb=32`. On 16 GB hosts
the same 12B target uses `gpu_memory_utilization=0.85` plus the same context, CPU-offload, and
KV-offload settings. `legacy-auto` still consumes existing `llb recommend` JSON when present, and
`benchmark`, `choose`, and `frontier` remain explicit operator modes. A vLLM pick sets
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
now prints the exact rerun command instead of suggesting a model pin. The PDF and mixed-corpus
quickstart wrappers pass `DRAFT_REQUIRE_PASSED_GATES=1`, so a zero-item or ungrounded draft writes
its inspection bundle and then exits non-zero instead of continuing to graph/validation. The wrapper
passes the full PDF RAG store at
`$QUICKSTART_PDF_RAG_DATA/llb/rag` into the needle retrieval-rank annotator. Model scoring remains
gated on `verify-review` and `verify-accept`.

The host Gemma 4 selector ranks CUDA/vLLM rows ahead of larger Ollama/offload rows, then chooses
the largest Gemma 4 parameter count within that backend class. Long-context callers pass a minimum
context requirement so short-context smoke cells cannot be selected for PDF drafting. The 12/16 GB
tiers therefore use the extra `gemma-4-12b-vllm` target, while 24/32 GB tiers use the primary 31B
vLLM Gemma 4 target.

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
failed calls remain resumable, while parsed empty objects carry an explicit marker. Legacy empty
rows without that marker are regenerated because the older format could not distinguish a valid
empty object from a parse or transport failure. A missing meta aborts the resume with a clear
message. The make target also rejects an explicitly empty command-line `DRAFT_RESUME`, preventing
an unset shell variable from silently starting a fresh default draft.

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

`src/llb/goldset/verify.py` handles stratified sampling, worksheet IO, acceptance arithmetic, and
accepted-ledger emission. `src/llb/goldset/verify_session.py` owns the interactive terminal loop.
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
  `verify.py`). Only the session queue is reordered; the CSV row order never changes.
- **On-card evidence** -- worksheet rows (new optional columns `retrieval_rank`,
  `page_citation`) carry the item's needle retrieval rank (joined from `needle_items.jsonl` /
  `item_provenance.jsonl`) and a `<source.pdf> p.N[-M]` citation resolved through the PDF lane's
  `*.citations.json` sidecars, so the reviewer can check the original page without leaving the
  terminal.
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
(golden-path session tests included); the worksheet CSV stays backward compatible -- the new
columns are optional and appended on load for older worksheets.

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
Operational notes from the pass: the stratified draw undershot the requested `VERIFY_N` at the
time (40 -> 39; the additive `VERIFY_MERGE=1` lane topped it up), and a bare
`x` reject with no failed checks exports `code: other` -- marking the failing check first (or
`x <code>`) keeps `rejection_reasons.json` actionable.

The undershoot itself is fixed (verify-sample-exact-allocation): `draw_stratified_sample` now
allocates through `stratum_quotas` -- a floor of one per non-empty stratum (largest strata first
when `n` cannot cover them all) plus a deterministic largest-remainder top-up, each stratum
capped at its own size -- so `verify-sample VERIFY_N=<n>` draws exactly `min(n, population)`
rows at every seed while staying seeded-reproducible. The sibling `sample_manifest.json`
records the final per-stratum allocation as before. Unit-tested against the undershooting
7/7/6-strata fixture in `tests/llb/goldset/test_goldset_verify.py`.

### Rejection feedback into re-drafting

The feedback loop no longer ends at the JSON file (draft-feedback-rejection-reasons):

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
(`src/llb/goldset/verify_multi.py` + policy extensions in `verify.py`; tests in
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
  a `reviewer_id` (a new optional column, appended last so older worksheets stay compatible).
  The manifest records the reviewer worksheets and intentionally omits the single-`worksheet`
  key, so a multi-reviewer bundle can only stamp `--data-verified` through its accepted ledger.
- **Agreement report** -- `verify-adjudicate` writes `agreement.json` beside the worksheets:
  observed agreement plus Cohen's kappa (2 reviewers) or Fleiss' kappa (3+) over the jointly
  decided rows, per-reviewer decided/accept/reject counts, and the disagreement item ids. A
  unanimous accept whose accept-with-edit answers differ counts as a disagreement (the edit
  changes what the ledger would certify).
- **Adjudication pass** -- disagreements are drawn into `adjudication.csv` (exactly those rows),
  human columns blank for an independent decision, prior verdicts carried forward in a read-only
  `prior_decisions` column (`r1=reject:bad_question;r2=accept`) shown on the review card.
  Rebuilding preserves adjudicator decisions already made. The ordinary `verify-review` session
  reviews it unchanged.
- **Consensus acceptance** -- `verify-accept` on a multi-reviewer bundle scores the consensus:
  unanimous decisions stand, adjudicated decisions override disagreements, and anything else
  (a reviewer still undecided, an unadjudicated disagreement) stays undecided and blocks
  acceptance. The accepted ledger and adoption-through-ledger invariant are unchanged.
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
- `src/llb/judge/rate.py`: interactive human rater;
- `src/llb/scoring/judge.py`: runtime gated judge scoring.

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

`src/llb/rag/chunking.py` keeps every chunk offset-exact. Strategies:

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
`glossary` step (`src/llb/rag/query_prep.py` `build_glossary_from_candidates`). Each candidate
`term` becomes a canonical entry carrying its recorded aliases plus a romanized Latin variant
(`--no-transliterations` disables the seeding); entries are sorted by canonical term for a
deterministic artifact. Hand-add more surzhyk / transliteration aliases by editing the emitted JSON
-- the `glossary` retrieval step appends every surface form of any entry the query triggers, never
mutating the stored corpus. A committed fixture lives at `samples/query-prep/` (dictionary
candidates + the generated `query_glossary.json`). The lane's retrieval behavior, A/B report, and
durable deltas live in the [RAG core](rag-core.md) query-side-processing section.
