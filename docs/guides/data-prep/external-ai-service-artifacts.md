# Creating test artifacts with AI provider services (open data)

This manual is a step-by-step procedure for producing loc-lm-bench test artifacts -- goldsets,
prompts, chain scenarios, security-test scenarios, and local run results -- with the help of
external AI provider services:

- **Claude Projects** -- <https://claude.ai/projects>
- **Google NotebookLM** -- <https://gemini.google.com/notebook> (a.k.a. notebooklm.google.com)
- **Google Gemini (Gems)** -- <https://gemini.google.com> (a custom Gem holds the instructions)
- **ChatGPT Projects** -- <https://chatgpt.com/projects>

Any other assistant qualifies when it can hold project instructions, ground its answers in
uploaded files, and return raw JSON. These services support Ukrainian well and can lift the
quality and completeness of draft test sets. This manual explains **when** you may use them,
**exactly what** to produce, and **how** to bring the results back into the local benchmark.

> **The benchmark's purpose is to evaluate LOCAL RAG and LOCAL LLM inference.** External assistant
> services in this manual are used only to *draft candidate test data*. They do not set
> `verified=true`, judge, or produce local leaderboard rows. A separate diagnostic lane can score
> an already-answered external RAG answer log when the thing under analysis is the external RAG
> product itself.

Read this page top to bottom on first use: the [workflow at a glance](#workflow-at-a-glance)
shows the whole sequence and which steps are yours, the
[step-by-step command chain](#step-by-step-the-command-chain-and-quality-gates) is the
copy-paste summary, and the detailed sections after it explain each step. Do NOT jump straight
into a service's project setup -- steps 1-2 happen locally first, and skipping them (for example
uploading original PDFs instead of the staged corpus) silently breaks quote re-grounding, and the
import will later drop most rows.

## The one rule that gates everything: open data only

External AI services are permitted **only for open data** -- documents that are public or have
been explicitly cleared for third-party processing. Uploading a document to Claude, NotebookLM,
or ChatGPT publishes its contents to that provider and it may be retained.

**If any part of the corpus is restricted or private, the local flow is the main approach** --
not a fallback. Draft locally with the ontology-assisted pipeline running an open-weights model
on your own GPU:

```bash
make quickstart-pdf-corpus            # full local PDF -> draft bundle pipeline
make ingest-uk-squad GOLDSET_MODE=draft CORPUS=<dir> DRAFT_MODEL=<local-tag>
```

The local lane produces the same artifact shapes this manual covers, with **zero data egress**.
Use it whenever classification is uncertain. This restates the settled
[data-egress policy](../../impl/current/scope-boundaries.md#data-egress).

Decision:

```text
Is EVERY document in the corpus public / cleared for third-party upload?
|-- No / unsure -> LOCAL flow only (make quickstart-pdf-corpus). Stop; do not read further.
'-- Yes --------> External-service flow below is allowed as a drafting aid.
```

## Workflow at a glance

Nine core steps plus an optional external-RAG diagnostic; the left column runs on your box, the
right column inside the external service. Steps marked `[HUMAN]` need your attention and judgment
-- everything else is a command.

```text
LOCAL (your box)                              EXTERNAL SERVICE (open data only)
------------------------------------------   ------------------------------------------
1. Stage the corpus: PDFs -> staged .md/.txt
   + manifest with doc ids  [HUMAN: confirm
   every document is open data]
2. Read corpus stats; size the drafting run
        |
        |  upload the STAGED files only ->    3. Create project / notebook / Gem; paste
        |                                        instructions 00; attach staged files +
        |                                        doc-id list         [HUMAN: per service]
        |                                     4. Run prompt 01 (inventory), then 02/03/04
        |                                        in batches (NotebookLM <=15 items)
        |                                        [HUMAN: drive the
        |  <- export each raw reply              chats, export every reply to a file]
        v
5. Write the external_provenance.json sidecar    [HUMAN: required before any import]
6. Curate: merge + repair + filter + dedup (make curate-drafts)
7. Import + structural validation (make ingest-squad / import-external-draft /
   validate-goldset)
8. Cross-check + human verification gate (make cross-check-goldset, verify-*)
                                                 [HUMAN: review the stratified sample]
9. Local scored runs (make build-index -> run-eval / bench-security)
```

The irreducible human actions are: the open-data decision (step 1), driving the service chats
(steps 3-4), authoring the provenance sidecar (step 5), and the verification review (step 8).
Steps 6-7 and 9 are mechanical commands with built-in quality gates.

## Step-by-step: the command chain and quality gates

The condensed chain for an operator who already knows the flow. Each step links to its detailed
section below; run the steps in order and check each gate before moving on.

1. **Stage the corpus** ([details](#step-1-stage-the-corpus-for-upload)):

   ```bash
   make pdf-to-markdown PDF_DIR=<pdf-dir> PDF_OUT_DIR=<staged-dir>   # PDFs
   # or: make ingest-corpus CORPUS_ROOT=<mixed-dir> CORPUS_OUT_DIR=<staged-dir>
   ```

   Gate: every document open / cleared; keep the manifest and note the doc ids.
2. **Size the drafting run** ([details](#step-2-read-the-corpus-statistics-and-size-the-run)):

   ```bash
   ls <staged-dir>/*.md | wc -l && wc -m <staged-dir>/*.md | sort -n
   ```

   Gate: a per-document item budget exists (roughly one item per 2-4k characters).
3. **Set up each service** ([details](#step-3-configure-the-service-project)): create the
   project, paste `00-project-instructions.md`, attach the STAGED files, paste the doc-id list.
   Gate: the model can name your documents exactly.
4. **Run the prompts** ([details](#step-4-run-the-prompts-and-export-the-replies)): `01` once per
   service, then `02`-`04` against the (merged) inventory, in batches; for NotebookLM prompt 02,
   use an uploaded coverage text source and request at most 15 items. Save every reply to its own
   file. Gate: each reply is raw JSON in one code block.
5. **Write the provenance sidecar** ([details](#step-5-write-the-provenance-sidecar)): author
   `external_provenance.json` beside the exports. Gate: sidecar exists and says
   `"data_classification": "open"` -- the importer refuses to run without it.
6. **Curate** ([details](#step-6-curate-the-merged-exports)):

   ```bash
   make curate-drafts CURATE_KIND=<squad|grounded|security|chains|inventory> \
     CURATE_INPUTS="<export> <export> ..." CURATE_OUT=<merged> CURATE_CORPUS=<staged-dir>
   ```

   Gate: read `*.curation_report.json`; a high invalid count from one service means its session
   drifted -- tighten and redraft.
7. **Import + validate** ([details](#step-7-import-and-validate-each-artifact)):

   ```bash
   make ingest-squad SQUAD_JSON=<merged-goldset>          # SQuAD goldset (Artifact A)
   make validate-goldset GOLDSET=<canonical.jsonl> CORPUS=<corpus-dir>
   # grounded JSONL (Artifact B): make import-external-draft ARTIFACT= CORPUS= SIDECAR=
   ```

   Gate: skip/drop counts near zero; a high skip count means the model paraphrased.
8. **Cross-check + human verification gate** ([details](#step-8-human-verification-gate)):

   ```bash
   make cross-check-goldset BUNDLE=<bundle> CROSS_CHECK_MODEL=<second-model>
   make verify-sample  BUNDLE=<bundle> VERIFY_N=30
   make verify-review  VERIFY_WS=<bundle>/verify_sample.csv
   make verify-accept  BUNDLE=<bundle> VERIFY_WS=<bundle>/verify_sample.csv
   ```

   Gate: reject rate within tolerance; only accepted items flip to `verified: true`.
9. **Run the local benchmark** ([details](#step-9-produce-local-run-results)):

   ```bash
   make build-index GOLDSET=<verified.jsonl> CORPUS=<corpus>
   make run-eval    MODEL=<local-tag> BACKEND=<ollama|vllm|llama.cpp> GOLDSET=<verified.jsonl>
   make bench-security SECURITY_CASES=<verified-cases.json> SECURITY_MODEL=<local-tag>
   ```

   Gate: retrieval `recall@10 >= 0.8` before reading any model ranking.

## What you produce and where it goes

Every artifact below is defined precisely by the
[external-service draft contract](../../design/external-draft-contract.md). The copy-paste
prompts are in [`external-service-prompts/`](external-service-prompts/README.md).

| Artifact | Prompt | Curate (merge/dedup/filter) | Import / run command | Status |
| --- | --- | --- | --- | --- |
| Ontology / topic inventory | `01` | `make curate-drafts CURATE_KIND=inventory` | steers `02`-`04` (not scored) | works today |
| Goldset draft (SQuAD JSON) | `02` | `make curate-drafts CURATE_KIND=squad` | `make ingest-squad SQUAD_JSON=` | works today |
| Goldset draft (grounded JSONL) | `02` | `make curate-drafts CURATE_KIND=grounded` | `make import-external-draft ARTIFACT= CORPUS= SIDECAR=` | works today |
| Chain-of-questions draft | `03` | `make curate-drafts CURATE_KIND=chains` | blocked on `chain-goldset-generation` | review-only |
| Security cases | `04` | `make curate-drafts CURATE_KIND=security` | `make bench-security SECURITY_CASES=` | works today |
| Local run results | -- | -- | `make run-eval` / `make bench-security` | works today |
| External RAG answer log | -- | -- | `make score-external-rag EXTERNAL_RAG_ANSWERS=` | diagnostic |

Session outputs live under `$DATA_DIR/external-drafts/<service>-<YYYYMMDD>/`, each with an
`external_provenance.json` sidecar (contract section 6). Nothing external ever sets
`verified: true`; every drafted item is reviewed locally before it can score a model.

## Step 1: Stage the corpus for upload

Upload the **staged corpus text**, not the original PDFs -- the drafting model must quote the same
text the local RAG index will contain, so quotes re-ground cleanly on import.

```bash
# PDFs -> canonical .md corpus (born-digital via PyMuPDF4LLM, image-only via Docling OCR)
make pdf-to-markdown PDF_DIR=<pdf-dir> PDF_OUT_DIR=<out-dir>
# .md / .txt corpora are already staged; use the files as-is.
```

Note the produced `.md` file ids (`pdf-<digest>.md`) and keep `pdf_corpus_manifest.json`; you
record the doc ids and the manifest sha256 in the sidecar. Upload the `.md`/`.txt` files to the
service. For a large corpus, upload in themed groups so the model keeps every document in context.

## Step 2: Read the corpus statistics and size the run

Before opening the first chat, look at what the staging step produced -- the numbers decide how
many items to request per document and where the coverage risk is:

```bash
ls <staged-dir>/*.md | wc -l                       # document count
wc -m <staged-dir>/*.md | sort -n                  # characters per staged document
python -m json.tool <staged-dir>/pdf_corpus_quality.json | head   # pages, image-only pages, skips
```

Real mixed corpora (regulations + system manuals + operational how-tos) follow a characteristic
profile, measured on two real reference corpora from different domains (one asset-management,
one HR/records; 5 and 8 documents, roughly 0.7M and 1.6M staged characters):

- **Extreme size skew.** One or two documents -- a full system manual, a long legal act -- carry
  40-93 percent of all characters; the median document is 100x smaller. A flat per-document item
  quota either starves the big manual or pads the one-page notes with weak questions.
- **A long tail of tiny documents.** How-to notes and support dialogs of 0.5-7k characters still
  carry unique facts (they are often the only document answering an operational question), so
  they deserve a small guaranteed budget rather than exclusion.
- **A few image-only or scanned pages.** The quality report marks them; their OCR text is where
  verbatim-quote drafting fails most, so spot-check quotes drafted from those documents.

Size the per-document request `<N>` from characters (the same table as prompt `02`): about one
item per 2-4k characters, floor 3-5 for tiny documents, batched section by section above ~150k
characters. For the whole corpus this lands at roughly 200-600 requested items per 1M staged
characters; request the full budget from EACH service you use -- overlap is removed at curation,
and the merged union covers more of the inventory than any single service reaches.

## Step 3: Configure the service project

Prerequisites from steps 1-2: the staged `.md`/`.txt` files and the doc-id list with sizes.
Only then open the service:

1. **Create a project** (Claude Projects / ChatGPT Projects), a **notebook** (NotebookLM), or a
   **custom Gem** (Gemini) -- the exact per-service steps are in the
   [prompt pack README](external-service-prompts/README.md#per-service-setup-one-time-per-corpus-project).
2. Paste [`00-project-instructions.md`](external-service-prompts/00-project-instructions.md) into
   the service's instructions field (or as the first message where none exists), then attach the
   staged corpus files and paste the doc-id list with sizes.

Service notes:

- **NotebookLM** is grounded in the uploaded sources by design, which suits inventory and QA
  drafting; ask it to output the same JSON shapes. It is more conversational -- paste the
  manifest doc-id list and state that every `doc` and `cross_document[].docs` value must be the
  staged `.md`/`.txt` id from the manifest, not the original PDF name or NotebookLM source title.
  Restate the "raw JSON only, one code block" instruction if it adds prose. For prompt 02, do not
  paste a large JSON coverage plan into NotebookLM chat. Convert the document slice to a text
  source with `make coverage-plan-text`, upload that `.txt` file as a source, and reference its
  file name in the prompt.
- **Claude / ChatGPT Projects** keep the instructions and files across chats in the project, so
  you can run `01`-`04` as separate chats without re-uploading.

## Step 4: Run the prompts and export the replies

1. Run [`01-ontology-inventory.md`](external-service-prompts/01-ontology-inventory.md) once per
   service. Save the JSON as `inventory.json`. This is your coverage plan -- the external
   analogue of the local ontology extraction. With several services, merge their inventories
   into one wider plan: `make curate-drafts CURATE_KIND=inventory`.
   For NotebookLM continuation batches, prefer saving each reply to its own file. If you keep
   a single file, make it a valid JSON array of complete inventory response objects:

   ```json
   [
     {
       "documents": [],
       "cross_document": []
     },
     {
       "documents": [],
       "cross_document": []
     }
   ]
   ```

   Do not paste several JSON objects back to back without commas. Do not use original PDF names
   in `doc`; use the staged ids such as `pdf-3c3a452a8e9c.md`.
2. Run [`02`](external-service-prompts/02-goldset-draft.md),
   [`03`](external-service-prompts/03-chain-questions.md), and
   [`04`](external-service-prompts/04-security-cases.md), feeding the (merged) inventory in as
   the coverage plan. For NotebookLM prompt 02, upload a per-document coverage text source instead
   of pasting the JSON slice:

   ```bash
   make coverage-plan-text \
     COVERAGE_JSON="$DATA_DIR/quickstart-pdf-corpus-md/coverage-pdf-6d8c2128b330.md.json" \
     COVERAGE_TEXT="$DATA_DIR/quickstart-pdf-corpus-md/coverage-6d8c2128b330.txt"
   ```

   Upload `coverage-6d8c2128b330.txt` as a NotebookLM source and write
   `COVERAGE PLAN: coverage-6d8c2128b330.txt` in the prompt. Ask for **batches of 10-20 items**
   on services that can return larger JSON replies; for NotebookLM, request at most 15 items per
   reply. Say "continue" until the plan is covered; large single replies truncate and produce
   invalid JSON.
3. Export each reply to its own file (the raw reply text with its fenced code block is fine).
   Do NOT hand-merge batches -- `make curate-drafts` merges, repairs, filters, and deduplicates
   them in step 6.

## Step 5: Write the provenance sidecar

Beside the exported files, write `external_provenance.json` (contract section 6):

```bash
nano "$DATA_DIR/external-drafts/claude-20260703/external_provenance.json"
```

Record service, model, date, operator, the doc ids you uploaded, the manifest sha256, and
`"data_classification": "open"`. **An artifact bundle without this sidecar must not be imported.**

## Step 6: Curate the merged exports

Curation turns the pile of per-service, per-batch exports into ONE importable file per artifact
kind -- and it is where multi-service drafting pays off:

```bash
make curate-drafts CURATE_KIND=squad \
  CURATE_INPUTS="$DATA_DIR/external-drafts/claude-<date>/goldset.json \
                 $DATA_DIR/external-drafts/gemini-<date>/goldset.json" \
  CURATE_OUT="$DATA_DIR/external-drafts/merged-<date>/goldset.json" \
  CURATE_CORPUS=<staged-corpus-dir>
```

What it does (kinds: `squad`, `grounded`, `security`, `chains`, `inventory`; command:
`llb curate-drafts`):

- **merges** raw exports -- whole JSON files, replies with fenced code blocks, or JSONL -- from
  any number of services and batches;
- **accepts inventory batch arrays** for `CURATE_KIND=inventory`: one file may contain a top-level
  array of complete prompt-01 response objects, useful for NotebookLM "continue" sessions;
- **repairs** near-verbatim quotes: an answer/context/grounding quote that differs from the
  corpus only by whitespace or case is re-snapped to the exact corpus text (and a wrong `title`
  is corrected to the document where the context was actually found);
- **filters invalid rows** with per-reason counts: answers that are not substrings of their
  context, contexts not found in the corpus, schema-invalid security cases, structurally broken
  chains;
- **filters flabby questions**: answer leaks (the question contains its answer), vague stubs,
  "according to this document" phrasing, whole-paragraph answer spans;
- **deduplicates** exact and near-duplicate questions across services (pinned-E5 cosine,
  threshold 0.9 by default; bias pairs and cross-language security groups are protected as
  intentional twins; `CURATE_DEDUP_AGAINST=<bundle>` also drops re-drafts of questions an
  earlier accepted bundle already covers);
- **writes a `*.curation_report.json`** beside the output with per-source and per-reason counts
  -- your first quality signal per service (a high invalid count from one service means its
  session drifted from the verbatim rule; tighten and redraft).

Then proceed with the merged file.

## Step 7: Import and validate each artifact

### 7a. Goldset (SQuAD JSON, Artifact A)

For a directory of prompt-02 SQuAD exports, use the all-in-one Make target. It discovers JSON,
JSONL, text, and markdown reply exports in the input directory, curates and deduplicates them,
imports the canonical goldset/corpus, validates the result, and builds the RAG index:

```bash
make external-squad-rag \
  SQUAD_DRAFT_INPUT_DIR=<directory-with-prompt-02-exports> \
  SQUAD_DRAFT_CORPUS=<staged-corpus-dir> \
  SQUAD_DRAFT_OUT_DIR=<output-work-dir>
```

The target sources the project `.env` before curation, so `HF_TOKEN` is available for semantic
deduplication and embedding downloads.

Use `SQUAD_DRAFT_INPUTS="<file> [<file> ...]"` instead of `SQUAD_DRAFT_INPUT_DIR` when the exports
are not all in one directory. The target writes the curated export and report inside the output
work dir, then writes the RAG-ready artifacts under:

```text
<output-work-dir>/llb/goldset/squad_uk.jsonl
<output-work-dir>/llb/corpus
<output-work-dir>/llb/rag
```

The explicit step-by-step form is still useful when you need to inspect each gate separately:

```bash
make curate-drafts CURATE_KIND=squad \
  CURATE_INPUTS="<export-a> <export-b>" \
  CURATE_OUT=<curated-squad-json> \
  CURATE_CORPUS=<staged-corpus-dir>
make ingest-squad SQUAD_JSON=<curated-squad-json>
# then structurally validate the canonical output against its corpus:
make validate-goldset GOLDSET=<canonical.jsonl> CORPUS=<corpus-dir>
make build-index CORPUS=<corpus-dir>
```

Import re-grounds each answer by substring search and **skips any answer that is not a verbatim
substring** of its context -- your first quality signal. A high skip count means the model
paraphrased; tighten the verbatim rule and redraft.

### 7b. Goldset (grounded JSONL, Artifact B) -- full-document needle realism

When you want the needle scored against the FULL original document (not a context-sized SQuAD doc),
draft Artifact B (one grounded row per line: `quote` + `source_doc_id`) and import it:

```bash
# merge multi-service exports first (optional):
make curate-drafts CURATE_KIND=grounded CURATE_INPUTS="<claude-export> <gemini-export>" \
  CURATE_CORPUS=<corpus-dir> CURATE_OUT=<merged>.jsonl
# then import into a canonical draft bundle (the open-data sidecar is enforced first):
make import-external-draft ARTIFACT=<merged>.jsonl CORPUS=<corpus-dir> \
  SIDECAR="$DATA_DIR/external-drafts/claude-20260703/external_provenance.json"
```

The importer re-grounds each `quote` against `<corpus-dir>/<source_doc_id>`, **drops and counts any
non-verbatim row**, computes exact `source_spans`, stamps `provenance: frontier-drafted` /
`verified: false`, records the service/model/classification in `provenance.json`, and writes
`question_type`/`difficulty` to `item_provenance.jsonl`. A **missing or non-open sidecar aborts
before any bundle is written**. Then run the usual `make validate-goldset` ->
`make cross-check-goldset` -> `verify-*` chain on the emitted bundle.

### 7c. Security cases (Artifact C)

```bash
python -m json.tool "$DATA_DIR/external-drafts/claude-20260703/security_cases.json" >/dev/null
make bench-security \
  SECURITY_CASES="$DATA_DIR/external-drafts/claude-20260703/security_cases.json" \
  SECURITY_MODEL=<local-model> SECURITY_BACKEND=ollama
```

The loader rejects unknown families and malformed detector kinds at load. Benign controls feed
the over-refusal metric; matched bias pairs feed decision-consistency; denial-guard asks feed
attack-success rate. Run only against **local** models -- this is a local-inference safety probe.

### 7d. Chains (Artifact D) -- review-only

Keep chain drafts as review material until the `chain-goldset-generation` forward task lands;
do not import them into scoring paths.

## Step 8: Human verification gate

Required before any headline use. External drafting does not shortcut review. Route imported
goldset drafts through cross-check and the human verification gate exactly as any draft bundle
(see [goldset-from-scratch](goldset-from-scratch.md) and
[verification tooling](../human-tooling/verification-tooling.md)):

```bash
make cross-check-goldset BUNDLE=<bundle> CROSS_CHECK_MODEL=<second-model>
make verify-sample  BUNDLE=<bundle> VERIFY_N=30
make verify-review  VERIFY_WS=<bundle>/verify_sample.csv
make verify-accept  BUNDLE=<bundle> VERIFY_WS=<bundle>/verify_sample.csv
```

Only accepted `verified: true` items score models. Security cases go through the same `verify-*`
review before any headline metric.

## Step 9: Produce local run results

With verified data in hand, the scored run is entirely local:

```bash
make build-index GOLDSET=<verified.jsonl> CORPUS=<corpus>
make run-eval    MODEL=<local-tag> BACKEND=<ollama|vllm|llama.cpp> GOLDSET=<verified.jsonl>
make bench-security SECURITY_CASES=<verified-cases.json> SECURITY_MODEL=<local-tag>
```

Run artifacts land under `$DATA_DIR/run-eval/<timestamp>-<run-id>/`. These local results -- not
anything produced inside the external service -- are the benchmark's output.

## Step 10: Score an external RAG answer log (diagnostic)

Use this only when the system you need to analyze is an external RAG product that reused the same
project documentation corpus and already wrote answers into a JSONL goldset export. Each row should
carry the normal gold fields plus `llm_answer` and, when available, `llm_sources`:

```json
{"id": "...", "question": "...", "reference_answer": "...", "llm_answer": "...",
 "llm_sources": [{"article_id": "...", "article_title": "...", "score": 0.63, "url": "..."}]}
```

Run the interactive scorer:

```bash
make score-external-rag \
  EXTERNAL_RAG_ANSWERS=<answered-jsonl>
```

The command shows one answer card at a time: `question`, `reference_answer`, gold source text,
raw `llm_answer`, the text used for objective scoring, first returned `llm_sources`, and
`llm_error`. The reviewer records the human judgment directly in the JSONL:

```text
a        accept, score=1
p        partial, score=0.5
r        reject, score=0
s <0..1> explicit score
o        edit human_notes
w        edit human_corrected_answer
n/b/u/j  navigate
q        save and quit
```

Partial sessions are safe. Re-run the same command to resume at the first row without
`human_score_0_1` plus `human_decision`. To restart the review, clear the JSONL-backed human
fields:

```bash
make score-external-rag \
  EXTERNAL_RAG_ANSWERS=<answered-jsonl> \
  EXTERNAL_RAG_CLEAR=1
```

Final outputs are written only after every row has a human score and decision:

```text
<answered-jsonl-stem>.csv
<answered-jsonl-stem>.report.md
```

The CSV is sorted by `review_priority_rank` and includes `question`, `reference_answer`, raw
`llm_answer`, the answer text used for objective scoring, first three source records, objective
columns (`exact`, `token_f1`, `contains`), and the JSONL-backed human fields:
`human_score_0_1`, `human_decision`, `human_notes`, `human_corrected_answer`, and `human_status`.

The Markdown report gives aggregate objective estimates, human decision counts, the human mean
score, split estimates, common sources, and links to project commands for improvement work:
`build-index`, `validate-retrieval`, `compare-embeddings`, `sweep`, `tune`,
`prompt-system-prepare`, `run-eval`, `analyze-misses`, and `recommend`.

Important limitation: if the external answer log returns only article titles or URLs, the scorer
cannot compute benchmark source-span recall for that external system by itself. Two ways to audit
retrieval:

- teach the external API to return `doc_id`, `char_start`, and `char_end` for each returned
  source, or
- supply a mapping sidecar with `EXTERNAL_RAG_SOURCE_MAP=<map.json|jsonl|csv>` (`--source-map`)
  translating provider `article_id` / `url` / `article_title` keys into corpus `doc_id` plus an
  optional char range. The CSV then gains `source_hit`, `source_first_hit_rank`,
  `source_hit_weak`, and mapped/unmapped counts, and the report a "Source-span audit" section
  with span-proof recall@3 and MRR. Mappings without a char range only support weak doc-level
  matches (flagged, never counted as span proof), and unmapped returned sources are reported as
  an audit gap, separate from retrieval misses. Example sidecar record:

```json
{"article_id": "a17", "url": "https://kb/articles/a17", "doc_id": "manual/chapter-3.md",
 "char_start": 1200, "char_end": 1420}
```

## End-to-end example: benchmark a mixed PDF corpus with external drafting

The full path for a typical open corpus of the profile in step 2 (a few regulations, one or
two large system manuals, a tail of how-to notes). It is the same corpus-to-recommendation spine
as `make quickstart-corpus`, with the local drafting stage swapped for the external lane:

```bash
# 1. Stage: PDFs (and any .md/.txt) -> canonical corpus + manifests
make ingest-corpus CORPUS_ROOT=<pdf-dir> CORPUS_OUT_DIR=<staged-dir>

# 2. Size the run from the stats (step 2), then draft in 1+ external services
#    (prompts 00-04 per service; export replies under $DATA_DIR/external-drafts/<service>-<date>/)

# 3. Curate: merge services/batches, repair quotes, filter, dedup
make curate-drafts CURATE_KIND=squad CURATE_INPUTS="<claude-export> <gemini-export>" \
  CURATE_OUT="$DATA_DIR/external-drafts/merged-<date>/goldset.json" CURATE_CORPUS=<staged-dir>
make curate-drafts CURATE_KIND=security CURATE_INPUTS="<exports...>" \
  CURATE_OUT="$DATA_DIR/external-drafts/merged-<date>/security_cases.json" CURATE_CORPUS=<staged-dir>

# 4. Import + structural validation
make ingest-squad SQUAD_JSON="$DATA_DIR/external-drafts/merged-<date>/goldset.json"
make validate-goldset GOLDSET=<canonical.jsonl> CORPUS=<canonical-corpus>

# 5. Mechanical cross-check, then the human verification gate (required)
make cross-check-goldset BUNDLE=<bundle> CROSS_CHECK_MODEL=<second-model>
make verify-sample  BUNDLE=<bundle> VERIFY_N=40
make verify-review  VERIFY_WS=<bundle>/verify_sample.csv
make verify-accept  BUNDLE=<bundle> VERIFY_WS=<bundle>/verify_sample.csv

# 6. RAG index over the corpus + retrieval sanity gate (recall@10 >= 0.8)
make build-index GOLDSET=<verified.jsonl> CORPUS=<corpus>
make validate-retrieval

# 7. Model benchmark: single run, or the sweep grid (models x top_k) + recommendation
make run-eval MODEL=<local-tag> BACKEND=<ollama|vllm|llamacpp> GOLDSET=<verified.jsonl>
make sweep SWEEP_ID=<corpus-name>          # one isolated cell per (model, top_k)
make recommend                             # ranks models at their best top_k for THIS corpus

# 8. Security tier over the verified corpus-specific cases
make bench-security SECURITY_CASES=<verified-cases.json> SECURITY_MODEL=<local-tag>
```

Yield expectations for planning (per 1M staged characters, using the step 2 sizing): request
roughly 200-600 items per service; after curation (cross-service dedup + invalid/flabby drops)
expect the merged kept set to be materially smaller than the sum of requests, and plan the human
review sample (`VERIFY_N`) against the merged count, not the requested one. Retrieval quality is
gated before any model ranking (step 6), so a corpus whose drafts cluster on one manual shows up
as a recall problem before it can distort model scores.

## What you may and may not send to a service

| Allowed to upload (open data only) | Never upload |
| --- | --- |
| Public/cleared corpus text for drafting | Restricted or private corpus documents |
| A topic or entity list to steer coverage | Model outputs to be judged or scored |
| Draft QA for a second-opinion rewrite | Any material for retrieval, answering, or ranking |

Scoring, judging, and ranking are local-only by design (see
[product decisions](../../impl/current/scope-boundaries.md)). The external service is a drafting
assistant at the front of the pipeline and nothing more.

## See also

- [External-service draft contract](../../design/external-draft-contract.md) -- exact artifact
  shapes.
- [Prompt pack](external-service-prompts/README.md) -- the copy-paste prompts.
- [Create a gold set (end-to-end)](goldset-from-scratch.md) -- the local drafting spine.
- [Data prep](../../impl/current/data-prep.md) -- the local ontology pipeline and PDF conversion.
- [Product decisions: data egress](../../impl/current/scope-boundaries.md#data-egress) -- the
  policy.
