# Scanned-Book Recovery Runs (`book-ocr`)

`book-ocr` drives a scanned PDF -- a book of page photographs with no text layer -- through a
registry of recovery stages one page at a time, records every stage's outcome per page, and resumes
after an interruption without repeating a completed stage. The registry holds `render` (each page
at its scan's native resolution), `signals` and `variants` (the measurements and the corrections
they call for), `classes` (the page classes a pilot and an engine route on), the recognition
stages `layout`, `ocr`, `formula`, and `table`, the `blocks` stage that merges them, the
`resolve` stage that chooses one text per block, and two deterministic stand-in stages.
`book-ocr structure` then infers the printed pages and the hierarchy from those ledgers.
`book-ocr render` writes the reading vault from that structure. For a
born-digital PDF, use the
[PDF corpus lane](pdf-corpus-prep.md) instead.

What it does and why it is built this way:
[pipeline and resume](../../impl/current/book-recovery/pipeline-and-resume.md).

## Check before running

```sh
make book-ocr BOOK_PDF=<scan.pdf> BOOK_PAGES=1-40 BOOK_DRY_RUN=1
```

The dry run opens the source, prints its SHA-256, page count, and embedded-text diagnostics, parses
the page ranges, and checks every configured stage -- registered, configured after the stages it
reads, valid parameters, model available -- reporting every problem at once. It writes nothing and
exits 2 when anything is refused.

## Run

```sh
make book-ocr BOOK_PDF=<scan.pdf>                                  # every page, render only
make book-ocr BOOK_PDF=<scan.pdf> BOOK_PAGES=1-20,37,900- BOOK_BATCH_PAGES=25
make book-ocr BOOK_PDF=<scan.pdf> BOOK_CONFIG=<book-ocr.yaml>
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `BOOK_PDF` | The scanned source PDF | required for a new run |
| `BOOK_PAGES` | 1-based inclusive PDF pages: `N`, `N-M`, `N-` (to the end), `-M`, comma-separated | `1-` |
| `BOOK_BATCH_PAGES` | Pages per batch; each stage runs over a whole batch before the next stage | `25` |
| `BOOK_MAX_ATTEMPTS` | Attempts per page stage before it is recorded as failed | `2` |
| `BOOK_STAGES` | Comma-separated stage names, in order (replaces the config file's list) | `render` |
| `BOOK_CONFIG` | YAML or JSON with `pages`, `batch_pages`, `max_attempts`, `source_policy`, `stages` | none |
| `BOOK_SOURCE_POLICY` | `symlink`, `copy` (read-only copy), or `reference` (absolute path only) | `symlink` |
| `BOOK_REUSE_FROM` | An earlier run whose stages are taken wherever the cache key matches | none |
| `BOOK_OUT_DIR` | Parent of the run directory | `$DATA_DIR/book-ocr` |
| `BOOK_DRY_RUN=1` | Validate only | off |

Page ranges are refused by name: a page before 1, a page past the last page, a reversed range, and
two parts that select the same page (`'1-5' and '4-6' overlap on pages 4-5`). The run directory may
not sit inside the directory that holds the source.

A config file gives each stage its parameters and model:

```yaml
pages: "1-120"
batch_pages: 20
stages:
  - name: render
    params: {dpi: native, colorspace: auto}
  - name: fake-ocr
    model: ollama:<vision-model-tag>
    params: {fail_pages: [7]}
```

A model is written `<backend>:<name>`: `ollama:<tag>` is checked against the running Ollama
daemon's served tags, `fake:<name>` is always present. `--stage-model <stage>=<model>` on the raw
CLI sets one without a file.

### Registered stages

| Stage | Reads | Writes | Parameters |
| --- | --- | --- | --- |
| `render` | the source page | `page.png` | `dpi`: `native` (the scan's own resolution) or an integer; `colorspace`: `auto` (grey when every image on the page is single-channel), `grey`, `rgb` |
| `signals` | `render` | `signals.json`, also published at `pages/<pNNNN>/signals.json` | none. The measured thresholds are part of the cache key, and so is the PDF page, because the file records the page id |
| `classes` | `render`, `signals` | `classes.json`, also published at `pages/<pNNNN>/classes.json` | none. The class cuts and the page's placement rotation are part of the cache key |
| `variants` | `render`, `signals` | `manifest.json` and `<name>/page.png` for each kept correction. This stage's directory is `pages/<pNNNN>/variants/` | `dewarp`: `opencv` (default), `page-dewarp`, `uvdoc`, or `auto` |
| `fake-preprocess` | `render` | `variant.json` naming the original image | `fail_pages`, `note` |
| `fake-ocr` | `render` | `text.txt`, a placeholder naming the page and image digest | `fail_pages`, `note` |
| `layout`, `ocr`, `formula`, `table` | `render`, `variants` when a non-original `variant` is named, and `classes` when `by_page_class` is set and `classes` is configured earlier | `reading.json`, that engine's blocks mapped back onto the render before the merge | `engine` (required), `variant` (`original` by default), `page_class`, `by_page_class` |
| `blocks` | `render` and every recognition stage configured before it | `blocks.json`, then the same file published at `pages/<pNNNN>/blocks.json` | none. The cache key also includes the PDF page, because the ledger records the page id |
| `resolve` | `blocks`, `correct` when that stage is configured before it, and `classes` when `by_page_class` is set | `overlay.json` and `resolved.json`, published at `pages/<pNNNN>/` | `primary` (required: `stage`, `engine`, `variant`, optional `model`); `by_page_class`; `page_class` |

The two `fake-` stages recognise nothing. They exist so the run ledger, resume, reuse, and failure
handling can be driven without an engine. `fail_pages` forces a failure on the named PDF pages.

A recognition stage's `engine` is an id from `make book-ocr-engines`. `by_page_class` maps a page
class to a different id. When `classes` ran earlier, the class is that page's primary class and
the class file is an input of the engine stage, so a class change misses only the stages that
route on it. Pages in one batch that share an engine run together, so each engine opens at most
once per batch. The cache key records the engine that ran. A `page_class` set with no `classes`
stage still applies to every page. `blocks` refuses a run that configured no recognition stage.
`resolve` refuses a run that names no primary, or a primary stage that is not configured before
it. The rule, the overlay, and the reviewer file are
[uncertainty and review](../../impl/current/book-recovery/uncertainty-and-review.md).
An engine that is not installed refuses the dry run by name (`EngineUnavailable`) and writes
nothing. What a block contains, and how two engines disagree, is
[the block ledger](../../impl/current/book-recovery/block-ledger.md). The class cuts are
[page classes](../../impl/current/book-recovery/tool-selection.md#page-classes).

```sh
make book-ocr-engines    # id, stages, and ready or refused for each registered engine
```

## Interrupt, resume, inspect

```sh
make book-ocr BOOK_RESUME=<run-dir>               # continue under the recorded config
make book-ocr-status BOOK_RUN=<run-dir>           # every page, every stage
make book-ocr-status BOOK_RUN=<run-dir> BOOK_SHOW=open
make book-ocr-status BOOK_RUN=<run-dir> BOOK_JSON=1
```

A resume reads `config.json` and refuses any run-shaping option beside it. It re-hashes the source
-- and `BOOK_PDF`, when given -- and refuses a mismatch. Completed stages whose outputs still hash
to their record are skipped, the interrupted attempt is discarded and re-run, and failed stages are
retried. `status` reads the stage records on disk, so it is accurate even right after a kill.

Exit codes: `0` when the run finished (failed pages are reported, not fatal), `2` when it was
refused before any stage ran, `130` on Ctrl-C.

## Run directory

```text
$DATA_DIR/book-ocr/<run_timestamp>/
  config.json                    what the run is (llb.book-ocr-run-config)
  source/<name>.pdf              the source, under the copy or symlink policy
  pages/<pNNNN>/<stage>/         one committed stage: outputs and stage.json
  pages/<pNNNN>/signals.json     the measured signal vector (llb.book-ocr-page-signals)
  pages/<pNNNN>/classes.json     the classes the page matches (llb.book-ocr-page-classes)
  pages/<pNNNN>/variants/        manifest and kept corrections (llb.book-ocr-variant-manifest)
  pages/<pNNNN>/blocks.json      the page block ledger (llb.book-ocr-block-ledger)
  pages/<pNNNN>/overlay.json     applied and orphaned corrections (llb.book-ocr-correction-overlay)
  pages/<pNNNN>/resolved.json    the resolved text (llb.book-ocr-resolved-page)
  review/decisions.json          reviewer edits (llb.book-ocr-reviewer-decisions)
  review/queue.json              uncertain spans for the workbench (llb.book-ocr-review-queue)
  review/<block>-s<start>-<end>.png   a padded crop of one uncertain block
  pilot/selection.json           the seeded pilot (llb.book-ocr-pilot-selection)
  metadata/page_manifest.json    per-page, per-stage ledger (llb.book-ocr-page-manifest)
  metadata/document_structure.json  printed pages, hierarchy, numbering (llb.book-ocr-document-structure)
  metadata/quality_manifest.json one row per uncertain span (llb.book-ocr-quality-manifest)
  metadata/processing_report.md  run summary: stages and failed pages
  logs/run.jsonl                 one JSON event per line, appended by every invocation
```

`pNNNN` is the 1-based PDF page, zero-padded to four digits (wider for a longer document). A full
render of a 1,067-page, 150 dpi greyscale scan is about 0.9 GB.

## Review queue

`book-ocr review-queue` reads a finished run and writes a marker, a crop, a manifest row, and a
queue entry for every uncertain span, plus one entry for a page that failed terminally. A blank
page is recorded on the manifest and has no queue entry. The marker syntax, the coverage field,
and what a decision writes are
[uncertainty and review](../../impl/current/book-recovery/uncertainty-and-review.md).

```sh
make book-ocr-review-queue BOOK_RUN=<run-dir>
make review-workbench REVIEW_PATH=<run-dir>/review/queue.json
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `BOOK_RUN` | A finished book-ocr run. A terminal page failure is included | required |

Exit code 2 is a path that is not a run, or a page that is still unfinished. The command prints
kind counts and coverage counts. It does not print the recovered text. After a decision,
resume the run so `resolve` applies it, then build the queue again.

## Structure

`book-ocr structure` reads a finished ledger and writes `metadata/document_structure.json`: the
printed page of every PDF page, the hierarchy, the running headers, and the figure, table, and
equation numbers. It is not a stage. An unfinished ledger is exit code 2. A page that failed
terminally is included and contributes no lines. The command prints counts and does not print the
recovered text. What the inference keeps, and what the sample scan showed,
is [structure](../../impl/current/book-recovery/structure-and-rendering.md).

```sh
make book-ocr-structure BOOK_RUN=<run-dir>
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `BOOK_RUN` | A book-ocr run. An unfinished ledger is refused; a terminal failure is included | required |

## Reading vault

`book-ocr render` writes an Obsidian vault from a run that already has a document structure. It
is not a stage. The default directory is `book/<book-slug>/` inside the run. A missing structure
is exit code 2. The command prints the vault path and does not print the recovered text. The
layout, the page links, and what the sample scan showed are in
[the reading vault](../../impl/current/book-recovery/structure-and-rendering.md#reading-vault).

```sh
make book-ocr-render BOOK_RUN=<run-dir>
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `BOOK_RUN` | A book-ocr run that already has a document structure | required |
| `BOOK_OUTPUT` | Vault directory | `book/<slug>/` inside the run |
| `BOOK_SOURCE_POLICY` | `copy`, `symlink`, or `reference` | `symlink` |
| `BOOK_PAGE_FALLBACK` | Set to `1` to write a rendered image of each page | off |
| `BOOK_SECTION_CAP` | Resolved characters in one note. The floor is 200 | `12000` |
| `BOOK_SLUG` | Lowercase ASCII vault name | the PDF file name |
| `BOOK_REFERENCE_GRAMMAR` | YAML grammar that replaces the handbook cues | the handbook grammar |

The command also prints how many citations resolved, stayed ambiguous, or stayed unresolved.
A grammar file replaces the handbook cues entirely. What those cues match, and what the sample
scan showed, is in
[cross-references](../../impl/current/book-recovery/structure-and-rendering.md#cross-references).
`render` then runs the vault check below. A failed check is exit code 1.

## Corpus projection

`book-ocr project` writes a retrieval corpus from a run that already has a document structure.
It is not a stage. One vault content file becomes one document: formulas as LaTeX, tables as
Markdown, and an unresolved span as a machine marker. Navigation, front matter, and page
furniture are left out. Each document gets a page-citation sidecar and a metadata sidecar.
`BOOK_LICENCE` is required and has no default. A copyrighted scan is `local-only`. The command
prints counts and does not print the recovered text. What the files contain, and what the
sample scan showed, is [corpus projection](../../impl/current/book-recovery/corpus-projection.md).

```sh
make book-ocr-project BOOK_RUN=<run-dir> BOOK_CORPUS=<corpus-dir> BOOK_LICENCE=local-only
make ingest-corpus CORPUS_ROOT=<corpus-dir>
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `BOOK_RUN` | A book-ocr run that already has a document structure | required |
| `BOOK_CORPUS` | The corpus directory to write | required |
| `BOOK_LICENCE` | `redistributable` or `local-only` | required |
| `BOOK_LANGUAGE` | Language recorded on every document | detected from the text |
| `BOOK_SECTION_CAP` | Packed source characters in one document. The floor is 200 | `12000` |
| `BOOK_SLUG` | Lowercase ASCII book name used in vault file paths | the PDF file name |

A missing structure, a missing licence, or an output that would replace the run is exit code 2.
An unchanged ledger projects to the same files. A changed text is a new file whose
`revision_of` names the previous one.

## Vault check

`book-ocr validate` checks a rendered vault and writes `metadata/processing_report.md` and
`metadata/validation.json` into that vault. It also runs at the end of `render`, and at the end
of `run` or `resume` when the run already holds a vault. The report lists page status,
warnings, engines, readings, confidence buckets, the open review queue, and the pages still
open for a person. Confidence buckets are diagnostics, not probabilities of correctness. The
command prints failed checks and does not print recovered text. What the check refuses is
[vault validation](../../impl/current/book-recovery/structure-and-rendering.md#vault-validation).

```sh
make book-ocr-validate BOOK_DIR=<book-dir>
make book-ocr-validate BOOK_DIR=<book-dir> BOOK_RUN=<run-dir>
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `BOOK_DIR` | A rendered reading vault | required |
| `BOOK_RUN` | The book-ocr run, when the vault was published outside it | the run recorded in the vault |

Exit code `0` means the vault passed. Exit code `1` means a check failed. Exit code `2` means
the directory is not a vault, or the run cannot be read.

## Pilot selection

`book-ocr pilot-select` reads a finished `classes` stage and writes a reproducible set of pages.
Each page records the class it represents and the signal values that placed it. The default is
25 pages at seed 0. A class with no qualifying page is reported by its prose name.

```sh
make book-ocr BOOK_PDF=<scan.pdf> BOOK_STAGES=render,signals,classes
make book-ocr-pilot-select BOOK_RUN=<run-dir>
make book-ocr-pilot-select BOOK_RUN=<run-dir> BOOK_PILOT_SIZE=25 BOOK_PILOT_SEED=0 \
  BOOK_PILOT_FROZEN=<frozen-pilot.json> BOOK_PILOT_OUT=<path>
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `BOOK_RUN` | A book-ocr run whose `classes` stage has finished for every configured page | required |
| `BOOK_PILOT_SIZE` | How many pages to select. A size below 20 is accepted | `25` |
| `BOOK_PILOT_SEED` | The seed that draws inside each class's strongest pages | `0` |
| `BOOK_PILOT_FROZEN` | Require the replay to match the source checksum, sampler settings, coverage and page records in this selection | unset |
| `BOOK_PILOT_OUT` | Also write the selection here. The run always keeps `pilot/selection.json` | unset |

The selector rehashes the run's source and checks that every class record names its configured PDF
page. Exit code 2 means the run could not be read, the source changed, class records were missing
or misplaced, or the replay differed from the frozen selection. These checks run before the
selection files are written. The cuts, the precedence, and the handbook list are
[page classes](../../impl/current/book-recovery/tool-selection.md#page-classes).

## Pilot reading

`book-ocr pilot-run` reads a frozen pilot with the existing Docling-plus-Tesseract lane, then with
each roster engine that fits this host, one engine at a time. It reuses a classified run's
`render`, `signals`, and `classes`. It does not score pages and it does not choose a winner.
Scoring is `book-ocr compare`, which still needs an accepted reference.

```sh
make book-ocr-pilot-run BOOK_PDF=<scan.pdf> BOOK_PILOT=<frozen-pilot.json> \
  BOOK_REUSE_FROM=<classified-run> BOOK_LANGUAGE=eng
make book-ocr-pilot-run BOOK_DRY_RUN=1 BOOK_PDF=<scan.pdf> BOOK_PILOT=<frozen-pilot.json> \
  BOOK_REUSE_FROM=<classified-run>
make book-ocr-pilot-run BOOK_RESUME=<campaign-dir>
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `BOOK_PDF` | The scan the frozen pilot was drawn from | required for a new run |
| `BOOK_PILOT` | Frozen pilot selection | required for a new run |
| `BOOK_REUSE_FROM` | Classified run whose render, signals, and classes are reused | unset |
| `BOOK_LANGUAGE` | OCR language recorded on the existing lane and on each recognition stage | `eng` |
| `BOOK_OUT_DIR` | Parent of the campaign directory and of each lane's run | `$DATA_DIR/book-ocr` |
| `BOOK_DRY_RUN` | Set to `1` to print the plan and write nothing | unset |
| `BOOK_RESUME` | Campaign directory to continue | unset |
| `BOOK_REFERENCE`, `BOOK_PREDRAFT`, `BOOK_ACCEPTS`, `BOOK_MANIFEST` | When all four are set, also write the compare manifest | unset |

The campaign directory holds `campaign.json`. Each fitting engine is a normal book-ocr run under
the same output root. An engine whose survey peak is above the host is listed and not started.
A lane that finishes no recognition call is `unavailable`. A lane that reads some pages and fails
others stays measured, and the failed PDF pages are named in the campaign index. Exit code 2 means
the PDF is missing, the pilot was drawn from another file, or a pending lane's dry run refused.
The decision rule is
[pilot comparison](../../impl/current/book-recovery/tool-selection.md#pilot-comparison).

## Review the frozen pilot

Use two different shortlisted recognition runs on the frozen pilot, such as two complete lanes
from the [pilot campaign](#pilot-reading). Pass the run with layout and structured blocks as
`BOOK_PRIMARY` and a second reading as `BOOK_ALTERNATE`. Each run needs a completed render and
block ledger for every pilot page. Set `BOOK_ALTERNATE_OVERRIDE` to a subset read by the second
engine on a better variant, such as pages that needed a half-turn.

```sh
make book-ocr-pilot-prepare BOOK_PDF=<scan.pdf> BOOK_PILOT=<frozen-pilot.json> \
  BOOK_PRIMARY=<layout-run> BOOK_ALTERNATE=<second-run>
make book-ocr-pilot-review BOOK_REVIEW=<layout-run>/pilot/reference-review
```

Open the printed loopback URL. The worksheet shows the source pixels next to the editable
transcription, with disagreements and unpaired readings indicated on the block list. Use the page
selector, Previous, Next, and Next pending to move through the frozen set. Click a region or select
a block to compare it. Use Zoom, Rotate, and Draw region to inspect or correct geometry. Region
coordinates stay in the original scan frame when you rotate the view.

Review the entire page, including material both readings may have missed. Correct the printed
folio, block types, text and reading order, formulas and equation numbers, table cell spans, and
figure boxes and links to caption blocks. For a table, enter a JSON list of rows. A plain string is
a one-cell entry and `{"text": "header", "colspan": 2}` spans two columns. Add or remove blocks
where the scan requires it. Enter your name, confirm the full-page check, and choose **Accept page**.

**Save draft** saves a revision. Changing pages also saves pending edits, including unfinished
table JSON. Reopen the same bundle to resume, or load the pre-draft or an earlier saved revision
as an editable draft. An edit to an accepted page needs a new acceptance. Copy the entire review
bundle when moving computers. Its `scans/`, `predraft/`, and `history/` directories preserve
the scans, starting draft, and progress alongside `review.json`.

```sh
make book-ocr-pilot-review-status BOOK_REVIEW=<review-bundle>
make book-ocr-pilot-finalize BOOK_REVIEW=<review-bundle>
make book-ocr-score BOOK_RUN=<recognition-run> BOOK_REFERENCE=<finalized-reference> \
  BOOK_ACCEPTANCE=<finalized-reference>/acceptance.json
```

Finalization checks every page has a current acceptance and validates scan checksums, the
pre-draft digest, reference structure, and LaTeX. The printed output path is an immutable
snapshot with `pages/` for scoring and an `acceptance.json` record with page decisions and
digests. Keep the review bundle and reference under `DATA_DIR`. Place only the acceptance
record beside the frozen `pilot_pages.json` for the repository handoff. Pass that copy as
`BOOK_ACCEPTANCE` when scoring to check that the file bytes and source match the recorded
reference. When generating the comparison manifest with `make book-ocr-pilot-run`, set
`BOOK_ACCEPTS` to an `accepts.json` with the `source_sha256` and one row per page containing
`pdf_page`, `reviewer`, and `accepted: true`.

## Score

`book-ocr score` compares a finished ledger, the resolved pipeline when the run has one, or the
existing PDF lane's citation sidecar, with a reference transcription. It reports character and
word error, missing and duplicated words, reading order, one count per dangerous-substitution
class, formula fidelity, and the structural metrics. The substitution counts are not folded into
a single score. Figure boxes, table grids, and caption links a lane cannot express are reported
absent. The definitions, the reference format, and the hand-computed pins are
[quality metrics](../../impl/current/book-recovery/quality-metrics.md).

```sh
make book-ocr-score BOOK_REFERENCE=<reference-dir> BOOK_RUN=<run-dir>
make book-ocr-score BOOK_REFERENCE=<reference-dir> BOOK_CITATIONS=<doc>.citations.json
```

`BOOK_ENGINE` scores one ledger engine and still includes the resolved lane. `BOOK_SCORE_OUT`
writes the JSON report and `BOOK_JSON=1` prints it. Without `BOOK_JSON`, the command prints a
short summary that names missing submissions, non-zero abstentions, and fields a page did not
emit. Exit code 2 is a missing input or a reference the loader refused. The command does not
recover a scan and does not write a reference.

## Compare

`book-ocr compare` reads a manifest of accepted pilot pages and writes the comparison report and
the book-ocr config the decision selected. It does not load a model and does not read the scan.
Every reference page must be accepted, and the existing lane's citation sidecar must record the
same OCR language the manifest names. Exit code 2 means the comparison cannot start. Exit code 0
with `negative: true` means no candidate beat the existing lane; that is a result, and the config
names `docling`. The decision rule, the selection-set label, and the VRAM refusal are
[pilot comparison](../../impl/current/book-recovery/tool-selection.md#pilot-comparison).

```sh
make book-ocr-compare BOOK_MANIFEST=<manifest.yaml> \
  BOOK_COMPARE_OUT=<report.json> BOOK_COMPARE_CONFIG=<book-ocr.yaml>
```

## Recovery-tool survey

`book-ocr survey` installs each candidate recovery tool into its own environment and runs it on one
page, so the pilot compares only tools that demonstrably run on this host. The candidates, the
shortlist per stage, and the smoke results are in
[tool selection](../../impl/current/book-recovery/tool-selection.md).

```sh
make book-survey-list                                   # every candidate: stages, route, licence
make book-survey-list BOOK_SURVEY_STAGE=formula         # only one stage's candidates
make book-survey-install                                # every isolated environment
make book-survey-install BOOK_SURVEY_ENVS=light,mineru  # just these
make book-survey-smoke                                  # every candidate on the synthetic page
make book-survey-smoke BOOK_SURVEY_CANDIDATES=tesseract,paddleocr-vl \
  BOOK_SURVEY_PAGES="synthetic <scan.pdf>:<page>"
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `BOOK_SURVEY_STAGE` | `preprocess`, `layout`, `ocr`, `formula`, `table`, `convert`, or `correct` | all |
| `BOOK_SURVEY_ENVS` | Comma-separated survey environments (`light`, `mineru`, `docling`, `marker`, `paddle`, `paddle-fw`) | `all` |
| `BOOK_SURVEY_CANDIDATES` | Comma-separated candidate ids from `make book-survey-list` | `all` |
| `BOOK_SURVEY_PAGES` | Space-separated pages: `synthetic` and/or `<pdf>:<page>` (a 1-based PDF page of a real scan) | `synthetic` |

**Install.** Each environment is a uv virtualenv under `$DATA_DIR/venvs/book-<env>/`, resolved
from exact top-level pins as of the survey cutoff (`--exclude-newer`), with its full resolution
frozen to `llb-freeze.txt` beside it. The shipped `.venv` is never touched. uv's download cache
defaults to `$DATA_DIR/cache/uv` (an explicit `UV_CACHE_DIR` wins), so the several CUDA torch
stacks land on the data volume rather than the home directory. Ollama candidates need their tag
pulled (`ollama pull <tag>`); Tesseract comes from `make apt-deps`. The vLLM-served candidates
(Marker `balanced`, dots.ocr, DeepSeek-OCR-2, olmOCR-2) need nothing installed beyond Docker with
the NVIDIA Container Toolkit, which `make venv` requires on a CUDA host: the smoke pulls
`vllm/vllm-openai:v0.25.1` (27.8 GB unpacked) on first use, starts its own container for the
candidate as the invoking user, and stops and removes it afterwards, keeping its log as
`server.log` in the candidate's directory. It refuses to pull onto a Docker data root with less
than 30 GB free.

**Smoke.** Candidates run one at a time; before each, the device's used VRAM is read as its
baseline, and afterwards the survey waits for VRAM to return to it. Every model a tool downloads
goes under `$DATA_DIR/cache/book-survey/` (Hugging Face, ModelScope, PaddleX, MinerU, and the XDG
cache), and usage telemetry is switched off. A candidate that is not installed, fails, times out,
runs out of GPU memory, or silently falls back to a weaker mode is recorded with that status, never
dropped. Each candidate's own output, its log, and its text (`reading.txt`) stay under its directory.

```sh
make book-survey-rescore BOOK_SURVEY_DIR=<run-dir>/survey   # recompute scores, no re-run
```

The survey writes `$DATA_DIR/book-ocr/<run_timestamp>/survey/` with `host.json` (device, VRAM
budget, cutoff), `pages/` (the smoke pages; a real page is rendered at its native resolution and
never leaves the run directory), one directory per page and candidate, and `survey.json` /
`survey.md`. These summaries are diagnostic outputs that nothing reads back.
