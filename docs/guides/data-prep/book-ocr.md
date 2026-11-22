# Scanned-Book Recovery Runs (`book-ocr`)

`book-ocr` drives a scanned PDF -- a book of page photographs with no text layer -- through a
registry of recovery stages one page at a time, records every stage's outcome per page, and resumes
after an interruption without repeating a completed stage. The registry holds `render` (each page
at its scan's native resolution), `signals` and `variants` (the measurements and the corrections
they call for), `classes` (the page classes a pilot and an engine route on), the recognition
stages `layout`, `ocr`, `formula`, and `table`, the `blocks` stage that merges them, the
`resolve` stage that chooses one text per block, and two deterministic stand-in stages.
Structure and rendering plug into the same run as they land. For a
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
  pilot/selection.json           the seeded pilot (llb.book-ocr-pilot-selection)
  metadata/page_manifest.json    per-page, per-stage ledger (llb.book-ocr-page-manifest)
  metadata/processing_report.md  human-readable summary and failed pages
  logs/run.jsonl                 one JSON event per line, appended by every invocation
```

`pNNNN` is the 1-based PDF page, zero-padded to four digits (wider for a longer document). A full
render of a 1,067-page, 150 dpi greyscale scan is about 0.9 GB.

## Pilot selection

`book-ocr pilot-select` reads a finished `classes` stage and writes a reproducible set of pages.
Each page records the class it represents and the signal values that placed it. The default is
25 pages at seed 0. A class with no qualifying page is reported by its prose name.

```sh
make book-ocr BOOK_PDF=<scan.pdf> BOOK_STAGES=render,signals,classes
make book-ocr-pilot-select BOOK_RUN=<run-dir>
make book-ocr-pilot-select BOOK_RUN=<run-dir> BOOK_PILOT_SIZE=25 BOOK_PILOT_SEED=0 \
  BOOK_PILOT_OUT=<path>
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `BOOK_RUN` | A book-ocr run whose `classes` stage has finished for every configured page | required |
| `BOOK_PILOT_SIZE` | How many pages to select. A size below 20 is accepted | `25` |
| `BOOK_PILOT_SEED` | The seed that draws inside each class's strongest pages | `0` |
| `BOOK_PILOT_OUT` | Also write the selection here. The run always keeps `pilot/selection.json` | unset |

Exit code 2 is a run that is not a book-ocr run, or a classes stage that has not finished. The
cuts, the precedence, and the handbook list are
[page classes](../../impl/current/book-recovery/tool-selection.md#page-classes).

## Score

`book-ocr score` compares a finished ledger, or the existing PDF lane's citation sidecar, with a
reference transcription. It reports character and word error, missing and duplicated words,
reading order, one count per dangerous-substitution class, formula fidelity, and the structural
metrics. The substitution counts are not folded into a single score. Figure boxes, table grids,
and caption links the existing lane cannot express are reported absent. The definitions, the
reference format, and the hand-computed pins are
[quality metrics](../../impl/current/book-recovery/quality-metrics.md).

```sh
make book-ocr-score BOOK_REFERENCE=<reference-dir> BOOK_RUN=<run-dir>
make book-ocr-score BOOK_REFERENCE=<reference-dir> BOOK_CITATIONS=<doc>.citations.json
```

`BOOK_ENGINE` scores one ledger engine. `BOOK_SCORE_OUT` writes the JSON report and `BOOK_JSON=1`
prints it. Without `BOOK_JSON`, the command prints a short summary. Exit code 2 is a missing
input or a reference the loader refused. The command does not recover a scan and does not write a
reference.

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
