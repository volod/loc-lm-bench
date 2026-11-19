# Scanned-Book Recovery Runs (`book-ocr`)

`book-ocr` drives a scanned PDF -- a book of page photographs with no text layer -- through a
registry of recovery stages one page at a time, records every stage's outcome per page, and resumes
after an interruption without repeating a completed stage. Today the registry holds the `render`
stage (each page at its scan's native resolution) and two deterministic stand-in stages; the
preprocessing, recognition, structure, and rendering stages of
[scanned book recovery](../../design/spec.md#scanned-book-recovery) plug into the same run as they
land. For a born-digital PDF, use the [PDF corpus lane](pdf-corpus-prep.md) instead.

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
| `fake-preprocess` | `render` | `variant.json` naming the original image | `fail_pages`, `note` |
| `fake-ocr` | `render` | `text.txt`, a placeholder naming the page and image digest | `fail_pages`, `note` |

The two `fake-` stages recognise nothing: they exist so the ledger, resume, reuse, and failure
handling can be driven over a whole real book before any engine is wired. `fail_pages` forces a
failure on the named PDF pages.

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
  pages/<pNNNN>/<stage>/         one committed stage: its outputs and stage.json (llb.book-ocr-stage-record)
  metadata/page_manifest.json    per-page, per-stage ledger (llb.book-ocr-page-manifest)
  metadata/processing_report.md  human-readable summary and failed pages
  logs/run.jsonl                 one JSON event per line, appended by every invocation
```

`pNNNN` is the 1-based PDF page, zero-padded to four digits (wider for a longer document). A full
render of a 1,067-page, 150 dpi greyscale scan is about 0.9 GB.
