# Book Recovery: Run Ledger and Resume

The skeleton every later recovery stage plugs into: one `book-ocr` run over a scanned PDF, with the
source identified by checksum, a per-page inventory, a per-page, per-stage status ledger, a
content-addressed stage cache, and a processing report. Render, signals, variants, page classes,
the recognition stages, and resolution ship on that seam, plus two deterministic stand-ins, so
resume, reuse, and failure isolation run over a real book. Operator usage:
[book-ocr guide](../../../guides/data-prep/book-ocr.md).

## Where it lives

| Module | Owns |
| --- | --- |
| `src/llb/prep/book/runner.py` | `start_run`, `resume_run`, `execute`: batch-major, then stage-major, walk; manifest and report after every batch and on the way out of an interrupt |
| `src/llb/prep/book/stage_exec.py` | One stage on one page: skip, reuse, or execute with bounded retry; the failed and blocked outcomes |
| `src/llb/prep/book/cache.py` | Cache keys, committed stage directories, the atomic `.partial` -> `<stage>` commit, debris removal, `ReuseIndex` |
| `src/llb/prep/book/config.py` | `RunRequest`, `preflight` (the whole dry run), `config.json` read and write |
| `src/llb/prep/book/ranges.py` | Page-range parsing, batching, page ids |
| `src/llb/prep/book/source.py` | Source fingerprint (the PDF lane's `_sha256_file`), the copy, symlink, and reference policies, the overlap refusal, resume verification |
| `src/llb/prep/book/inventory.py` | Document diagnostics (the PDF lane's `inspect_pdf`), per-page inventory, native-resolution rendering |
| `src/llb/prep/book/ledger/` | The page manifest projected from the stage records (`manifest.py`); block assembly is the [block ledger](block-ledger.md) |
| `src/llb/prep/book/report.py`, `logs.py` | The Markdown report and the JSON-lines event log |
| `src/llb/prep/book/models.py` | Stage-model availability: `ollama:<tag>` against the daemon's served tags, `fake:<name>` |
| `src/llb/prep/book/stages/` | The `Stage` protocol and registry, `render`, `signals`, `variants`, `classes`, `fake-preprocess`, `fake-ocr`, the recognition stages, `correct`, and `resolve` |
| `src/llb/cli/book/` | The `llb book-ocr` Typer group: `run`, `status`, `score`, `pilot-select`, `engines`, `review-queue`, `structure`, `render`, and `validate` |
| `src/llb/core/contracts/data_prep/book.py`, `book_blocks.py`, `book_preprocess.py`, `book_classes.py`, `book_resolve.py`, `book_review.py`, `book_structure.py`, `src/llb/artifacts/data_prep/book.py` | The fifteen registered families |
| `make/data-prep/book.mk` | `make book-ocr`, `make book-ocr-status`, `make book-ocr-score`, `make book-ocr-pilot-select`, `make book-ocr-engines`, `make book-ocr-review-queue`, `make book-ocr-structure`, `make book-ocr-render`, `make book-ocr-validate` |

Tests: `tests/llb/prep/book/` (ranges, run, resume) and `tests/llb/cli/test_cli_book_ocr.py`, over a
synthetic six-page scan built by `tests/llb/prep/book/_scan_factory.py` -- grey line patterns at
150 dpi, one landscape page, one RGB page, one landscape scan placed rotated onto a portrait page,
one page with embedded text -- so no page of a real book enters git.

## How a run decides what to do

**Walk order.** Pages go in chapter-sized batches (`--batch-pages`, default 25). Inside a batch each
stage runs over every page before the next stage starts, so a model-backed stage will load its model
once per batch rather than once per page. Concurrent GPU stages are out of scope.

**The stage seam.** A stage declares its `name`, `version`, the upstream stages it reads
(`inputs`), whether it reads the source PDF, whether it needs a model, its `tool_identity`, how it
validates parameters, and `effective_params` -- its parameters with every default and `auto`
resolved for this page. It writes files into a private work directory and never decides whether it
should run.

**The cache key** is the SHA-256 of the stage name and version, the tool identity, the model, the
digest of the EFFECTIVE parameters, and the input digests: the source checksum and PDF page for a
stage that reads the PDF, each upstream stage's output digest otherwise. The `render` key is a
function of the source bytes, and a recognition key is a function of the page-image digest. A stage
whose output names the page -- the ledger, the signal file -- also includes the PDF page, so
identical images do not share that file. A value a stage derives from the page -- `render`'s native
resolution, its automatic colorspace -- is in the key. Every component is stored beside the key in
`stage.json`, so a reader can see why two records share one. `resolve` also digests the page's
reviewer decisions; that file lives outside the stage cache
([uncertainty and review](uncertainty-and-review.md)). `correct` digests the facing block on
each neighbour page, so an interior change there does not re-run this page
([evidence-constrained correction](correction.md)).

**Per page and stage**, in order:

1. An upstream stage not `done` on this page -> `blocked` (nothing runs; nothing is written).
2. A committed record under the same key whose outputs still hash to what it recorded -> skipped.
3. A record under the same key in the `--reuse-from` run -> its directory is copied in and committed
   with `reused_from` set.
4. Otherwise the stage runs in `pages/<pNNNN>/.<stage>.partial/`. On success the record is written
   into that directory last and the directory is renamed to `pages/<pNNNN>/<stage>/`, so a reader
   sees a whole stage or none. An `Exception` is retried up to `--max-attempts` and then committed as
   a `failed` record with the error and no outputs; it fails that page's stage, never the job.

A `BaseException` -- Ctrl-C, or the kill the interpreter never sees -- is not caught: it leaves only
the `.partial` directory, which nothing reads and the next invocation deletes before it starts.

**The ledger.** `metadata/page_manifest.json` lists every selected page with its batch, inventory,
page-image digest, and each stage's status (`done`, `failed`, `blocked`, `pending`), cache key,
attempts, error, and reuse origin, plus per-stage totals. It is rebuilt from the stage records rather
than accumulated, so `status` after a kill reports exactly what a resume will find.

**The source.** Identified by the PDF lane's SHA-256 fingerprint and opened read-only. `symlink`
(default) and `copy` place it under `source/`, the copy made read-only; `reference` records only its
absolute path. A run refuses an `--out-dir` that is, or lies inside, the directory holding the
source. A resume re-hashes the stored source and any PDF given on the command line and refuses a
mismatch rather than mixing pages of two documents.

**The inventory** is what the PDF says before anything renders: page size, `/Rotate`, page-box
orientation, image count, the largest image's pixel size, its native resolution, the quarter turn
it is PLACED at, and embedded-text characters. The resolution is measured along the image's own
pixel rows through its placement matrix, not across the bounding box -- see the evidence below for
why that matters. The placement rotation says nothing about whether the printed content reads
upright; that measurement is the signals stage
([page signals and variants](page-preprocessing.md#what-is-measured)).

## Contracts

`config.json`, every `stage.json`, `pages/<pNNNN>/blocks.json`, and `metadata/page_manifest.json`
are registered families -- `llb.book-ocr-run-config`, `llb.book-ocr-stage-record`,
`llb.book-ocr-block-ledger`, and `llb.book-ocr-page-manifest`, each at `1.0.0` with no older form
(see [data-prep contracts](../artifact-contracts/data-prep-contracts.md#scanned-book-run-families)).
The block ledger is what a later stage reads
([block ledger](block-ledger.md)). Resolution adds `llb.book-ocr-resolved-page`,
`llb.book-ocr-correction-overlay`, `llb.book-ocr-model-corrections`, and
`llb.book-ocr-reviewer-decisions`
([uncertainty and review](uncertainty-and-review.md)). The review queue adds
`llb.book-ocr-review-queue` and `llb.book-ocr-quality-manifest`; that command is not a stage.
`llb book-ocr structure` adds `llb.book-ocr-document-structure`; that command is not a stage either
([structure](structure-and-rendering.md)). The
processing report and the JSON-lines
log are human and diagnostic outputs and stay unregistered, as the data-prep policy does for
Markdown reports.

## Evidence

**What ran.** On 2026-11-19 on the RTX PRO 3000 Blackwell laptop host (Intel Core Ultra 7 265HX,
CPU-only for this capability: no model loaded, no GPU memory allocated), over the 1,067-page target
handbook scan (163.7 MB; every page an image, no embedded text), with the stage list `render,
fake-preprocess, fake-ocr`, default batches of 25 and the symlink policy.

- **Full run.** 3,201 page stages (1,067 x 3) completed in 49 s wall at about 0.5 GB resident
  memory, writing 0.9 GB. The mean render took 0.04 s per page. Rendering, not OCR, is all this
  measures; it says the skeleton adds no meaningful cost to a 1,000-page job.
- **A real kill.** A fresh run killed with `SIGKILL` after 12 s had committed 698 page stages
  (render through page 248) and left exactly one `.partial` directory (page 249's render). The resume
  skipped all 698, ran the remaining 2,503, deleted the partial, and its 1,067 page images were
  byte-identical to the uninterrupted run's. The log's `stage_done` events total 3,201 across both
  invocations: no stage executed twice.
- **The inventory caught a mis-measured resolution.** 1,022 pages place a portrait image upright
  (1,240 x 1,654 on all but the 1,238 x 1,592 cover); 45 (PDF pages 126-150, 152-160, 181,
  445-454) place a 1,654 x 1,240 landscape scan rotated onto the portrait page, 26 at a quarter
  turn and 19 at three quarters. The first build
  measured resolution across the bounding box and read those 45 as 200 dpi; measured along the
  placement matrix all 1,067 read 150 dpi. Sampled page 126 renders upside down, so content
  orientation cannot be read from the placement. Measuring and correcting it is the signals stage
  ([page signals and variants](page-preprocessing.md)).
- **Content addressing held.** That fix changed the `render` key of every page (its effective
  parameters now carry the resolved resolution). A run reusing the pre-fix run re-rendered all
  1,067 pages, took `fake-preprocess` and `fake-ocr` from the earlier run for the 1,022 pages whose
  image bytes did not change, and re-ran them for exactly the 45 rotated pages whose image did.
- **Failure isolation and reuse at small scale.** On PDF pages 1-40 with forced failures on page 4
  (`fake-preprocess`) and page 7 (`fake-ocr`), each failed after 2 attempts and failed only its own
  stage; page 4's `fake-ocr`, which reads only the render, completed. A fresh run with
  `--reuse-from` and one changed `fake-ocr` parameter took all 40 renders and 39 completed
  preprocess stages from the earlier run and re-ran only `fake-ocr`.

**What would overturn it.** A completed stage executing again on resume under an unchanged key; a
key that stays the same when anything its output depends on changes (the class of the resolution
bug above -- a stage that resolves a value from the page and does not return it from
`effective_params` reintroduces it); or a reader seeing a stage directory without its record.

The `make ci` gates pin each property on the synthetic scan: range refusals by name, deterministic
page and stage naming, a simulated kill mid-write that leaves only a `.partial` and is re-run alone
on resume, a forced failure that fails only its page and appears in the report, a downstream stage
blocked by its failed upstream while an unrelated sibling completes, a parameter change that re-runs
only its stage, a resolved-resolution change that re-renders, a source-checksum mismatch on resume
that refuses, and the source byte-identical after a run under all three policies.
