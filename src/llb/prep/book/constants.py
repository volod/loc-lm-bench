"""Names, layout, and defaults of a `book-ocr` run directory.

One run lives under `$DATA_DIR/book-ocr/<run_timestamp>/`:

    config.json                     the run configuration (`llb.book-ocr-run-config`)
    source/<name>.pdf               the source PDF under the copy or symlink policy
    pages/<pNNNN>/<stage>/          one committed stage output, with its `stage.json` record
    pages/<pNNNN>/.<stage>.partial/ an uncommitted stage attempt (never read; removed on resume)
    pages/<pNNNN>/blocks.json       the page block ledger (`llb.book-ocr-block-ledger`)
    pages/<pNNNN>/overlay.json      applied and orphaned corrections (`llb.book-ocr-correction-overlay`)
    pages/<pNNNN>/resolved.json     the resolved text (`llb.book-ocr-resolved-page`)
    review/decisions.json           reviewer edits, outside the stage cache
    review/queue.json               uncertain spans for the review workbench
    metadata/page_manifest.json     the per-page, per-stage ledger (`llb.book-ocr-page-manifest`)
    metadata/quality_manifest.json  one row per uncertain span (`llb.book-ocr-quality-manifest`)
    metadata/processing_report.md   the human-readable run report
    logs/run.jsonl                  structured JSON event log, one object per line
"""

from typing import Final

METHOD_DIRNAME = "book-ocr"
CONFIG_FILENAME = "config.json"
SOURCE_DIRNAME = "source"
PAGES_DIRNAME = "pages"
METADATA_DIRNAME = "metadata"
LOGS_DIRNAME = "logs"
PAGE_MANIFEST_FILENAME = "page_manifest.json"
PROCESSING_REPORT_FILENAME = "processing_report.md"
RUN_LOG_FILENAME = "run.jsonl"
STAGE_RECORD_FILENAME = "stage.json"
BLOCKS_FILENAME = "blocks.json"  # published page ledger; also the `blocks` stage output
READING_FILENAME = "reading.json"  # one engine stage's normalized blocks, pre-ledger
PARTIAL_SUFFIX = ".partial"
TRASH_SUFFIX = ".trash"

RUN_CONFIG_SCHEMA_ID: Final = "llb.book-ocr-run-config"
PAGE_MANIFEST_SCHEMA_ID: Final = "llb.book-ocr-page-manifest"
STAGE_RECORD_SCHEMA_ID: Final = "llb.book-ocr-stage-record"
BLOCK_LEDGER_SCHEMA_ID: Final = "llb.book-ocr-block-ledger"
CONTRACT_VERSION: Final = "1.0.0"

# Page ids are `p` plus the 1-based PDF page, zero-padded to at least this many digits (a
# 1,067-page book is `p0001`..`p1067`); a longer document widens every id equally.
PAGE_ID_MIN_DIGITS = 4
PAGE_ID_PREFIX = "p"

SOURCE_POLICY_COPY: Final = "copy"
SOURCE_POLICY_SYMLINK: Final = "symlink"
SOURCE_POLICY_REFERENCE: Final = "reference"
SOURCE_POLICIES = (SOURCE_POLICY_COPY, SOURCE_POLICY_SYMLINK, SOURCE_POLICY_REFERENCE)
DEFAULT_SOURCE_POLICY = SOURCE_POLICY_SYMLINK

DEFAULT_PAGES = "1-"  # every page
DEFAULT_BATCH_PAGES = 25  # roughly a chapter of a technical handbook
DEFAULT_MAX_ATTEMPTS = 2  # one retry, then the page's stage is recorded as failed
DEFAULT_STAGES = ("render",)

RUN_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S.%fZ"

STATUS_DONE: Final = "done"
STATUS_FAILED: Final = "failed"
STATUS_PENDING: Final = "pending"
STATUS_BLOCKED: Final = "blocked"  # an upstream stage of the same page failed
PAGE_STATUSES = (STATUS_DONE, STATUS_FAILED, STATUS_BLOCKED, STATUS_PENDING)
