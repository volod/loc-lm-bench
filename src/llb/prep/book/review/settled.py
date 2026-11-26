"""Whether a page failed for good, and whether the run has stopped moving.

A terminal failure has a failed stage, nothing still pending, and no resolved
text. An unfinished page is still pending, or it has no record at all. The
queue and the structure step accept the first and refuse the second.
"""

from pathlib import Path

from llb.core.contracts.data_prep.book import BookPageManifest, PageLedgerRecord
from llb.prep.book.constants import PAGES_DIRNAME, STATUS_FAILED, STATUS_PENDING
from llb.prep.book.ledger.manifest import read_manifest
from llb.prep.book.resolve.constants import RESOLVED_FILENAME

# The vault title has to stay one short line. The same cut the page-failed marker uses.
_REASON_LIMIT = 120


def terminal_failure(run_dir: Path, page_id: str) -> bool:
    """True when recognition failed and the page will not gain a resolved text."""
    page = _record(run_dir, page_id)
    if page is None or _pending(page) or not _failed(page):
        return False
    return not (run_dir / PAGES_DIRNAME / page_id / RESOLVED_FILENAME).is_file()


def failure_reason(run_dir: Path, page_id: str) -> str:
    """The failed stage's error, shortened to one line, or a generic reason."""
    page = _record(run_dir, page_id)
    if page is None:
        return "recovery failed"
    for stage in page.stages:
        if stage.status == STATUS_FAILED and stage.error:
            return _plain(stage.error)
    return "recovery failed"


def run_settled(manifest: BookPageManifest | None) -> bool:
    """True when every configured page is done, failed, or blocked behind a failure."""
    if manifest is None:
        return False
    return all(not _pending(page) for page in manifest.pages)


def _record(run_dir: Path, page_id: str) -> PageLedgerRecord | None:
    manifest = read_manifest(run_dir)
    if manifest is None:
        return None
    for page in manifest.pages:
        if page.page_id == page_id:
            return page
    return None


def _pending(page: PageLedgerRecord) -> bool:
    return any(stage.status == STATUS_PENDING for stage in page.stages)


def _failed(page: PageLedgerRecord) -> bool:
    return any(stage.status == STATUS_FAILED for stage in page.stages)


def _plain(text: str) -> str:
    cleaned = "".join(ch if 32 <= ord(ch) < 127 and ch != '"' else " " for ch in text)
    compact = " ".join(cleaned.split())
    return (compact or "recovery failed")[:_REASON_LIMIT]
