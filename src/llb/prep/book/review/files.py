"""The run files the review queue reads, and the two documents it writes."""

from pathlib import Path

import numpy as np

from llb.core.contracts.data_prep.book_blocks import PageBlockLedger
from llb.core.contracts.data_prep.book_resolve import (
    CorrectionEdit,
    PageCorrectionOverlay,
    PageResolvedText,
)
from llb.core.contracts.data_prep.book_review import (
    MANIFEST_KIND,
    MANIFEST_SCHEMA_ID,
    QUEUE_KIND,
    QUEUE_SCHEMA_ID,
    REVIEW_SCHEMA_VERSION,
    QualityManifest,
    QualityPage,
    QualityRow,
    ReviewQueue,
    ReviewQueueEntry,
)
from llb.prep.book.constants import BLOCKS_FILENAME, PAGES_DIRNAME
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.preprocess.images import read_pixels
from llb.prep.book.resolve.constants import OVERLAY_FILENAME, RESOLVED_FILENAME, REVIEW_DIRNAME
from llb.prep.book.resolve.io import decisions_path, read_decisions, read_overlay, read_resolved
from llb.prep.book.review.constants import MARKER_KIND_ORDER
from llb.prep.book.review.crops import CROP_NAME
from llb.prep.book.stages.render import PAGE_IMAGE_FILENAME, RENDER_STAGE


def quality_row(entry: ReviewQueueEntry) -> QualityRow:
    """The manifest row for one queue entry. The two lists name the same span."""
    return QualityRow(
        entry_id=entry.entry_id,
        block_id=entry.block_id,
        kind=entry.kind,
        reason=entry.reason,
        start=entry.start,
        end=entry.end,
        classes=list(entry.classes),
        marker=entry.marker,
        crop=entry.crop,
        raw_text=entry.raw_text,
    )


def queue_documents(
    source_sha256: str, entries: list[ReviewQueueEntry], pages: list[QualityPage]
) -> tuple[ReviewQueue, QualityManifest]:
    """The queue and the manifest, with counts that match the rows."""
    counts = {
        kind: sum(row.kind == kind for page in pages for row in page.rows)
        for kind in MARKER_KIND_ORDER
    }
    present: dict[str, int] = {kind: count for kind, count in counts.items() if count}
    queue = ReviewQueue(
        schema_id=QUEUE_SCHEMA_ID,
        schema_version=REVIEW_SCHEMA_VERSION,
        kind=QUEUE_KIND,
        source_sha256=source_sha256,
        entries=entries,
    )
    manifest = QualityManifest(
        schema_id=MANIFEST_SCHEMA_ID,
        schema_version=REVIEW_SCHEMA_VERSION,
        kind=MANIFEST_KIND,
        source_sha256=source_sha256,
        counts=present,
        pages=pages,
    )
    return queue, manifest


def load_page(
    run_dir: Path, page_id: str
) -> tuple[PageBlockLedger, PageResolvedText, PageCorrectionOverlay, np.ndarray | None] | None:
    """One page's ledger, resolved text, overlay, and render. None when resolve is unfinished."""
    base = run_dir / PAGES_DIRNAME / page_id
    resolved_path = base / RESOLVED_FILENAME
    ledger_path = base / BLOCKS_FILENAME
    overlay_path = base / OVERLAY_FILENAME
    if not resolved_path.is_file() or not ledger_path.is_file() or not overlay_path.is_file():
        return None
    image_path = base / RENDER_STAGE / PAGE_IMAGE_FILENAME
    image = read_pixels(image_path) if image_path.is_file() else None
    return read_ledger(ledger_path), read_resolved(resolved_path), read_overlay(overlay_path), image


def reviewer_edits(run_dir: Path) -> list[CorrectionEdit]:
    """Reviewer edits already on disk. A missing file is an empty list."""
    path = decisions_path(run_dir)
    if not path.is_file():
        return []
    return list(read_decisions(path).edits)


def pdf_target(stored: str, run_dir: Path) -> str:
    """The PDF link target, relative to the run when the source lives inside it."""
    path = Path(stored)
    if path.is_absolute():
        try:
            path = path.relative_to(run_dir)
        except ValueError:
            return stored.replace("\\", "/")
    return path.as_posix()


def drop_stale_crops(run_dir: Path, keep: set[str]) -> None:
    """Remove review crops this build did not write. Decisions and the queue stay."""
    review = run_dir / REVIEW_DIRNAME
    if not review.is_dir():
        return
    for path in review.glob("*.png"):
        if CROP_NAME.match(path.name) and f"{REVIEW_DIRNAME}/{path.name}" not in keep:
            path.unlink()


def missing_pages(page_ids: list[str]) -> str:
    """A short list of page ids, with a count when the list is long."""
    shown = ", ".join(page_ids[:8])
    extra = f" (+{len(page_ids) - 8})" if len(page_ids) > 8 else ""
    return f"{shown}{extra}"
