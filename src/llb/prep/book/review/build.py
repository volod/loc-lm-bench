"""Build the review queue, its crops, and the page quality manifest for one run."""

import logging
from pathlib import Path

import numpy as np

from llb.core.contracts.data_prep.book_blocks import Block, PageBlockLedger
from llb.core.contracts.data_prep.book_resolve import (
    CorrectionEdit,
    PageCorrectionOverlay,
    PageResolvedText,
    ResolvedBlock,
)
from llb.core.contracts.data_prep.book_review import (
    CoverageStatus,
    QualityManifest,
    QualityPage,
    ReviewQueue,
    ReviewQueueEntry,
)
from llb.prep.book.config import read_config
from llb.prep.book.ranges import page_id
from llb.prep.book.resolve.constants import REVIEW_DIRNAME
from llb.prep.book.cache import page_dir
from llb.prep.book.review.classify import page_marks
from llb.prep.book.review.coverage import account_page
from llb.prep.book.review.crops import crop_filename, write_block_crop
from llb.prep.book.review.gaps import failure_entry, omission_marks
from llb.prep.book.review.settled import failure_reason, terminal_failure
from llb.prep.book.review.files import (
    drop_stale_crops,
    load_page,
    missing_pages,
    pdf_target,
    quality_row,
    queue_documents,
    reviewer_edits,
)
from llb.prep.book.review.io import manifest_path, queue_path, write_manifest, write_queue
from llb.prep.book.review.marks import SpanMark
from llb.prep.book.review.markers import render_marker, render_passage

_LOG = logging.getLogger(__name__)
_Prepared = tuple[SpanMark, str, str | None, str]


def build_review_queue(run_dir: Path) -> tuple[ReviewQueue, QualityManifest]:
    """Write the queue, the crops, and the manifest.

    A page whose recognition failed terminally is a page-failure entry. A page
    that has not finished is refused. A blank scan is recorded and not queued.
    """
    config = read_config(run_dir)
    edits = reviewer_edits(run_dir)
    link = pdf_target(config.source.stored_path, run_dir)
    pages: list[QualityPage] = []
    entries: list[ReviewQueueEntry] = []
    crops: set[str] = set()
    missing: list[str] = []
    for pdf_page in config.pages:
        pid = page_id(pdf_page, config.source.page_count)
        pdf_link = f"{link}#page={pdf_page}"
        loaded = load_page(run_dir, pid)
        if loaded is None:
            if not terminal_failure(run_dir, pid):
                missing.append(pid)
                continue
            failure = failure_entry(pid, pdf_page, pdf_link, failure_reason(run_dir, pid), run_dir)
            entries.append(failure)
            if failure.crop:
                crops.add(failure.crop)
            pages.append(_quality(pid, pdf_page, "failed", [failure]))
            continue
        page_entries, page_crops, coverage = _accounted(*loaded, edits, pdf_link, run_dir, pid)
        entries.extend(page_entries)
        crops.update(page_crops)
        pages.append(_quality(pid, pdf_page, coverage, page_entries))
    if missing:
        raise ValueError(f"resolve has not finished for {missing_pages(missing)}")
    drop_stale_crops(run_dir, crops)
    queue, manifest = queue_documents(config.source.sha256, entries, pages)
    write_queue(queue_path(run_dir), queue)
    write_manifest(manifest_path(run_dir), manifest)
    _LOG.info("[book-review] %s entries across %s pages", len(entries), len(pages))
    return queue, manifest


def _quality(
    page_id: str, pdf_page: int, coverage: CoverageStatus, entries: list[ReviewQueueEntry]
) -> QualityPage:
    return QualityPage(
        page_id=page_id,
        pdf_page=pdf_page,
        coverage=coverage,
        rows=[quality_row(item) for item in entries],
    )


def _accounted(
    ledger: PageBlockLedger,
    resolved: PageResolvedText,
    overlay: PageCorrectionOverlay,
    image: np.ndarray | None,
    edits: list[CorrectionEdit],
    pdf_link: str,
    run_dir: Path,
    pid: str,
) -> tuple[list[ReviewQueueEntry], set[str], CoverageStatus]:
    account = account_page(image, ledger, page_dir(run_dir, pid))
    page_entries, page_crops = _page_entries(
        ledger,
        resolved,
        overlay,
        image,
        edits,
        pdf_link,
        run_dir,
        omission_marks(account, pid, ledger.pdf_page),
    )
    return page_entries, page_crops, account.coverage


def _page_entries(
    ledger: PageBlockLedger,
    resolved: PageResolvedText,
    overlay: PageCorrectionOverlay,
    image: np.ndarray | None,
    edits: list[CorrectionEdit],
    pdf_link: str,
    run_dir: Path,
    extra: list[SpanMark],
) -> tuple[list[ReviewQueueEntry], set[str]]:
    blocks = {block.block_id: block for block in ledger.blocks}
    marks = page_marks(ledger.page_id, blocks, resolved, overlay, edits)
    marks.extend(extra)
    prepared = [_prepare(mark, blocks, image, pdf_link, run_dir) for mark in marks]
    passages = _passages(prepared, {block.block_id: block for block in resolved.blocks})
    entries = [
        _entry(mark, marker, crop, pdf_link, passages[mark.block_id])
        for mark, marker, crop, _link in prepared
    ]
    return entries, {crop for _mark, _marker, crop, _link in prepared if crop}


def _prepare(
    mark: SpanMark,
    blocks: dict[str, Block],
    image: np.ndarray | None,
    pdf_link: str,
    run_dir: Path,
) -> _Prepared:
    crop = _crop(mark, blocks.get(mark.block_id), image, run_dir)
    return (
        mark,
        render_marker(mark.kind, pdf_link, mark.block_id, mark.start, mark.end, crop),
        crop,
        pdf_link,
    )


def _crop(
    mark: SpanMark, block: Block | None, image: np.ndarray | None, run_dir: Path
) -> str | None:
    region = mark.region
    if region is None and block is not None and mark.block_present:
        region = block.region
    if image is None or region is None:
        return None
    name = crop_filename(mark.block_id, mark.start, mark.end)
    if not write_block_crop(image, region, run_dir / REVIEW_DIRNAME / name):
        return None
    return f"{REVIEW_DIRNAME}/{name}"


def _passages(prepared: list[_Prepared], resolved: dict[str, ResolvedBlock]) -> dict[str, str]:
    grouped: dict[str, list[tuple[SpanMark, str]]] = {}
    for mark, marker, _crop, _link in prepared:
        grouped.setdefault(mark.block_id, []).append((mark, marker))
    passages: dict[str, str] = {}
    for block_id, items in grouped.items():
        markers = [marker for _mark, marker in items]
        block = resolved.get(block_id)
        passages[block_id] = _passage(block, items) if block is not None else " ".join(markers)
    return passages


def _passage(block: ResolvedBlock, items: list[tuple[SpanMark, str]]) -> str:
    markers = [marker for _mark, marker in items]
    replacements = {
        (mark.start, mark.end): marker for mark, marker in items if mark.reason == "unresolved"
    }
    prefixes = [marker for mark, marker in items if mark.reason != "unresolved"]
    text = render_passage(block, replacements, prefixes)
    return text if text.strip() else " ".join(markers)


def _entry(
    mark: SpanMark, marker: str, crop: str | None, pdf_link: str, rendered: str
) -> ReviewQueueEntry:
    return ReviewQueueEntry(
        entry_id=_entry_id(mark),
        page_id=mark.page_id,
        pdf_page=mark.pdf_page,
        block_id=mark.block_id,
        ordinal=mark.ordinal,
        anchor=mark.anchor,
        kind=mark.kind,
        reason=mark.reason,
        start=mark.start,
        end=mark.end,
        raw_text=mark.raw_text,
        candidate=mark.candidate,
        classes=list(mark.classes),
        marker=marker,
        crop=crop,
        pdf_link=pdf_link,
        rendered=rendered,
        priors=list(mark.priors),
        verdict=mark.verdict,
        block_present=mark.block_present,
        box=_box(mark),
    )


def _entry_id(mark: SpanMark) -> str:
    return crop_filename(mark.block_id, mark.start, mark.end).removesuffix(".png")


def _box(mark: SpanMark) -> list[float]:
    if mark.region is None:
        return []
    return [mark.region.x0, mark.region.y0, mark.region.x1, mark.region.y1]
