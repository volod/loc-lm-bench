"""Read a run into resolved blocks. A page with no resolved text is a failure.

The text on a block is the resolved text. A raw reading is never chosen here,
including on a page whose recovery failed: that page becomes one marker.
"""

from pathlib import Path

import numpy as np

from llb.core.contracts.data_prep.book_blocks import Block, PageBlockLedger, Region
from llb.core.contracts.data_prep.book_resolve import (
    CorrectionEdit,
    PageCorrectionOverlay,
    PageResolvedText,
    ResolvedBlock,
)
from llb.prep.book.cache import page_dir
from llb.prep.book.config import read_config
from llb.prep.book.constants import BLOCKS_FILENAME
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.ledger.manifest import read_manifest
from llb.prep.book.preprocess.images import read_grey
from llb.prep.book.ranges import page_id
from llb.prep.book.render.constants import FAIL_REASON_LIMIT
from llb.prep.book.render.types import BookLoad, RenderBlock
from llb.prep.book.resolve.constants import (
    OVERLAY_FILENAME,
    OVERLAY_KIND,
    OVERLAY_SCHEMA_ID,
    RESOLVED_FILENAME,
    SCHEMA_VERSION,
)
from llb.prep.book.resolve.io import read_overlay, read_resolved
from llb.prep.book.review.classify import page_marks
from llb.prep.book.review.constants import CROP_PADDING_PX
from llb.prep.book.review.crops import crop_filename, padded_box
from llb.prep.book.review.files import reviewer_edits
from llb.prep.book.review.marks import SpanMark
from llb.prep.book.source import stored_source_path
from llb.prep.book.stages.render import PAGE_IMAGE_FILENAME, RENDER_STAGE
from llb.prep.book.structure.io import read_structure, structure_path


def load_book(run_dir: Path) -> BookLoad:
    """The resolved blocks of a run, in reading order, plus one marker per failure."""
    config = read_config(run_dir)
    path = structure_path(run_dir)
    if not path.is_file():
        raise ValueError(f"{run_dir} has no document structure; run book-ocr structure first")
    structure = read_structure(path)
    if structure.source_sha256 != config.source.sha256:
        raise ValueError("document structure is for a different source PDF")
    if [page.pdf_page for page in structure.pages] != list(config.pages):
        raise ValueError(
            "document structure does not match the run's pages; run book-ocr structure"
        )
    pdf = stored_source_path(run_dir, config.source)
    if not pdf.is_file():
        raise ValueError(f"source PDF {pdf} is missing")
    edits = reviewer_edits(run_dir)
    blocks: list[RenderBlock] = []
    images: dict[int, np.ndarray] = {}
    for pdf_page in config.pages:
        pid = page_id(pdf_page, config.source.page_count)
        page_blocks, image = _page(run_dir, pid, pdf_page, edits)
        blocks.extend(page_blocks)
        if image is not None:
            images[pdf_page] = image
    blocks.sort(key=lambda block: (block.pdf_page, block.reading_order, block.ordinal))
    return BookLoad(config, structure, pdf, blocks, images)


def _page(
    run_dir: Path, pid: str, pdf_page: int, edits: list[CorrectionEdit]
) -> tuple[list[RenderBlock], np.ndarray | None]:
    base = page_dir(run_dir, pid)
    resolved_path = base / RESOLVED_FILENAME
    if not resolved_path.is_file():
        return [_failed(pid, pdf_page, _reason(run_dir, pid))], _image(base)
    ledger_path = base / BLOCKS_FILENAME
    if not ledger_path.is_file():
        raise ValueError(f"{pid} has resolved text and no block ledger")
    ledger = read_ledger(ledger_path)
    resolved = read_resolved(resolved_path)
    overlay = _overlay(base, pid, pdf_page)
    image = _image(base)
    marks = _marks(pid, ledger, resolved, overlay, edits)
    return _blocks(pid, ledger, resolved, marks, image), image


def _blocks(
    pid: str,
    ledger: PageBlockLedger,
    resolved: PageResolvedText,
    marks: list[SpanMark],
    image: np.ndarray | None,
) -> list[RenderBlock]:
    by_id = {block.block_id: block for block in resolved.blocks}
    grouped: dict[str, list[SpanMark]] = {}
    for mark in marks:
        grouped.setdefault(mark.block_id, []).append(mark)
    built: list[RenderBlock] = []
    seen: set[str] = set()
    for source in sorted(ledger.blocks, key=lambda block: (block.reading_order, block.ordinal)):
        seen.add(source.block_id)
        built.append(_from_ledger(pid, source, by_id.get(source.block_id), grouped, image))
    for block in resolved.blocks:
        if block.block_id not in seen:
            built.append(_from_resolved(pid, resolved.pdf_page, block, grouped, image))
    return built


def _from_ledger(
    pid: str,
    source: Block,
    resolved: ResolvedBlock | None,
    grouped: dict[str, list[SpanMark]],
    image: np.ndarray | None,
) -> RenderBlock:
    own = tuple(grouped.get(source.block_id, ()))
    if resolved is None:
        return _render_block(pid, source, None, "", own, image, "missing-block")
    return _render_block(pid, source, resolved, resolved.text, own, image, source.kind)


def _from_resolved(
    pid: str,
    pdf_page: int,
    resolved: ResolvedBlock,
    grouped: dict[str, list[SpanMark]],
    image: np.ndarray | None,
) -> RenderBlock:
    own = tuple(grouped.get(resolved.block_id, ()))
    return RenderBlock(
        pdf_page=pdf_page,
        page_id=pid,
        ordinal=0,
        reading_order=10**6,
        kind="unknown",
        role="unknown",
        block_id=resolved.block_id,
        digest=resolved.anchor,
        text=resolved.text,
        region=None,
        resolved=resolved,
        marks=own,
        crops=_crops(own, None, image),
    )


def _render_block(
    pid: str,
    source: Block,
    resolved: ResolvedBlock | None,
    text: str,
    marks: tuple[SpanMark, ...],
    image: np.ndarray | None,
    role: str,
) -> RenderBlock:
    digest = resolved.anchor if resolved is not None else ""
    return RenderBlock(
        pdf_page=source.pdf_page,
        page_id=pid,
        ordinal=source.ordinal,
        reading_order=source.reading_order,
        kind=source.kind,
        role=role,
        block_id=source.block_id,
        digest=digest,
        text=text,
        region=source.region,
        resolved=resolved,
        marks=marks,
        crops=_crops(marks, source.region, image),
    )


def _crops(
    marks: tuple[SpanMark, ...], region: Region | None, image: np.ndarray | None
) -> tuple[tuple[int, int, str], ...]:
    if region is None or image is None or padded_box(region, image.shape, CROP_PADDING_PX) is None:
        return ()
    return tuple(
        (mark.start, mark.end, f"review/{crop_filename(mark.block_id, mark.start, mark.end)}")
        for mark in marks
    )


def _failed(pid: str, pdf_page: int, reason: str) -> RenderBlock:
    return RenderBlock(
        pdf_page=pdf_page,
        page_id=pid,
        ordinal=0,
        reading_order=0,
        kind="page-failed",
        role="page-failed",
        block_id="",
        digest="",
        text="",
        region=None,
        resolved=None,
        failed=True,
        fail_reason=reason,
    )


def _overlay(base: Path, pid: str, pdf_page: int) -> PageCorrectionOverlay:
    path = base / OVERLAY_FILENAME
    if path.is_file():
        return read_overlay(path)
    return PageCorrectionOverlay(
        schema_id=OVERLAY_SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        kind=OVERLAY_KIND,
        page_id=pid,
        pdf_page=pdf_page,
    )


def _image(base: Path) -> np.ndarray | None:
    path = base / RENDER_STAGE / PAGE_IMAGE_FILENAME
    if not path.is_file():
        return None
    return read_grey(path)


def _marks(
    pid: str,
    ledger: PageBlockLedger,
    resolved: PageResolvedText,
    overlay: PageCorrectionOverlay,
    edits: list[CorrectionEdit],
) -> list[SpanMark]:
    blocks = {block.block_id: block for block in ledger.blocks}
    return page_marks(pid, blocks, resolved, overlay, edits)


def _reason(run_dir: Path, pid: str) -> str:
    manifest = read_manifest(run_dir)
    if manifest is None:
        return "recovery failed"
    for page in manifest.pages:
        if page.page_id != pid:
            continue
        for stage in page.stages:
            if stage.status == "failed" and stage.error:
                return _plain(stage.error)
    return "recovery failed"


def _plain(text: str) -> str:
    cleaned = "".join(ch if 32 <= ord(ch) < 127 and ch != '"' else " " for ch in text)
    compact = " ".join(cleaned.split())
    return (compact or "recovery failed")[:FAIL_REASON_LIMIT]
