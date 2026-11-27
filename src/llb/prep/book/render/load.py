"""Read a run into resolved blocks. A page with no resolved text is a failure.

The text on a block is the resolved text. A raw reading is never chosen here,
including on a page whose recovery failed: that page becomes one marker.
"""

from pathlib import Path

from llb.core.contracts.data_prep.book_blocks import Block, PageBlockLedger, RawReading, Region
from llb.core.contracts.data_prep.book_resolve import (
    CorrectionEdit,
    PageCorrectionOverlay,
    PageResolvedText,
    ResolvedBlock,
)
from llb.prep.book.cache import page_dir
from llb.prep.book.config import read_config
from llb.prep.book.constants import BLOCKS_FILENAME
from llb.prep.book.engines.image import png_size
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.ranges import page_id
from llb.prep.book.preprocess.images import read_grey
from llb.prep.book.render.bars import (
    caption_is_separate,
    formula_needs_crop,
    formula_of,
    table_needs_evidence,
    table_needs_image,
    table_of,
)
from llb.prep.book.render.omissions import with_coverage
from llb.prep.book.render.types import BookLoad, RenderBlock
from llb.prep.book.review.settled import failure_reason
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
from llb.prep.book.source import verify_source
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
    pdf = verify_source(run_dir, config.source)
    edits = reviewer_edits(run_dir)
    blocks: list[RenderBlock] = []
    images: dict[int, Path] = {}
    for pdf_page in config.pages:
        pid = page_id(pdf_page, config.source.page_count)
        path = page_dir(run_dir, pid) / RENDER_STAGE / PAGE_IMAGE_FILENAME
        shape = None
        if path.is_file():
            images[pdf_page] = path
            width, height = png_size(path)
            shape = (height, width)
        blocks.extend(_page(run_dir, pid, pdf_page, edits, shape))
    blocks.sort(key=lambda block: (block.pdf_page, block.reading_order, block.ordinal))
    return BookLoad(config, structure, pdf, blocks, images)


def _page(
    run_dir: Path,
    pid: str,
    pdf_page: int,
    edits: list[CorrectionEdit],
    shape: tuple[int, int] | None,
) -> list[RenderBlock]:
    base = page_dir(run_dir, pid)
    resolved_path = base / RESOLVED_FILENAME
    if not resolved_path.is_file():
        return [_failed(pid, pdf_page, failure_reason(run_dir, pid), shape)]
    ledger_path = base / BLOCKS_FILENAME
    if not ledger_path.is_file():
        raise ValueError(f"{pid} has resolved text and no block ledger")
    ledger = read_ledger(ledger_path)
    resolved = read_resolved(resolved_path)
    overlay = _overlay(base, pid, pdf_page)
    marks = _marks(pid, ledger, resolved, overlay, edits)
    built = _blocks(pid, ledger, resolved, marks, shape)
    image = read_grey(base / RENDER_STAGE / PAGE_IMAGE_FILENAME) if shape is not None else None
    return with_coverage(built, image, ledger, base)


def _blocks(
    pid: str,
    ledger: PageBlockLedger,
    resolved: PageResolvedText,
    marks: list[SpanMark],
    shape: tuple[int, int] | None,
) -> list[RenderBlock]:
    by_id = {block.block_id: block for block in resolved.blocks}
    grouped: dict[str, list[SpanMark]] = {}
    for mark in marks:
        grouped.setdefault(mark.block_id, []).append(mark)
    captions = _caption_texts(ledger, by_id)
    built: list[RenderBlock] = []
    seen: set[str] = set()
    for source in sorted(ledger.blocks, key=lambda block: (block.reading_order, block.ordinal)):
        seen.add(source.block_id)
        built.append(
            _from_ledger(pid, source, by_id.get(source.block_id), grouped, shape, captions)
        )
    for block in resolved.blocks:
        if block.block_id not in seen:
            built.append(_from_resolved(pid, resolved.pdf_page, block, grouped))
    return built


def _from_ledger(
    pid: str,
    source: Block,
    resolved: ResolvedBlock | None,
    grouped: dict[str, list[SpanMark]],
    shape: tuple[int, int] | None,
    captions: set[str],
) -> RenderBlock:
    own = tuple(grouped.get(source.block_id, ()))
    if resolved is None:
        return _render_block(pid, source, None, "", own, shape, "missing-block", captions)
    return _render_block(pid, source, resolved, resolved.text, own, shape, source.kind, captions)


def _from_resolved(
    pid: str,
    pdf_page: int,
    resolved: ResolvedBlock,
    grouped: dict[str, list[SpanMark]],
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
    )


def _render_block(
    pid: str,
    source: Block,
    resolved: ResolvedBlock | None,
    text: str,
    marks: tuple[SpanMark, ...],
    shape: tuple[int, int] | None,
    role: str,
    captions: set[str],
) -> RenderBlock:
    digest = resolved.anchor if resolved is not None else ""
    reading = _primary(source, resolved)
    formula = formula_of(reading)
    table = table_of(reading)
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
        crops=_crops(marks, source.region, shape),
        paragraph=source.paragraph,
        formula=formula,
        table=table,
        emit_caption=not caption_is_separate(table, captions, text),
        formula_crop=formula_needs_crop(source, reading),
        table_image=table_needs_image(source, reading),
        table_evidence=table_needs_evidence(source),
    )


def _primary(source: Block, resolved: ResolvedBlock | None) -> RawReading | None:
    if not source.readings:
        return None
    if resolved is None:
        return source.readings[0]
    for reading in source.readings:
        if reading.engine == resolved.engine and reading.variant == resolved.variant:
            if resolved.model is None or reading.model == resolved.model:
                return reading
    return source.readings[0]


def _caption_texts(ledger: PageBlockLedger, by_id: dict[str, ResolvedBlock]) -> set[str]:
    found: set[str] = set()
    for block in ledger.blocks:
        resolved = by_id.get(block.block_id)
        text = resolved.text if resolved is not None else block.readings[0].text
        found.add(" ".join(text.split()))
    return found


def _crops(
    marks: tuple[SpanMark, ...], region: Region | None, shape: tuple[int, int] | None
) -> tuple[tuple[int, int, str], ...]:
    if region is None or shape is None or padded_box(region, shape, CROP_PADDING_PX) is None:
        return ()
    return tuple(
        (mark.start, mark.end, f"review/{crop_filename(mark.block_id, mark.start, mark.end)}")
        for mark in marks
    )


def _failed(pid: str, pdf_page: int, reason: str, shape: tuple[int, int] | None) -> RenderBlock:
    region = None
    crops: tuple[tuple[int, int, str], ...] = ()
    block_id = ""
    if shape is not None:
        height, width = shape
        block_id = f"{pid}-r0000"
        region = Region(x0=0, y0=0, x1=float(width), y1=float(height), width=width, height=height)
        crops = ((0, 0, f"review/{crop_filename(block_id, 0, 0)}"),)
    return RenderBlock(
        pdf_page=pdf_page,
        page_id=pid,
        ordinal=0,
        reading_order=0,
        kind="page-failed",
        role="page-failed",
        block_id=block_id,
        digest="",
        text="",
        region=region,
        resolved=None,
        failed=True,
        fail_reason=reason,
        crops=crops,
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


def _marks(
    pid: str,
    ledger: PageBlockLedger,
    resolved: PageResolvedText,
    overlay: PageCorrectionOverlay,
    edits: list[CorrectionEdit],
) -> list[SpanMark]:
    blocks = {block.block_id: block for block in ledger.blocks}
    return page_marks(pid, blocks, resolved, overlay, edits)
