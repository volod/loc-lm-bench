"""Hypothesis pages: one engine of a block ledger, or one existing-lane citation sidecar.

A ledger page keeps its regions, and a table block keeps a grid when its text parses as one.
Caption links are the nearest caption to each figure. A citation sidecar is one text block per
PDF page with no region, no grid, and no figure box: those metrics are reported absent.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from llb.core.contracts.data_prep.book_blocks import PageBlockLedger, RawReading, Region
from llb.core.contracts.data_prep.corpus import PdfCitations
from llb.prep.book.constants import BLOCKS_FILENAME, PAGES_DIRNAME
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.metrics.confusion import equation_numbers
from llb.prep.book.metrics.constants import LANE_BOOK, LANE_EXISTING, ORIGINAL_VARIANT
from llb.prep.book.metrics.geometry import Box, nearest_links
from llb.prep.book.metrics.latex import looks_like_latex
from llb.prep.book.metrics.tables import Cell, parse_table_text

Table = tuple[tuple[Cell, ...], ...]


@dataclass(frozen=True)
class HypBlock:
    """One block of a submission. ``region`` is None when the lane has no boxes."""

    block_id: str
    kind: str
    region: Box | None
    reading_order: int
    text: str
    latex: str | None
    table: Table | None
    caption_id: str | None
    equation_number: str | None


@dataclass(frozen=True)
class PageSubmission:
    """One lane's reading of one PDF page.

    ``unsupported`` names structural metrics this page did not emit. ``uses_emitted_folio``
    is set when the submission carries the printed-page mapping the renderers publish;
    ``emitted_folio`` is that string, or None for an unnumbered page. ``abstentions`` counts
    unresolved spans per uncertainty class.
    """

    lane: str
    pdf_page: int
    blocks: tuple[HypBlock, ...]
    text_only: bool
    present: bool = True
    unsupported: frozenset[str] = frozenset()
    uses_emitted_folio: bool = False
    emitted_folio: str | None = None
    abstentions: tuple[tuple[str, int], ...] = ()


def load_run(run_dir: Path, engine: str | None) -> list[PageSubmission]:
    """One submission per ledger page per engine. ``engine`` selects a single reading."""
    paths = sorted((run_dir / PAGES_DIRNAME).glob(f"*/{BLOCKS_FILENAME}"))
    if not paths:
        raise ValueError(f"{run_dir} has no {PAGES_DIRNAME}/*/{BLOCKS_FILENAME}")
    ledgers = [read_ledger(path) for path in paths]
    names = sorted(
        {
            reading.engine
            for ledger in ledgers
            for block in ledger.blocks
            for reading in block.readings
        }
    )
    if engine is not None and engine not in names:
        raise ValueError(f"{run_dir} has no reading from engine {engine}")
    selected = [engine] if engine is not None else names
    return [_from_ledger(ledger, name) for name in selected for ledger in ledgers]


def load_citations(path: Path) -> list[PageSubmission]:
    """The existing PDF lane: page text sliced from the citation sidecar's markdown."""
    try:
        record = PdfCitations.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"{path} is not a pdf citations sidecar: {exc}") from exc
    markdown_path = path.parent / record.doc_id
    if not markdown_path.is_file():
        raise ValueError(f"{path} cites {record.doc_id}, which is not beside the sidecar")
    markdown = markdown_path.read_text(encoding="utf-8")
    pages: list[PageSubmission] = []
    for page in record.pages:
        if page.text_end > len(markdown) or page.text_start > page.text_end:
            raise ValueError(f"{path} page {page.page} offsets fall outside {record.doc_id}")
        body = markdown[page.text_start : page.text_end]
        block = HypBlock(
            block_id=f"citation-{page.page}",
            kind="body",
            region=None,
            reading_order=0,
            text=body,
            latex=None,
            table=None,
            caption_id=None,
            equation_number=None,
        )
        pages.append(PageSubmission(LANE_EXISTING, page.page, (block,), True))
    return pages


def _from_ledger(ledger: PageBlockLedger, engine: str) -> PageSubmission:
    blocks: list[HypBlock] = []
    for block in ledger.blocks:
        reading = _pick(block.readings, engine)
        if reading is None:
            continue
        text = reading.text
        numbers = equation_numbers(text) if block.kind == "formula" else []
        latex = text if block.kind == "formula" and looks_like_latex(text) else None
        table = parse_table_text(text) if block.kind == "table" else None
        blocks.append(
            HypBlock(
                block_id=block.block_id,
                kind=block.kind,
                region=_box(block.region),
                reading_order=block.reading_order,
                text=text,
                latex=latex,
                table=table,
                caption_id=None,
                equation_number=numbers[0] if numbers else None,
            )
        )
    return PageSubmission(
        f"{LANE_BOOK}/{engine}", ledger.pdf_page, tuple(_with_captions(blocks)), False
    )


def _pick(readings: list[RawReading], engine: str) -> RawReading | None:
    own = [reading for reading in readings if reading.engine == engine]
    if not own:
        return None
    original = [reading for reading in own if reading.variant == ORIGINAL_VARIANT]
    pool = original or own
    return sorted(pool, key=lambda reading: reading.variant)[0]


def _box(region: Region) -> Box:
    return Box(region.x0, region.y0, region.x1, region.y1)


def _with_captions(blocks: list[HypBlock]) -> list[HypBlock]:
    figures = [
        (block.block_id, block.region)
        for block in blocks
        if block.kind == "figure" and block.region is not None
    ]
    captions = [
        (block.block_id, block.region)
        for block in blocks
        if block.kind == "caption" and block.region is not None
    ]
    links = nearest_links(figures, captions)
    linked: list[HypBlock] = []
    for block in blocks:
        caption_id = links.get(block.block_id) if block.kind == "figure" else None
        linked.append(
            HypBlock(
                block.block_id,
                block.kind,
                block.region,
                block.reading_order,
                block.text,
                block.latex,
                block.table,
                caption_id,
                block.equation_number,
            )
        )
    return linked
