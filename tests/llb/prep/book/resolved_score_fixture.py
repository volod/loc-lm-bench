"""Synthetic runs for the resolved-lane score tests. No page of a real book."""

from pathlib import Path

from llb.core.contracts.data_prep.book_blocks import Block, PageBlockLedger, RawReading, Region
from llb.core.contracts.data_prep.book_resolve import PageResolvedText, ResolvedBlock, ResolvedSpan
from llb.core.contracts.data_prep.book_structure import (
    STRUCTURE_KIND,
    STRUCTURE_SCHEMA_ID,
    STRUCTURE_SCHEMA_VERSION,
    DocumentStructure,
    NumberedItem,
    PrintedPage,
    StructureNode,
)
from llb.prep.book.constants import BLOCK_LEDGER_SCHEMA_ID, CONTRACT_VERSION
from llb.prep.book.ledger.io import write_ledger
from llb.prep.book.metrics.reference import RefBlock, RefPage, RefRegion, RefTable
from llb.prep.book.ranges import page_id
from llb.prep.book.resolve.constants import RESOLVED_KIND, RESOLVED_SCHEMA_ID, SCHEMA_VERSION
from llb.prep.book.resolve.io import write_resolved
from llb.prep.book.structure.io import write_structure

PAGE_COUNT = 9
ANCHOR = "a" * 64
SHA = "b" * 64


def _body(pdf: int, text: str, deskew: str | None = None) -> Block:
    readings = [
        RawReading(engine="alpha", variant="original", text=text, model="m"),
        RawReading(engine="beta", variant="original", text=text, model="m"),
    ]
    if deskew is not None:
        readings.append(RawReading(engine="alpha", variant="deskew", text=deskew, model="m"))
    return _placed(pdf, 0, "body", text, _box(10, 20, 80, 40), readings)


def _placed(
    pdf: int,
    ordinal: int,
    kind: str,
    text: str,
    region: Region,
    readings: list[RawReading] | None = None,
) -> Block:
    own = readings or [RawReading(engine="alpha", variant="original", text=text, model="m")]
    return Block(
        block_id=f"p{pdf}-b{ordinal}",
        pdf_page=pdf,
        ordinal=ordinal,
        kind=kind,  # type: ignore[arg-type]
        region=region,
        reading_order=ordinal,
        readings=own,
    )


def _box(x0: float, y0: float, x1: float, y1: float) -> Region:
    return Region(x0=x0, y0=y0, x1=x1, y1=y1, width=200, height=400)


def _adopted(block: Block, text: str, variant: str = "original") -> ResolvedBlock:
    return ResolvedBlock(
        block_id=block.block_id,
        anchor=ANCHOR,
        engine="alpha",
        variant=variant,
        fallback=False,
        primary_text=text,
        text=text,
        spans=[ResolvedSpan(provenance="primary", text=text, start=0, end=len(text), adopted=True)],
    )


def _withheld(block: Block) -> ResolvedBlock:
    primary = "catX"
    return ResolvedBlock(
        block_id=block.block_id,
        anchor=ANCHOR,
        engine="alpha",
        variant="original",
        fallback=False,
        primary_text=primary,
        text="cat",
        spans=[
            ResolvedSpan(provenance="primary", text="cat", start=0, end=3, adopted=True),
            ResolvedSpan(
                provenance="unresolved",
                text="X",
                start=3,
                end=4,
                adopted=False,
                classes=["symbol"],
            ),
        ],
    )


def _save(
    run: Path, blocks: list[Block], resolved: list[ResolvedBlock] | ResolvedBlock | None
) -> None:
    pdf = blocks[0].pdf_page
    base = run / "pages" / page_id(pdf, PAGE_COUNT)
    write_ledger(
        base / "blocks.json",
        PageBlockLedger(
            schema_id=BLOCK_LEDGER_SCHEMA_ID,
            schema_version=CONTRACT_VERSION,
            kind="book-ocr-block-ledger",
            page_id=page_id(pdf, PAGE_COUNT),
            pdf_page=pdf,
            blocks=blocks,
        ),
    )
    if resolved is None:
        return
    rows = resolved if isinstance(resolved, list) else [resolved]
    write_resolved(
        base / "resolved.json",
        PageResolvedText(
            schema_id=RESOLVED_SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
            kind=RESOLVED_KIND,
            page_id=page_id(pdf, PAGE_COUNT),
            pdf_page=pdf,
            page_class="ignored",
            primary_stage="ocr",
            primary_engine="alpha",
            primary_variant=rows[0].variant,
            blocks=rows,
        ),
    )


def _mapping(run: Path, *pages: tuple[int, str], items: list[NumberedItem] | None = None) -> None:
    rows = [
        PrintedPage(
            pdf_page=pdf,
            page_id=page_id(pdf, PAGE_COUNT),
            printed=printed,
            scheme="arabic",
            status="observed",
            block_id=None,
            fallback=False,
        )
        for pdf, printed in pages
    ]
    root = StructureNode(
        node_id="book",
        level="book",
        pdf_start=pages[0][0],
        pdf_end=pages[-1][0],
        evidence=["root"],
        fallback=False,
    )
    write_structure(
        run / "metadata" / "document_structure.json",
        DocumentStructure(
            schema_id=STRUCTURE_SCHEMA_ID,
            schema_version=STRUCTURE_SCHEMA_VERSION,
            kind=STRUCTURE_KIND,
            source_sha256=SHA,
            pages=rows,
            hierarchy=root,
            inventory=items or [],
        ),
    )


def _ref(
    pdf: int,
    page_class: str,
    printed: str,
    rows: list[tuple[str, ...]],
    boxes: list[tuple[float, float, float, float]] | None = None,
    equation_number: str | None = None,
    table_rows: list[list[str]] | None = None,
) -> RefPage:
    blocks: list[RefBlock] = []
    for order, row in enumerate(rows):
        kind = row[1]
        extra: dict[str, object] = {}
        if kind == "figure" and len(row) > 3:
            extra["caption_id"] = row[3]
        if kind == "formula":
            extra["latex"] = row[2]
            if equation_number is not None:
                extra["equation_number"] = equation_number
        if kind == "table":
            extra["table"] = RefTable.model_validate({"rows": table_rows or [["a"]]})
        box = boxes[order] if boxes is not None else (10, 20 + order * 30, 80, 40 + order * 30)
        blocks.append(
            RefBlock(
                block_id=row[0],
                kind=kind,  # type: ignore[arg-type]
                region=RefRegion(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
                reading_order=order,
                text=row[2],
                **extra,  # type: ignore[arg-type]
            )
        )
    return RefPage(
        pdf_page=pdf,
        page_class=page_class,
        printed_page=printed,
        width=200,
        height=400,
        blocks=blocks,
    )


def _write_reference(directory: Path, pages: list[RefPage]) -> None:
    for page in pages:
        path = directory / "pages" / f"p{page.pdf_page:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page.model_dump_json(indent=2) + "\n", encoding="utf-8")
