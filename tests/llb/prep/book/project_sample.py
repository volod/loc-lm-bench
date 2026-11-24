"""Synthetic three-page book run for corpus-projection tests."""

import hashlib
from pathlib import Path

from llb.core.contracts.data_prep.book import BookOcrRunConfig, BookSourceRecord, BookStageSpec
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
from llb.prep.book.config import write_config
from llb.prep.book.constants import BLOCK_LEDGER_SCHEMA_ID, CONTRACT_VERSION, RUN_CONFIG_SCHEMA_ID
from llb.prep.book.ledger.io import write_ledger
from llb.prep.book.ranges import page_id
from llb.prep.book.resolve.constants import RESOLVED_KIND, RESOLVED_SCHEMA_ID, SCHEMA_VERSION
from llb.prep.book.resolve.io import write_resolved
from llb.prep.book.structure.io import write_structure

_W = 800
_H = 1200
_CHAPTER = ("The vehicle delta-v budget is 3200 meters per second. " * 12).strip()
_SECTION = "TOK-BODY-SEC " + ("The section states the adopted reading of the scan. " * 12)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _region(y: float) -> Region:
    return Region(x0=80, y0=y, x1=520, y1=y + 28, width=_W, height=_H)


def _block(pdf: int, ordinal: int, kind: str, text: str) -> Block:
    return Block(
        block_id=f"p{pdf:04d}-b{ordinal:04d}",
        pdf_page=pdf,
        ordinal=ordinal,
        kind=kind,  # type: ignore[arg-type]
        region=_region(40 + ordinal * 40),
        reading_order=ordinal,
        readings=[
            RawReading(
                engine="rapidocr", model=None, variant="original", text=text, confidence=0.99
            )
        ],
    )


def _adopted(block: Block, text: str | None = None) -> ResolvedBlock:
    body = block.readings[0].text if text is None else text
    return ResolvedBlock(
        block_id=block.block_id,
        anchor="a" * 64,
        engine="rapidocr",
        variant="original",
        fallback=False,
        primary_text=body,
        text=body,
        spans=[ResolvedSpan(provenance="primary", text=body, start=0, end=len(body), adopted=True)],
    )


def _unresolved(block: Block) -> ResolvedBlock:
    primary = "ALPHA 0 BETA"
    start = primary.index("0")
    spans = [
        ResolvedSpan(provenance="primary", text="ALPHA ", start=0, end=start, adopted=True),
        ResolvedSpan(
            provenance="unresolved",
            text="0",
            start=start,
            end=start + 1,
            adopted=False,
            classes=["digit"],
        ),
        ResolvedSpan(
            provenance="primary", text=" BETA", start=start + 1, end=len(primary), adopted=True
        ),
    ]
    return ResolvedBlock(
        block_id=block.block_id,
        anchor="b" * 64,
        engine="rapidocr",
        variant="original",
        fallback=False,
        primary_text=primary,
        text="ALPHA  BETA",
        spans=spans,
    )


def _save(run: Path, blocks: list[Block], resolved: list[ResolvedBlock] | None) -> None:
    pdf = blocks[0].pdf_page
    pid = page_id(pdf, 3)
    base = run / "pages" / pid
    write_ledger(
        base / "blocks.json",
        PageBlockLedger(
            schema_id=BLOCK_LEDGER_SCHEMA_ID,
            schema_version=CONTRACT_VERSION,
            kind="book-ocr-block-ledger",
            page_id=pid,
            pdf_page=pdf,
            blocks=blocks,
        ),
    )
    if resolved is None:
        return
    write_resolved(
        base / "resolved.json",
        PageResolvedText(
            schema_id=RESOLVED_SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
            kind=RESOLVED_KIND,
            page_id=pid,
            pdf_page=pdf,
            page_class="",
            primary_stage="ocr",
            primary_engine="rapidocr",
            primary_variant="original",
            blocks=resolved,
        ),
    )


def _node(
    node_id: str, level: str, number: str, title: str, start: int, end: int, block: str | None
) -> StructureNode:
    return StructureNode(
        node_id=node_id,
        level=level,  # type: ignore[arg-type]
        number=number,
        title=title,
        pdf_start=start,
        pdf_end=end,
        printed_start=str(start),
        printed_end=str(end),
        block_id=block,
        evidence=["numbered-heading"] if block else ["root"],
        fallback=False,
        children=[],
    )


def sample_run(tmp: Path) -> Path:
    run = tmp / "run"
    pdf = run / "source" / "book.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4\n%project\n")
    write_config(
        run,
        BookOcrRunConfig(
            schema_id=RUN_CONFIG_SCHEMA_ID,
            schema_version=CONTRACT_VERSION,
            kind="book-ocr-run-config",
            run_id="project-test",
            created_at="2026-01-01T00:00:00+00:00",
            source=BookSourceRecord(
                original_path=str(pdf.resolve()),
                sha256=_sha(pdf),
                size_bytes=pdf.stat().st_size,
                page_count=3,
                policy="copy",
                stored_path="source/book.pdf",
            ),
            pages_spec="1-",
            pages=[1, 2, 3],
            batch_pages=25,
            max_attempts=2,
            stages=[BookStageSpec(name="render")],
        ),
    )
    page1 = [
        _block(1, 0, "page-number", "PN-PAGE-ONE"),
        _block(1, 1, "running-header", "RUNNING-HEADER"),
        _block(1, 2, "heading", "TOK-HEAD-CH"),
        _block(1, 3, "body", _CHAPTER),
        _block(1, 4, "formula", "TOK-FORMULA"),
        _block(1, 5, "table", "| a | b |\n| --- | --- |\n| 1 | 2 |"),
        _block(1, 6, "figure", "TOK-FIGURE"),
        _block(1, 7, "caption", "TOK-CAPTION"),
    ]
    raw = _block(2, 1, "body", "ALPHA 0 BETA")
    page2 = [_block(2, 0, "heading", "TOK-HEAD-SEC"), raw, _block(2, 2, "body", _SECTION)]
    _save(run, page1, [_adopted(item) for item in page1])
    _save(run, page2, [_adopted(page2[0]), _unresolved(raw), _adopted(page2[2])])
    _save(run, [_block(3, 0, "body", "RAW-FAILED-PAGE")], None)
    section = _node("book/chapter-1/section-1-1", "section", "1.1", "Purpose", 2, 3, "p0002-b0000")
    chapter = _node("book/chapter-1", "chapter", "1", "Introduction", 1, 3, "p0001-b0002")
    chapter = chapter.model_copy(update={"children": [section]})
    root = _node("book", "book", "", "Book", 1, 3, None)
    root = root.model_copy(update={"children": [chapter]})
    write_structure(
        run / "metadata" / "document_structure.json",
        DocumentStructure(
            schema_id=STRUCTURE_SCHEMA_ID,
            schema_version=STRUCTURE_SCHEMA_VERSION,
            kind=STRUCTURE_KIND,
            source_sha256=_sha(pdf),
            pages=[
                PrintedPage(
                    pdf_page=1,
                    page_id="p0001",
                    printed="1",
                    scheme="arabic",
                    status="observed",
                    block_id="p0001-b0000",
                    fallback=False,
                ),
                PrintedPage(
                    pdf_page=2,
                    page_id="p0002",
                    printed="2",
                    scheme="arabic",
                    status="observed",
                    block_id=None,
                    fallback=False,
                ),
                PrintedPage(
                    pdf_page=3,
                    page_id="p0003",
                    printed="3",
                    scheme="arabic",
                    status="observed",
                    block_id=None,
                    fallback=False,
                ),
            ],
            hierarchy=root,
            inventory=[
                NumberedItem(
                    kind="equation",
                    number="1-1",
                    pdf_page=1,
                    printed="1",
                    block_id="p0001-b0004",
                    evidence="caption",
                )
            ],
        ),
    )
    return run
