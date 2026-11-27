"""The blocks and files passed between the vault renderer's steps."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


from llb.core.contracts.data_prep.book import BookOcrRunConfig
from llb.core.contracts.data_prep.book_blocks import FormulaReading, Region, TableReading
from llb.core.contracts.data_prep.book_resolve import ResolvedBlock
from llb.prep.book.resolve.numbers import number_text
from llb.core.contracts.data_prep.book_structure import DocumentStructure, StructureNode
from llb.prep.book.review.marks import SpanMark


@dataclass(frozen=True, slots=True)
class RenderBlock:
    """One ledger block in reading order, already resolved, or one failed page."""

    pdf_page: int
    page_id: str
    ordinal: int
    reading_order: int
    kind: str
    role: str
    block_id: str
    digest: str
    text: str
    region: Region | None
    resolved: ResolvedBlock | None
    marks: tuple[SpanMark, ...] = ()
    crops: tuple[tuple[int, int, str], ...] = ()
    failed: bool = False
    fail_reason: str = ""
    # Copied from the ledger. A finished region does not absorb the next body block.
    paragraph: bool = False
    formula: FormulaReading | None = None
    table: TableReading | None = None
    # The table caption this block prints from its structure; empty when another block prints it.
    caption: str = ""
    # The vault writes a crop when the formula is below the validity or agreement bar.
    formula_crop: bool = False
    # An unfaithful table keeps the page image. A disputed cell keeps a review crop.
    table_image: bool = False
    table_evidence: bool = False


def numbered_text(block: RenderBlock) -> str:
    """The text a printed number is read from: no number survives an unresolved span."""
    if block.resolved is None:
        return block.text
    return number_text(block.resolved)


@dataclass(frozen=True, slots=True)
class Unit:
    """One unsplittable piece of a section: a paragraph, a formula, a pair."""

    role: str
    blocks: tuple[RenderBlock, ...]
    text: str
    splittable: bool
    keep_anchors: bool = True
    text_range: tuple[int, int] | None = None


@dataclass
class VaultFile:
    """One content note before its navigation lines are wrapped around it."""

    stem: str
    node: StructureNode
    units: list[Unit]
    pdf_start: int
    pdf_end: int
    depth: int
    body: str = ""
    review_count: int = 0
    # False for a later span of a node, the prose that sits between its children.
    # The contents lists the opening span, including its length-split parts.
    opening: bool = False
    page_anchors: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RenderContext:
    """Lookups the Markdown writer needs. Built once for a vault."""

    book: str
    fallback: bool
    anchors: dict[str, str]
    equations: dict[str, str]
    assets: dict[str, str]
    levels: dict[str, int]
    content_href: Callable[[int], str]
    toc_href: Callable[[int], str]
    pages_with_images: frozenset[int] = field(default_factory=frozenset)
    formula_crops: dict[str, str] = field(default_factory=dict)
    table_images: dict[str, str] = field(default_factory=dict)
    table_evidence: dict[str, str] = field(default_factory=dict)
    # The printed caption block of a figure asset, and blocks already shown inside it.
    figure_captions: dict[str, RenderBlock] = field(default_factory=dict)
    suppressed: frozenset[str] = field(default_factory=frozenset)


@dataclass
class BookLoad:
    """What `render` reads from one run. Keep page-image paths so renders can stream assets."""

    config: BookOcrRunConfig
    structure: DocumentStructure
    pdf: Path
    blocks: list[RenderBlock]
    images: dict[int, Path]
