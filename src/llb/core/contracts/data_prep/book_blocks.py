"""The page block ledger: one scanned page's blocks, raw readings, and disagreements.

Later stages read this file and nothing upstream of it. A block that cannot name its PDF page and
the region it was read from is not a block -- both are required, so a text with nowhere to point
fails validation instead of entering a rendering.
"""

from typing import Literal

from pydantic import Field, model_validator

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.data_prep.corpus import DataPrepRow

BlockKind = Literal[
    "body",
    "heading",
    "formula",
    "table",
    "figure",
    "caption",
    "footnote",
    "running-header",
    "page-number",
    "ornament",
    "neighbour-page-bleed",
    "unknown",
]
BLOCK_KINDS: tuple[BlockKind, ...] = (
    "body",
    "heading",
    "formula",
    "table",
    "figure",
    "caption",
    "footnote",
    "running-header",
    "page-number",
    "ornament",
    "neighbour-page-bleed",
    "unknown",
)
UncertaintyClass = Literal["digit", "sign", "decimal", "symbol"]
UNCERTAINTY_CLASSES: tuple[UncertaintyClass, ...] = ("digit", "sign", "decimal", "symbol")


class Region(DataPrepRow):
    """An axis-aligned box on the rendered page image, origin at the top left, in pixels."""

    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float
    y1: float
    width: int = Field(ge=1)
    height: int = Field(ge=1)

    @model_validator(mode="after")
    def _box_inside_page(self) -> "Region":
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("region must have positive area")
        if self.x1 > self.width or self.y1 > self.height:
            raise ValueError("region extends outside the page image")
        return self


class RawReading(DataPrepRow):
    """One engine's text for one image variant. Raw: nothing here has been chosen or corrected."""

    engine: str = Field(min_length=1)
    model: str | None = None
    variant: str = Field(min_length=1)
    text: str
    confidence: float | None = None


class UncertaintySlice(DataPrepRow):
    """The characters one reading has across a disagreement, as offsets into that reading."""

    engine: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _offsets_match_text(self) -> "UncertaintySlice":
        if self.end < self.start:
            raise ValueError("uncertainty slice end is before its start")
        if len(self.text) != self.end - self.start:
            raise ValueError("uncertainty slice text length does not match its offsets")
        return self


class UncertaintySpan(DataPrepRow):
    """A character-aligned disagreement. It records every reading's slice and picks none of them."""

    classes: list[UncertaintyClass] = Field(min_length=1)
    slices: list[UncertaintySlice] = Field(min_length=2)


class Block(DataPrepRow):
    """One recovered region: its kind, where it sits, and every raw reading of it."""

    block_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    ordinal: int = Field(ge=0)
    kind: BlockKind
    region: Region
    reading_order: int = Field(ge=0)
    readings: list[RawReading] = Field(min_length=1)
    uncertainty: list[UncertaintySpan] = Field(default_factory=list)


class PageBlockLedger(ArtifactContract):
    """`pages/<pNNNN>/blocks.json`: the block ledger of one PDF page."""

    schema_id: Literal["llb.book-ocr-block-ledger"]
    schema_version: Literal["1.0.0"]
    kind: Literal["book-ocr-block-ledger"]
    page_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)
    blocks: list[Block] = Field(default_factory=list)
