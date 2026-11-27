"""Figure pieces, grouped drawings, and the vault files they become."""

from dataclasses import dataclass

from llb.prep.book.assets.boxes import Box, from_region, unite
from llb.prep.book.render.publish import Asset
from llb.prep.book.render.types import RenderBlock


@dataclass
class Piece:
    """One ledger region that a detector called a figure."""

    block: RenderBlock
    chapter: str
    number: str | None = None
    caption_id: str | None = None


@dataclass
class FigureGroup:
    """One drawing. ``pieces`` is in reading order. The first is where it is embedded."""

    pieces: list[Piece]
    number: str | None
    caption_id: str | None

    @property
    def primary(self) -> RenderBlock:
        """The region the asset is named and embedded from."""
        return self.pieces[0].block

    @property
    def union(self) -> Box:
        """Every source region, in the original render."""
        boxes: list[Box] = []
        for piece in self.pieces:
            region = piece.block.region
            if region is not None:
                boxes.append(from_region(region))
        if not boxes:
            raise ValueError(f"{self.primary.block_id} has no region")
        return unite(boxes)


def piece_key(piece: Piece) -> tuple[int, int, int, str]:
    """Page, then reading order, then ordinal, then id."""
    block = piece.block
    return (block.pdf_page, block.reading_order, block.ordinal, block.block_id)


@dataclass(frozen=True)
class FigureRender:
    """What the vault writer needs: embeds, crops, and the region record."""

    paths: dict[str, str]
    blobs: dict[str, Asset]
    captions: dict[str, str]
    suppressed: frozenset[str]
    ornaments: frozenset[str]
    record: bytes
