"""The vault's figure record: one row per asset, and the regions that were refused."""

import json

from llb.prep.book.assets.types import FigureGroup, Piece
from llb.prep.book.render.types import RenderBlock


def figure_record(figures: list[dict[str, object]], rejected: list[dict[str, object]]) -> bytes:
    """ASCII JSON. List order is reading order; keys are sorted."""
    payload = {"figures": figures, "rejected": rejected}
    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    return text.encode("ascii")


def figure_row(
    group: FigureGroup, asset: str, variant: str, crop: tuple[int, int, int, int]
) -> dict[str, object]:
    """One asset: its name, the variant the pixels came from, and every source region."""
    x0, y0, x1, y1 = crop
    return {
        "asset": asset,
        "block_id": group.primary.block_id,
        "caption_block_id": group.caption_id,
        "crop": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
        "kind": "figure",
        "number": group.number,
        "page_id": group.primary.page_id,
        "pdf_page": group.primary.pdf_page,
        "regions": [region_row(piece) for piece in group.pieces],
        "variant": variant,
    }


def region_row(piece: Piece) -> dict[str, object]:
    """One source region in the original render."""
    region = piece.block.region
    if region is None:
        raise ValueError(f"{piece.block.block_id} has no region")
    return {
        "block_id": piece.block.block_id,
        "height": region.height,
        "ordinal": piece.block.ordinal,
        "reading_order": piece.block.reading_order,
        "width": region.width,
        "x0": region.x0,
        "x1": region.x1,
        "y0": region.y0,
        "y1": region.y1,
    }


def rejected_row(block: RenderBlock, reason: str) -> dict[str, object]:
    """A figure-labelled region that did not become an asset."""
    return {
        "block_id": block.block_id,
        "page_id": block.page_id,
        "pdf_page": block.pdf_page,
        "reason": reason,
    }
