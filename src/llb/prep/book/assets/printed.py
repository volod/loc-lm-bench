"""The figure number a page prints, and which drawing it belongs to.

A unique caption-evidence inventory row is that number. So is one caption
that itself prints the number, including when the inventory called the line a
footer. Two captions that print the same number are ambiguous, and a footer
with no caption is not a number.
"""

from collections.abc import Sequence

from llb.core.contracts.data_prep.book_structure import NumberedItem
from llb.prep.book.assets.boxes import edge_gap, from_region
from llb.prep.book.assets.constants import CAPTION_GAP_MAX_PX
from llb.prep.book.assets.types import Piece
from llb.prep.book.render.catalog import canonical
from llb.prep.book.render.types import RenderBlock, numbered_text
from llb.prep.book.structure.inventory import caption_figure_number


def assign_printed(
    pieces: list[Piece], captions: list[RenderBlock], inventory: Sequence[NumberedItem]
) -> None:
    """Give each piece the unique figure number the page prints for it."""
    piece_pages = _pieces_by_page(pieces)
    caption_pages = _blocks_by_page(captions)
    for page in sorted(set(piece_pages) | set(caption_pages)):
        _assign_page(piece_pages.get(page, []), caption_pages.get(page, []), inventory, page)


def _assign_page(
    pieces: list[Piece],
    captions: list[RenderBlock],
    inventory: Sequence[NumberedItem],
    page: int,
) -> None:
    reliable = _reliable(inventory, page)
    _add_printed(reliable, captions, page)
    _assign_owned(pieces, captions, reliable)
    _assign_nearest(pieces, captions, reliable)


def _reliable(inventory: Sequence[NumberedItem], page: int) -> dict[str, NumberedItem]:
    grouped: dict[str, list[NumberedItem]] = {}
    for item in inventory:
        if item.kind != "figure" or item.pdf_page != page or item.evidence != "caption":
            continue
        grouped.setdefault(canonical(item.number), []).append(item)
    return {key: rows[0] for key, rows in grouped.items() if len(rows) == 1}


def _add_printed(reliable: dict[str, NumberedItem], captions: list[RenderBlock], page: int) -> None:
    """A caption that prints a number is that number, even when the inventory said footer."""
    for key, block in _unique_printed(captions).items():
        if key in reliable:
            continue
        number = caption_figure_number(numbered_text(block))
        if not number:
            continue
        reliable[key] = NumberedItem(
            kind="figure",
            number=number,
            pdf_page=page,
            block_id=block.block_id,
            evidence="caption",
        )


def _unique_printed(captions: list[RenderBlock]) -> dict[str, RenderBlock]:
    grouped: dict[str, list[RenderBlock]] = {}
    for block in captions:
        number = caption_figure_number(numbered_text(block))
        if number:
            grouped.setdefault(canonical(number), []).append(block)
    return {key: rows[0] for key, rows in grouped.items() if len(rows) == 1}


def _assign_owned(
    pieces: list[Piece], captions: list[RenderBlock], reliable: dict[str, NumberedItem]
) -> None:
    by_id = {piece.block.block_id: piece for piece in pieces}
    for item in reliable.values():
        piece = by_id.get(item.block_id)
        if piece is None or piece.number:
            continue
        piece.number = item.number
        caption = _printing(captions, item)
        if caption is not None:
            piece.caption_id = caption.block_id


def _assign_nearest(
    pieces: list[Piece], captions: list[RenderBlock], reliable: dict[str, NumberedItem]
) -> None:
    used_figures = {piece.block.block_id for piece in pieces if piece.number}
    used_captions = {piece.caption_id for piece in pieces if piece.caption_id}
    claimed = {canonical(piece.number) for piece in pieces if piece.number}
    ranked = _nearest_pairs(pieces, captions, reliable, used_figures, used_captions, claimed)
    for _gap, caption_id, figure_id, number in ranked:
        if figure_id in used_figures or caption_id in used_captions:
            continue
        piece = next(item for item in pieces if item.block.block_id == figure_id)
        piece.number = number
        piece.caption_id = caption_id
        used_figures.add(figure_id)
        used_captions.add(caption_id)


def _nearest_pairs(
    pieces: list[Piece],
    captions: list[RenderBlock],
    reliable: dict[str, NumberedItem],
    used_figures: set[str],
    used_captions: set[str],
    claimed: set[str],
) -> list[tuple[float, str, str, str]]:
    pairs: list[tuple[float, str, str, str]] = []
    for item in reliable.values():
        if canonical(item.number) in claimed:
            continue
        caption = _printing(captions, item)
        if caption is None or caption.block_id in used_captions:
            continue
        target = _closest(pieces, caption, used_figures)
        if target is None or target.block.region is None or caption.region is None:
            continue
        gap = edge_gap(from_region(target.block.region), from_region(caption.region))
        pairs.append((gap, caption.block_id, target.block.block_id, item.number))
    pairs.sort()
    return pairs


def _closest(pieces: list[Piece], caption: RenderBlock, used: set[str]) -> Piece | None:
    if caption.region is None:
        return None
    free = [
        piece
        for piece in pieces
        if piece.block.block_id not in used and not piece.number and piece.block.region is not None
    ]
    if not free:
        return None
    return min(free, key=lambda piece: _caption_key(piece, caption))


def _caption_key(piece: Piece, caption: RenderBlock) -> tuple[float, str]:
    piece_region = piece.block.region
    caption_region = caption.region
    if piece_region is None or caption_region is None:
        return (CAPTION_GAP_MAX_PX + 1.0, piece.block.block_id)
    gap = edge_gap(from_region(piece_region), from_region(caption_region))
    return (gap, piece.block.block_id)


def _printing(captions: list[RenderBlock], item: NumberedItem) -> RenderBlock | None:
    by_id = {block.block_id: block for block in captions}
    owned = by_id.get(item.block_id)
    if owned is not None:
        return owned
    key = canonical(item.number)
    matches = [block for block in captions if _same_number(numbered_text(block), key)]
    if not matches:
        return None
    return min(matches, key=lambda block: (block.reading_order, block.ordinal, block.block_id))


def _same_number(text: str, key: str) -> bool:
    number = caption_figure_number(text)
    return number is not None and canonical(number) == key


def _pieces_by_page(pieces: list[Piece]) -> dict[int, list[Piece]]:
    found: dict[int, list[Piece]] = {}
    for piece in pieces:
        found.setdefault(piece.block.pdf_page, []).append(piece)
    return found


def _blocks_by_page(blocks: list[RenderBlock]) -> dict[int, list[RenderBlock]]:
    found: dict[int, list[RenderBlock]] = {}
    for block in blocks:
        found.setdefault(block.pdf_page, []).append(block)
    return found
