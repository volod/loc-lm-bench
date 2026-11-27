"""Proximity captions, after the printed number has been assigned.

An unnumbered caption can belong to a drawing when it is clearly the nearest
one. A caption that prints a figure, table, or equation number is not a guess.
Two captions at a similar distance attach to neither drawing.
"""

from llb.prep.book.assets.boxes import edge_gap, from_region, span_limit
from llb.prep.book.assets.constants import (
    CAPTION_AMBIGUOUS_RATIO,
    CAPTION_GAP_FRAC,
    CAPTION_GAP_MAX_PX,
    CAPTION_GAP_MIN_PX,
)
from llb.prep.book.assets.types import FigureGroup
from llb.prep.book.render.types import RenderBlock
from llb.prep.book.structure.inventory import (
    caption_equation_number,
    caption_figure_number,
    caption_table_number,
)


def assign_proximity(groups: list[FigureGroup], captions: list[RenderBlock]) -> None:
    """Attach one unnumbered caption when it is clearly the nearest."""
    used = {group.caption_id for group in groups if group.caption_id}
    for group in groups:
        if group.caption_id:
            continue
        chosen = _nearest_plain(group, captions, used)
        if chosen is None:
            continue
        group.caption_id = chosen
        used.add(chosen)


def caption_in_crop(group: FigureGroup, captions: list[RenderBlock]) -> RenderBlock | None:
    """The associated caption when it sits close enough to join the image."""
    block = _block(group.caption_id, captions)
    if block is None or block.region is None or group.primary.region is None:
        return None
    if block.pdf_page != group.primary.pdf_page:
        return None
    limit = _caption_limit(group)
    gap = edge_gap(group.union, from_region(block.region))
    if gap > limit:
        return None
    return block


def _nearest_plain(group: FigureGroup, captions: list[RenderBlock], used: set[str]) -> str | None:
    limit = _caption_limit(group)
    choices: list[tuple[float, str]] = []
    for block in captions:
        if block.pdf_page != group.primary.pdf_page or block.block_id in used:
            continue
        if block.region is None or _numbered_caption(block.text):
            continue
        gap = edge_gap(group.union, from_region(block.region))
        if gap <= limit:
            choices.append((gap, block.block_id))
    return _choose(choices)


def _numbered_caption(text: str) -> bool:
    return (
        caption_figure_number(text) is not None
        or caption_table_number(text) is not None
        or caption_equation_number(text) is not None
    )


def _choose(choices: list[tuple[float, str]]) -> str | None:
    if not choices:
        return None
    choices.sort()
    if len(choices) > 1 and choices[1][0] <= choices[0][0] * CAPTION_AMBIGUOUS_RATIO:
        return None
    return choices[0][1]


def _caption_limit(group: FigureGroup) -> float:
    region = group.primary.region
    if region is None:
        return CAPTION_GAP_MIN_PX
    return span_limit(
        region.width, region.height, CAPTION_GAP_FRAC, CAPTION_GAP_MIN_PX, CAPTION_GAP_MAX_PX
    )


def _block(block_id: str | None, captions: list[RenderBlock]) -> RenderBlock | None:
    if not block_id:
        return None
    for block in captions:
        if block.block_id == block_id:
            return block
    return None
