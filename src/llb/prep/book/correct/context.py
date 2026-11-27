"""Bounded text around one block: one neighbour each side, and the facing page edge."""

from collections.abc import Sequence
from dataclasses import dataclass

from llb.core.contracts.data_prep.book_blocks import Block
from llb.prep.book.correct.constants import NEIGHBOUR_CHARS


@dataclass(frozen=True, slots=True)
class Neighbours:
    """Text the proposer may see besides the block and its crop. Each field is clipped."""

    before: str
    after: str
    previous_page: str
    next_page: str


def neighbour_window(
    blocks: Sequence[Block],
    index: int,
    texts: Sequence[str],
    previous_page: str,
    next_page: str,
) -> Neighbours:
    """The blocks immediately before and after `index`, in reading order.

    `texts[i]` is the primary reading of `blocks[i]`. The previous page's edge is included
    only for the first block, and the next page's edge only for the last.
    """
    before = texts[index - 1] if index > 0 else ""
    after = texts[index + 1] if index + 1 < len(texts) else ""
    return Neighbours(
        before=_clip(before),
        after=_clip(after),
        previous_page=_clip(previous_page) if index == 0 else "",
        next_page=_clip(next_page) if index == len(blocks) - 1 else "",
    )


def reading_order(blocks: Sequence[Block]) -> list[Block]:
    """Blocks in the order a reader meets them."""
    return sorted(blocks, key=lambda block: (block.reading_order, block.ordinal, block.block_id))


def _clip(text: str) -> str:
    if len(text) <= NEIGHBOUR_CHARS:
        return text
    return text[:NEIGHBOUR_CHARS]
