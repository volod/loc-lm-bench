"""Which body lines are one paragraph, and a hyphen split across a column or page.

A line engine emits one block per line. Consecutive lines of one column join.
A finished region does not join the next block unless its text ends in a hyphen.
A footnote and the column's Table/Fig/Eq index line stay separate. Reading
order finishes the left column before the right one, so a word broken at the
column edge is stitched back without moving the rest of the right column.
The same stitch reaches the next page only when that page's first unit is the tail.
"""

import re
from dataclasses import replace

from llb.prep.book.render.constants import COLUMN_BREAK, INDENT_FRACTION, LINE_STEP_BREAK
from llb.prep.book.render.types import RenderBlock, Unit

_SENTENCE = (".", "?", "!")
_INDEX_LINE = re.compile(r"^Table\b.*\bFig\b.*\bEq\b")


def paragraph_runs(blocks: list[RenderBlock]) -> list[list[RenderBlock]]:
    """Groups of body lines that read as one paragraph. Anything else stands alone."""
    runs: list[list[RenderBlock]] = []
    for block in blocks:
        if runs and _continues(runs[-1][-1], block):
            runs[-1].append(block)
        else:
            runs.append([block])
    return runs


def stitch_columns(units: list[Unit]) -> list[Unit]:
    """Move a column-break hyphenation's tail back onto the line that began the word."""
    stitched = list(units)
    for index, unit in enumerate(stitched):
        tail = _column_tail(stitched, index, unit)
        if tail is None:
            continue
        stitched[index], stitched[tail] = _join_hyphen(unit, stitched[tail])
    return stitched


def _continues(prev: RenderBlock, nxt: RenderBlock) -> bool:
    if prev.role != "body" or nxt.role != "body":
        return False
    if prev.paragraph and not prev.text.rstrip().endswith("-"):
        return False
    if prev.pdf_page != nxt.pdf_page or prev.region is None or nxt.region is None:
        return False
    if _column(prev, nxt) or _gap(prev, nxt) or _indent(prev, nxt) or _aside(prev, nxt):
        return False
    return True


def _column(prev: RenderBlock, nxt: RenderBlock) -> bool:
    assert prev.region is not None and nxt.region is not None
    return abs(prev.region.x0 - nxt.region.x0) > prev.region.width * COLUMN_BREAK


def _gap(prev: RenderBlock, nxt: RenderBlock) -> bool:
    assert prev.region is not None and nxt.region is not None
    height = max(prev.region.y1 - prev.region.y0, 1.0)
    if nxt.region.y0 + 4 < prev.region.y0:
        return True
    return nxt.region.y0 - prev.region.y0 > height * LINE_STEP_BREAK


def _indent(prev: RenderBlock, nxt: RenderBlock) -> bool:
    assert prev.region is not None and nxt.region is not None
    if not _sentence(prev.text):
        return False
    height = max(prev.region.y1 - prev.region.y0, 1.0)
    return nxt.region.x0 > prev.region.x0 + height * INDENT_FRACTION


def _sentence(text: str) -> bool:
    stripped = text.rstrip().rstrip("\"')}]")
    return bool(stripped) and stripped.endswith(_SENTENCE)


def _aside(prev: RenderBlock, nxt: RenderBlock) -> bool:
    """A footnote or the column's Table/Fig/Eq index line is not the next sentence."""
    return _footnote(nxt.text) or _index_line(prev.text) or _index_line(nxt.text)


def _footnote(text: str) -> bool:
    return text.lstrip().startswith("*")


def _index_line(text: str) -> bool:
    return _INDEX_LINE.match(text.strip()) is not None


def _column_tail(units: list[Unit], index: int, unit: Unit) -> int | None:
    if unit.role != "paragraph" or not unit.blocks or not unit.text.rstrip().endswith("-"):
        return None
    last = unit.blocks[-1]
    for cursor in range(index + 1, len(units)):
        found = _tail_at(units[cursor], last)
        if found is not None:
            return cursor if found else None
    return None


def _tail_at(unit: Unit, last: RenderBlock) -> bool | None:
    """True when `unit` holds the hyphen's tail, False when the search should stop."""
    if not unit.blocks:
        return None
    head = unit.blocks[0]
    if head.pdf_page != last.pdf_page:
        return _page_tail(unit, last, head)
    if unit.role != "paragraph" or _footnote(unit.text) or _index_line(unit.text):
        return None
    if last.region is None or head.region is None or not _column(last, head):
        return False
    word = unit.text.lstrip().split(" ", 1)[0]
    return bool(word) and word[0].islower() and "[" not in word


def _page_tail(unit: Unit, last: RenderBlock, head: RenderBlock) -> bool:
    """The next page continues the word only when its first unit is the hyphen's tail."""
    if head.pdf_page != last.pdf_page + 1 or unit.role != "paragraph":
        return False
    if head.region is None or last.region is None or _column(last, head):
        return False
    if _footnote(unit.text) or _index_line(unit.text):
        return False
    word = unit.text.lstrip().split(" ", 1)[0]
    return bool(word) and word[0].islower() and "[" not in word


def _join_hyphen(left: Unit, right: Unit) -> tuple[Unit, Unit]:
    base = left.text.rstrip()
    word, _sep, tail = right.text.lstrip().partition(" ")
    joined = replace(left, text=base[:-1] + word)
    return joined, replace(right, text=tail)
