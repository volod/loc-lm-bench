"""Index lines: the term, an indented parent, and the printed pages on the tail.

Page tails are the same pieces the vault linker cites. A range maps to every printed
page it covers, up to a fixed width; a longer span keeps its endpoints. A line of only
page numbers continues the previous term. A line with no letters and no pages is skipped.
"""

import re
from dataclasses import dataclass

from llb.prep.book.render.index_tails import IndexSegment, index_line_parts
from llb.prep.book.structure.text import int_to_roman, norm_printed, roman_value

# A printed range longer than this maps only its endpoints.
_MAX_INDEX_RANGE = 40

_DASH = r"[\-\u2013\u2014]"
_RANGE = re.compile(rf"(?i)^\s*([ivxlcdm]+|\d+)\s*{_DASH}\s*([ivxlcdm]+|\d+)\.?\s*$")
_TOKEN = re.compile(r"(?i)(?<![A-Za-z])([ivxlcdm]+|\d+)(?![A-Za-z])")


@dataclass(slots=True)
class IndexLabel:
    """One mapped label and the exact token or range that prints it."""

    printed: str
    start: int
    end: int


@dataclass(slots=True)
class IndexLine:
    """One index term in document coordinates, plus the printed pages on its tail."""

    term_start: int
    term_end: int
    parent_start: int | None
    parent_end: int | None
    pages: tuple[IndexLabel, ...]


def index_lines(text: str) -> tuple[IndexLine, ...]:
    """The index terms in ``text``. A continuation line adds its pages to the previous term."""
    found: list[IndexLine] = []
    stack: list[tuple[int, int, int]] = []
    for line in index_line_parts(text):
        _one_line(line, found, stack)
    return tuple(found)


def _one_line(
    line: IndexSegment, found: list[IndexLine], stack: list[tuple[int, int, int]]
) -> None:
    body, offset = line.body, line.offset
    pages = _pages(body, offset, line.pages)
    if line.term is None:
        _continue(found, pages)
        return
    indent = len(body) - len(body.lstrip(" \t"))
    start, end = line.term
    while stack and stack[-1][0] >= indent:
        stack.pop()
    parent = stack[-1] if stack and indent > stack[-1][0] else None
    found.append(
        IndexLine(
            offset + start,
            offset + end,
            None if parent is None else parent[1],
            None if parent is None else parent[2],
            pages,
        )
    )
    stack.append((indent, offset + start, offset + end))


def _continue(found: list[IndexLine], pages: tuple[IndexLabel, ...]) -> None:
    if not pages or not found:
        return
    previous = found[-1]
    seen = {page.printed for page in previous.pages}
    extra = tuple(page for page in pages if page.printed not in seen)
    if extra:
        previous.pages = previous.pages + extra


def _pages(line: str, offset: int, pieces: list[tuple[int, int]]) -> tuple[IndexLabel, ...]:
    keys: dict[str, IndexLabel] = {}
    for start, end in sorted(pieces):
        for key, left, right in _keys(line[start:end]):
            keys.setdefault(key, IndexLabel(key, offset + start + left, offset + start + right))
    return tuple(keys.values())


def _keys(piece: str) -> list[tuple[str, int, int]]:
    match = _RANGE.match(piece)
    if match is not None:
        expanded = _expand(match.group(1), match.group(2))
        if expanded:
            return [(key, match.start(1), match.end(2)) for key in expanded]
    return [
        (norm_printed(token[1]), token.start(1), token.end(1)) for token in _TOKEN.finditer(piece)
    ]


def _expand(left: str, right: str) -> tuple[str, ...] | None:
    if left.isdigit() and right.isdigit():
        start, stop = int(left), int(right)
        if stop < start or stop - start + 1 > _MAX_INDEX_RANGE:
            return None
        return tuple(str(number) for number in range(start, stop + 1))
    first, second = roman_value(left), roman_value(right)
    if first is None or second is None or second < first:
        return None
    if second - first + 1 > _MAX_INDEX_RANGE:
        return None
    return tuple(int_to_roman(number) for number in range(first, second + 1))
