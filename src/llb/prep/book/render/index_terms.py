"""Index lines: the term, an indented parent, and the printed pages on the tail.

Page tails are the same pieces the vault linker cites. A range maps to every printed
page it covers, up to a fixed width; a longer span keeps its endpoints. A line of only
page numbers continues the previous term. A line with no letters and no pages is skipped.
"""

import re
from dataclasses import dataclass

from llb.prep.book.render.index_links import _page_pieces
from llb.prep.book.render.index_segments import index_segments
from llb.prep.book.structure.text import int_to_roman, norm_printed, roman_value

# A printed range longer than this maps only its endpoints.
_MAX_INDEX_RANGE = 40

_HEADING = re.compile(r"^\s{0,3}#{1,6}(\s|$)")
_MARKER = "{{llb-marker:"
_DASH = r"[\-\u2013\u2014]"
_RANGE = re.compile(rf"(?i)^\s*([ivxlcdm]+|\d+)\s*{_DASH}\s*([ivxlcdm]+|\d+)\.?\s*$")
_TOKEN = re.compile(r"(?i)(?<![A-Za-z])([ivxlcdm]+|\d+)(?![A-Za-z])")


@dataclass(slots=True)
class IndexLine:
    """One index term in document coordinates, plus the printed pages on its tail."""

    term_start: int
    term_end: int
    parent_start: int | None
    parent_end: int | None
    pages: tuple[str, ...]


def index_lines(text: str) -> tuple[IndexLine, ...]:
    """The index terms in ``text``. A continuation line adds its pages to the previous term."""
    found: list[IndexLine] = []
    stack: list[tuple[int, int, int]] = []
    for offset, body in index_segments(text):
        _one_line(body, offset, found, stack)
    return tuple(found)


def _one_line(
    body: str, offset: int, found: list[IndexLine], stack: list[tuple[int, int, int]]
) -> None:
    if not body.strip() or _HEADING.match(body) or body.lstrip().startswith(_MARKER):
        return
    pieces = _page_pieces(body)
    pages = _pages(body, pieces)
    term = _term(body, pieces)
    if term is None:
        _continue(found, pages)
        return
    indent = len(body) - len(body.lstrip(" \t"))
    start, end = term
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


def _continue(found: list[IndexLine], pages: tuple[str, ...]) -> None:
    if not pages or not found:
        return
    previous = found[-1]
    seen = set(previous.pages)
    extra = tuple(page for page in pages if page not in seen)
    if extra:
        previous.pages = previous.pages + extra


def _pages(line: str, pieces: list[tuple[int, int]]) -> tuple[str, ...]:
    keys: list[str] = []
    seen: set[str] = set()
    for start, end in sorted(pieces):
        for key in _keys(line[start:end]):
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return tuple(keys)


def _keys(piece: str) -> tuple[str, ...]:
    match = _RANGE.match(piece.strip())
    if match is not None:
        expanded = _expand(match.group(1), match.group(2))
        if expanded:
            return expanded
    return tuple(norm_printed(token) for token in _TOKEN.findall(piece))


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


def _term(line: str, pieces: list[tuple[int, int]]) -> tuple[int, int] | None:
    cut = min(start for start, _end in pieces) if pieces else len(line)
    head = line[:cut].rstrip(" \t,.")
    term = head.lstrip(" \t")
    if not re.search(r"[A-Za-z]", term):
        return None
    start = len(head) - len(term)
    return start, len(head)
