"""Recognize the same index entries and page tails in the vault and projection."""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from llb.prep.book.render.index_segments import index_segments
from llb.prep.book.structure.text import norm_printed, roman_value

_HEADING = re.compile(r"^\s{0,3}#{1,6}(\s|$)")
_MARKER = "{{llb-marker:"
_DASH = r"[\-\u2013\u2014]"
_PAGE_ONE = re.compile(r"(?i)^\s*(?:[ivxlcdm]+|\d+)\.?\s*$")
_PAGE_RANGE = re.compile(rf"(?i)^\s*([ivxlcdm]+|\d+)\s*{_DASH}\s*([ivxlcdm]+|\d+)\.?\s*$")
_DOT = re.compile(
    rf"(?i)^(?P<head>.*[A-Za-z].*?)\.{{2,}}\s*"
    rf"(?P<body>(?:[ivxlcdm]+|\d+)(?:\s*{_DASH}\s*(?:[ivxlcdm]+|\d+))?)\s*$"
)


@dataclass(frozen=True, slots=True)
class IndexSegment:
    offset: int
    body: str
    term: tuple[int, int] | None
    pages: list[tuple[int, int]]


def index_line_parts(text: str) -> Iterator[IndexSegment]:
    """Page-only lines continue an entry that already has an explicit page tail."""
    continuing_entry = False
    for offset, body in index_segments(text):
        if not body.strip():
            continue
        if _HEADING.match(body) or body.lstrip().startswith(_MARKER):
            continuing_entry = False
            continue
        continuation = _parts(body)
        if all(_is_page_piece(body[start:end]) for start, end in continuation):
            if continuing_entry:
                yield IndexSegment(offset, body, None, continuation)
            continue
        pages = _page_pieces(body)
        continuing_entry = bool(pages)
        if (term := _term(body, pages)) is not None:
            yield IndexSegment(offset, body, term, pages)


def _page_pieces(line: str) -> list[tuple[int, int]]:
    tail = _comma_tail(line)
    if tail:
        return tail
    return _dot_tail(line)


def _comma_tail(line: str) -> list[tuple[int, int]]:
    parts = _parts(line)
    tokens: list[tuple[int, int]] = []
    index = len(parts) - 1
    while index > 0 and _is_page_piece(line[parts[index][0] : parts[index][1]]):
        tokens.append(parts[index])
        index -= 1
    if not tokens or not re.search(r"[A-Za-z]", line[: tokens[-1][0]]):
        return []
    return tokens


def _dot_tail(line: str) -> list[tuple[int, int]]:
    match = _DOT.match(line)
    if match is None or not _is_page_piece(match.group("body")):
        return []
    return [(match.start("body"), match.end("body"))]


def _parts(line: str) -> list[tuple[int, int]]:
    parts: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r",", line):
        parts.append((start, match.start()))
        start = match.end()
    parts.append((start, len(line)))
    return parts


def _is_page_piece(text: str) -> bool:
    if _PAGE_ONE.match(text):
        return True
    match = _PAGE_RANGE.match(text)
    return match is not None and _forward(match.group(1), match.group(2))


def _forward(left: str, right: str) -> bool:
    return _order(norm_printed(left)) <= _order(norm_printed(right))


def _order(key: str) -> int:
    if key.isdigit():
        return int(key)
    value = roman_value(key)
    return value if value is not None else 10**9


def _term(line: str, pieces: list[tuple[int, int]]) -> tuple[int, int] | None:
    cut = min(start for start, _end in pieces) if pieces else len(line)
    head = line[:cut].rstrip(" \t,.")
    term = head.lstrip(" \t")
    if not re.search(r"[A-Za-z]", term):
        return None
    start = len(head) - len(term)
    return start, len(head)
