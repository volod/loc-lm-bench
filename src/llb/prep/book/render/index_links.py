"""Page tails inside an index note: comma lists and dot leaders.

Imported by ``link`` only while linking an index note, so the two modules do
not import each other at load time.
"""

import re

from llb.prep.book.render.catalog import Catalog
from llb.prep.book.render.grammar import page_token
from llb.prep.book.render.link import Replacement, Resolve, _blocked, _page_resolver, _rewrite
from llb.prep.book.structure.text import norm_printed, roman_value

_PAGE_TOKEN = page_token()
_DASH = r"[\-\u2013\u2014]"
_PAGE_ONE = re.compile(r"(?i)^\s*(?:[ivxlcdm]+|\d+)\.?\s*$")
_PAGE_RANGE = re.compile(rf"(?i)^\s*([ivxlcdm]+|\d+)\s*{_DASH}\s*([ivxlcdm]+|\d+)\.?\s*$")
_DOT = re.compile(
    rf"(?i)^(?P<head>.*[A-Za-z].*?)\.{{2,}}\s*"
    rf"(?P<body>(?:[ivxlcdm]+|\d+)(?:\s*{_DASH}\s*(?:[ivxlcdm]+|\d+))?)\s*$"
)


def index_replacements(
    text: str, blocked: list[tuple[int, int]], here: str, catalog: Catalog
) -> list[Replacement]:
    """One replacement per page token on an index line."""
    found: list[Replacement] = []
    offset = 0
    resolve = _page_resolver(here, catalog, "index")
    for line in text.splitlines(keepends=True):
        body = line[:-1] if line.endswith("\n") else line
        found.extend(_index_line(body, offset, blocked, resolve))
        offset += len(line)
    return found


def _index_line(
    body: str, offset: int, blocked: list[tuple[int, int]], resolve: Resolve
) -> list[Replacement]:
    found: list[Replacement] = []
    for start, end in _page_pieces(body):
        if _blocked(offset + start, offset + end, blocked):
            continue
        rewritten, events = _rewrite(body[start:end], _PAGE_TOKEN, resolve, "index")
        if events:
            found.append(Replacement(offset + start, offset + end, rewritten, events))
    return found


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
