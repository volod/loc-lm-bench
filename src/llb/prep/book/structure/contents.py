"""Table-of-contents pages, and the entries printed on them.

An entry is evidence for a heading. It never becomes a hierarchy node by itself.
"""

import re
from dataclasses import dataclass, replace

from llb.prep.book.structure.constants import (
    COLUMN_CUT,
    COLUMN_MIN_LINES,
    CONTENTS_BOTTOM,
    CONTENTS_DENSITY,
    HEADER_BAND,
    ROW_Y_FRACTION,
)
from llb.prep.book.structure.lines import Line
from llb.prep.book.structure.text import (
    clean_title,
    norm_space,
    page_token,
    peel_printed,
    split_heading,
)

_CONTENTS_TITLE = re.compile(r"^(?:table\s+of\s+)?contents$", re.IGNORECASE)
_LEADERS = re.compile(r"\.{2,}|…")
_BARE_NUMBER = re.compile(r"^\d+(?:\.\d+)*$")


@dataclass(frozen=True)
class Entry:
    """One contents line: where it was read, and the heading it announces."""

    level: str
    number: str
    title: str
    printed: str
    pdf_page: int
    block_id: str


@dataclass(frozen=True)
class _Row:
    kind: str
    level: str
    number: str
    title: str
    printed: str
    pdf_page: int
    block_id: str


def contents_pages(grouped: list[tuple[int, list[Line]]]) -> list[int]:
    """PDF pages that are the contents, including a following page that continues them."""
    found: list[int] = []
    previous = False
    for pdf, lines in grouped:
        title = any(_CONTENTS_TITLE.match(norm_space(line.text)) for line in lines)
        if title or (previous and _density(lines) >= CONTENTS_DENSITY):
            found.append(pdf)
            previous = True
        else:
            previous = False
    return found


def contents_entries(grouped: list[tuple[int, list[Line]]], pages: set[int]) -> list[Entry]:
    """Contents entries in reading order. Pages that are not contents contribute none."""
    entries: list[Entry] = []
    for pdf, lines in grouped:
        if pdf not in pages:
            continue
        entries.extend(_page_entries(lines))
    return entries


def _density(lines: list[Line]) -> float:
    """Leaders and bare numbers in the body. A page number in the margin does not count."""
    body = [line for line in lines if HEADER_BAND < line.y_frac < CONTENTS_BOTTOM]
    if not body:
        return 0.0
    hits = sum(1 for line in body if _entryish(norm_space(line.text)))
    return hits / len(body)


def _entryish(text: str) -> bool:
    """A contents line carries leaders, a bare section number, or only a page token."""
    if _LEADERS.search(text) or _BARE_NUMBER.match(text):
        return True
    return page_token(text, 1.0) is not None


def _page_entries(lines: list[Line]) -> list[Entry]:
    entries: list[Entry] = []
    for column in _columns(lines):
        entries.extend(_stitch(_rows(column)))
    return entries


def _columns(lines: list[Line]) -> list[list[Line]]:
    body = [line for line in lines if line.y_frac > HEADER_BAND]
    left = [line for line in body if line.center_frac < COLUMN_CUT]
    right = [line for line in body if line.center_frac >= COLUMN_CUT]
    if len(left) >= COLUMN_MIN_LINES and len(right) >= COLUMN_MIN_LINES:
        return [left, right]
    return [body]


def _rows(lines: list[Line]) -> list[_Row]:
    rows: list[_Row] = []
    for group in _cluster(lines):
        ordered = sorted(group, key=lambda line: line.x0)
        text = norm_space(" ".join(line.text for line in ordered))
        row = _classify(text, ordered[0])
        if row is not None:
            rows.append(row)
    return rows


def _cluster(lines: list[Line]) -> list[list[Line]]:
    ordered = sorted(lines, key=lambda line: (line.y0, line.x0))
    groups: list[list[Line]] = []
    for line in ordered:
        if groups and abs(line.y0 - groups[-1][0].y0) <= line.page_height * ROW_Y_FRACTION:
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


def _classify(text: str, line: Line) -> _Row | None:
    if _CONTENTS_TITLE.match(text):
        return None
    token = page_token(text, line.x_frac)
    if token is not None:
        return _row("page", "", "", "", token, line)
    body, printed = peel_printed(text)
    parts = split_heading(body)
    if parts is not None and (parts.title or parts.number):
        return _row("heading", parts.level, parts.number, parts.title, printed, line)
    if re.fullmatch(r"\d+(?:\.\d+)*", body):
        return _row("number", _number_level(body), body, "", printed, line)
    title = clean_title(body)
    if title:
        return _row("title", "chapter", "", title, printed, line)
    return None


def _number_level(number: str) -> str:
    if "." not in number:
        return "chapter"
    if number.count(".") == 1:
        return "section"
    return "subsection"


def _row(kind: str, level: str, number: str, title: str, printed: str, line: Line) -> _Row:
    return _Row(kind, level, number, title, printed, line.pdf_page, line.block_id)


def _stitch(rows: list[_Row]) -> list[Entry]:
    entries: list[Entry] = []
    pending: _Row | None = None
    for row in rows:
        pending = _take(entries, pending, row)
    _emit(entries, pending)
    return entries


def _take(entries: list[Entry], pending: _Row | None, row: _Row) -> _Row | None:
    if pending is not None and _continues(pending, row):
        pending = _merge(pending, row)
        return _close(entries, pending)
    _emit(entries, pending)
    if row.kind == "page":
        return None
    return _close(entries, row)


def _continues(pending: _Row, row: _Row) -> bool:
    if pending.printed:
        return False
    return row.kind in {"page", "title"}


def _merge(pending: _Row, row: _Row) -> _Row:
    title = pending.title
    if row.title:
        title = f"{title} {row.title}".strip() if title else row.title
    return replace(pending, title=clean_title(title), printed=pending.printed or row.printed)


def _close(entries: list[Entry], pending: _Row | None) -> _Row | None:
    if pending is not None and pending.printed and pending.title:
        _emit(entries, pending)
        return None
    return pending


def _emit(entries: list[Entry], pending: _Row | None) -> None:
    if pending is None or not pending.title:
        return
    if not pending.printed and not pending.number:
        return
    entries.append(
        Entry(
            pending.level,
            pending.number,
            pending.title,
            pending.printed,
            pending.pdf_page,
            pending.block_id,
        )
    )
