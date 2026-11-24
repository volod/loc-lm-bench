"""Page-number and running-header lines, found from where they sit and what repeats.

The block kind is not read. A page number is a short numeral in the outer margin of the
band where such numerals repeat. A running header is the other short text in the top band.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

from llb.core.contracts.data_prep.book_structure import RunningHeader
from llb.prep.book.structure.constants import (
    CENTER_HIGH,
    CENTER_LOW,
    FOOTER_BAND,
    HEADER_BAND,
    HEADER_MAX_CHARS,
    HEADER_MAX_WORDS,
    HEADER_MIN_PAGES,
    HEADER_REPEAT,
    PAGE_NUMBER_BAND,
    NARROW_WIDTH,
    OUTER_MARGIN,
)
from llb.prep.book.structure.lines import Line
from llb.prep.book.structure.text import header_key, norm_space, roman_value, split_heading

_CONTENTS_TITLE = re.compile(r"^(?:table\s+of\s+)?contents$", re.IGNORECASE)
_LEADERS = re.compile(r"\.{2,}|…")


@dataclass(frozen=True)
class PageHit:
    """One numeral that sits where a printed page number sits."""

    pdf: int
    value: int
    scheme: str
    block_id: str


def page_hits(lines: list[Line]) -> list[PageHit]:
    """The page-number candidates in the band that holds them on the most pages."""
    header: list[PageHit] = []
    footer: list[PageHit] = []
    for line in lines:
        hit = _hit(line)
        if hit is None:
            continue
        if line.y_frac <= PAGE_NUMBER_BAND and _outer(line):
            header.append(hit)
        elif line.y_frac >= FOOTER_BAND and (_outer(line) or _centered(line)):
            footer.append(hit)
    header_pages = {hit.pdf for hit in header}
    footer_pages = {hit.pdf for hit in footer}
    if len(footer_pages) > len(header_pages):
        return footer
    return header


def header_spans(lines: list[Line], number_ids: set[str], pages: list[int]) -> list[RunningHeader]:
    """Running headers grouped into contiguous page spans."""
    pool = [line for line in lines if _header_line(line, number_ids)]
    by_key: dict[str, list[Line]] = defaultdict(list)
    for line in pool:
        by_key[header_key(line.text)].append(line)
    band_pages = {line.pdf_page for line in pool}
    established = len(band_pages) >= HEADER_MIN_PAGES
    kept = [line for line in pool if _kept(header_key(line.text), by_key, established)]
    return _spans(kept, pages)


def _hit(line: Line) -> PageHit | None:
    if line.width_frac > NARROW_WIDTH:
        return None
    token = _numeral(line.text)
    if token is None:
        return None
    scheme, value = token
    return PageHit(line.pdf_page, value, scheme, line.block_id)


def _numeral(text: str) -> tuple[str, int] | None:
    cleaned = re.sub(r"[\s|]+", "", text).strip(".-")
    if not cleaned or len(cleaned) > 8:
        return None
    if re.fullmatch(r"\d{1,4}", cleaned):
        value = int(cleaned)
        if value < 1:
            return None
        return ("arabic", value)
    roman = roman_value(cleaned)
    if roman is None or roman < 1:
        return None
    return ("roman", roman)


def _outer(line: Line) -> bool:
    return line.center_frac < OUTER_MARGIN or line.center_frac > 1 - OUTER_MARGIN


def _centered(line: Line) -> bool:
    return CENTER_LOW <= line.center_frac <= CENTER_HIGH


def _header_line(line: Line, number_ids: set[str]) -> bool:
    if line.block_id in number_ids or line.y_frac > HEADER_BAND:
        return False
    text = norm_space(line.text)
    if not text or _CONTENTS_TITLE.match(text) or _LEADERS.search(text):
        return False
    if len(text) > HEADER_MAX_CHARS or len(text.split()) > HEADER_MAX_WORDS:
        return False
    if _numeral(text) is not None:
        return False
    parts = split_heading(text)
    return not (parts is not None and parts.number)


def _kept(key: str, by_key: dict[str, list[Line]], established: bool) -> bool:
    if len(key) < 2:
        return False
    pages = {line.pdf_page for line in by_key[key]}
    return len(pages) >= HEADER_REPEAT or established


def _spans(lines: list[Line], pages: list[int]) -> list[RunningHeader]:
    index = {pdf: position for position, pdf in enumerate(pages)}
    grouped: dict[str, list[Line]] = defaultdict(list)
    for line in lines:
        grouped[header_key(line.text)].append(line)
    spans: list[RunningHeader] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda line: index[line.pdf_page])
        spans.extend(_runs(ordered, index))
    spans.sort(key=lambda span: (span.pdf_start, span.text))
    return spans


def _runs(lines: list[Line], index: dict[int, int]) -> list[RunningHeader]:
    runs: list[list[Line]] = [[lines[0]]]
    for line in lines[1:]:
        previous = runs[-1][-1].pdf_page
        if index[line.pdf_page] == index[previous] + 1 and line.pdf_page != previous:
            runs[-1].append(line)
        elif line.pdf_page == previous:
            runs[-1].append(line)
        else:
            runs.append([line])
    return [_span(run) for run in runs]


def _span(lines: list[Line]) -> RunningHeader:
    ordered = sorted(lines, key=lambda line: (line.pdf_page, line.x0))
    return RunningHeader(
        text=ordered[0].text,
        pdf_start=ordered[0].pdf_page,
        pdf_end=ordered[-1].pdf_page,
        block_ids=[line.block_id for line in ordered],
    )
