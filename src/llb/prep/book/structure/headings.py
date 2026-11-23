"""Headings read from numbering and from type size. A contents page is not a heading page."""

import re
from dataclasses import dataclass

from llb.prep.book.structure.constants import (
    FOOTER_BAND,
    HEADER_BAND,
    CHAPTER_OPENER_Y,
    HEADING_COMPONENT_MAX,
    HEADING_JOIN_GAP,
    HEADING_LONG_WORD,
    HEADING_MAX_CHARS,
    HEADING_MAX_WORDS,
    HEADING_MIN_WIDTH,
    HEADING_NUMBER_MAX_X,
    HEADING_STACK_FRACTION,
    MARGIN_HEADING_WIDTH,
    MARGIN_HEADING_X,
    TYPOGRAPHY_MAX_Y,
)
from llb.prep.book.structure.lines import Line
from llb.prep.book.structure.text import HeadingParts, clean_title, norm_space, split_heading

_FRONT = frozenset(
    {
        "preface",
        "foreword",
        "acknowledgements",
        "acknowledgments",
        "glossary",
        "bibliography",
        "references",
        "index",
    }
)
_BARE_CHAPTER = re.compile(r"^\d{1,2}$")
_SENTENCE_END = re.compile(r"[.,;:=\-]\s*$")


@dataclass(frozen=True)
class Heading:
    """One heading observed on a page."""

    level: str
    number: str
    title: str
    pdf_page: int
    block_id: str
    evidence: str


def headings_of(
    grouped: list[tuple[int, list[Line]]],
    contents: set[int],
    furniture_ids: set[str],
) -> list[Heading]:
    """Numbered and typographic headings, in reading order, outside the contents."""
    found: list[Heading] = []
    for pdf, lines in grouped:
        if pdf in contents:
            continue
        found.extend(_page(lines, furniture_ids))
    return found


def _page(lines: list[Line], furniture_ids: set[str]) -> list[Heading]:
    body = [line for line in lines if _body(line, furniture_ids)]
    found: list[Heading] = []
    index = 0
    while index < len(body):
        step, heading = _at(body, index)
        if heading is not None:
            found.append(heading)
        index += step
    return found


def _body(line: Line, furniture_ids: set[str]) -> bool:
    if line.block_id in furniture_ids:
        return False
    if line.y_frac <= HEADER_BAND or line.y_frac >= FOOTER_BAND:
        return False
    return bool(line.text)


def _at(body: list[Line], index: int) -> tuple[int, Heading | None]:
    line = body[index]
    text = norm_space(line.text)
    parts = split_heading(text)
    if parts is not None and _numbered(parts, text, line) and not _margin(line):
        return 1, _make(parts, line, "numbered-heading")
    joined = _join(body, index)
    if joined is not None:
        return 2, joined
    plain = _typography(line)
    if plain is not None:
        return 1, plain
    return 1, None


def _numbered(parts: HeadingParts, text: str, line: Line) -> bool:
    if not parts.number or not parts.title or not _plausible(parts.number):
        return False
    if not _bare_chapter_at_top(parts.number, line):
        return False
    words = parts.title.split()
    if len(text) > HEADING_MAX_CHARS or len(words) > HEADING_MAX_WORDS:
        return False
    if "," in parts.title or "(" in parts.title or parts.title.endswith(")"):
        return False
    if _SENTENCE_END.search(text):
        return False
    return _substantial(parts.title, line)


def _substantial(title: str, line: Line) -> bool:
    """A heading is words. A cell is a unit, a short token, or a parenthesis."""
    words = [word for word in title.split() if any(char.isalpha() for char in word)]
    if not words or (len(words) == 1 and len(words[0]) <= 2):
        return False
    if line.width_frac >= HEADING_MIN_WIDTH:
        return True
    return max(len(word) for word in words) >= HEADING_LONG_WORD


def _bare_chapter_at_top(number: str, line: Line) -> bool:
    """Section numbers occur anywhere. A bare chapter number only opens a page."""
    if "." in number or any(char.isalpha() for char in number):
        return True
    return line.y_frac <= CHAPTER_OPENER_Y


def _plausible(number: str) -> bool:
    """A chapter or section index. A page number or a measurement is not one."""
    if any(char.isalpha() for char in number):
        return number.isalnum() and len(number) <= 4
    parts = number.split(".")
    if not 1 <= len(parts) <= 4:
        return False
    for part in parts:
        if not part.isdigit() or len(part) > 2 or int(part) > HEADING_COMPONENT_MAX:
            return False
    return not (len(parts) == 1 and int(parts[0]) == 0)


def _join(body: list[Line], index: int) -> Heading | None:
    if index + 1 >= len(body):
        return None
    line = body[index]
    nxt = body[index + 1]
    number = norm_space(line.text)
    title = norm_space(nxt.text)
    if line.center_frac > HEADING_NUMBER_MAX_X:
        return None
    if _BARE_CHAPTER.match(number) is None or not _near(line, nxt) or not _short(title):
        return None
    if not title[:1].isupper() or title[-1:] in ".,;:)" or "(" in title:
        return None
    parts = split_heading(f"{number} {title}")
    if parts is None or not _numbered(parts, title, line):
        return None
    return _make(parts, line, "numbered-heading")


def _margin(line: Line) -> bool:
    """A short token in the outer right column is a page or cell number, not a heading."""
    return line.center_frac > MARGIN_HEADING_X and line.width_frac < MARGIN_HEADING_WIDTH


def _near(line: Line, nxt: Line) -> bool:
    if abs(line.x0 - nxt.x0) > line.page_width * 0.15:
        return False
    # The title is the next line down. A token on the same row is a table cell.
    if nxt.y0 < line.y0 + (line.y1 - line.y0) * HEADING_STACK_FRACTION:
        return False
    return nxt.y0 - line.y1 <= line.page_height * HEADING_JOIN_GAP


def _short(text: str) -> bool:
    words = text.split()
    return 1 <= len(words) <= HEADING_MAX_WORDS and len(text) <= HEADING_MAX_CHARS


def _typography(line: Line) -> Heading | None:
    text = norm_space(line.text)
    letters = re.sub(r"[^a-z]", "", text.lower())
    if letters not in _FRONT or line.y_frac > TYPOGRAPHY_MAX_Y or not _short(text):
        return None
    title = clean_title(text)
    if not title:
        return None
    return Heading("chapter", "", title, line.pdf_page, line.block_id, "typography")


def _make(parts: HeadingParts, line: Line, evidence: str) -> Heading:
    return Heading(parts.level, parts.number, parts.title, line.pdf_page, line.block_id, evidence)
