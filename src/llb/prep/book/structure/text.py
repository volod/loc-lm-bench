"""Shared parsers for page numbers, headings, and contents lines."""

import re
from dataclasses import dataclass

_ROMAN_VALUES: dict[str, int] = {
    "i": 1,
    "v": 5,
    "x": 10,
    "l": 50,
    "c": 100,
    "d": 500,
    "m": 1000,
}
_ROMAN_GLYPHS: tuple[tuple[int, str], ...] = (
    (1000, "m"),
    (900, "cm"),
    (500, "d"),
    (400, "cd"),
    (100, "c"),
    (90, "xc"),
    (50, "l"),
    (40, "xl"),
    (10, "x"),
    (9, "ix"),
    (5, "v"),
    (4, "iv"),
    (1, "i"),
)
_PART = re.compile(r"^part\s+([ivxlcdm]+|\d+)\b[\s.:\-]*\s*(.*)$", re.IGNORECASE)
_CHAPTER_WORD = re.compile(r"^chapter\s+(\d+|[ivxlcdm]+)\b[\s.:\-]*\s*(.*)$", re.IGNORECASE)
_APPENDIX = re.compile(r"^appendix\s+([a-z]|\d+)\b[\s.:\-]*\s*(.*)$", re.IGNORECASE)
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")
_DOTTED_PAGE = re.compile(
    r"^(?P<body>.*?\S)\s*(?:\.{2,}|…+)\s*(?P<page>[ivxlcdm]+|\d+)\s*$",
    re.IGNORECASE,
)
_SPACED_PAGE = re.compile(
    r"^(?P<body>.*?\S)\s{2,}(?P<page>[ivxlcdm]+|\d+)\s*$",
    re.IGNORECASE,
)
_GLUED_PAGE = re.compile(
    r"^(?P<body>.*[A-Za-z][^.]{0,40}?)\.{1,}(?P<page>\d+)\s*$",
)
_NUMBERED_PAGE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)\.?\s+(?P<title>.+?[A-Za-z])\s+(?P<page>\d+)\s*$",
)
_PAGE_ONLY = re.compile(r"^[\s.…·.]*([ivxlcdm]+|\d+)$", re.IGNORECASE)
_CAPTION = re.compile(r"^(?:figure|fig\.|table|equation|eq\.)\b", re.IGNORECASE)
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


@dataclass(frozen=True)
class HeadingParts:
    """A heading split into its level, its printed number, and its title."""

    level: str
    number: str
    title: str


def norm_space(text: str) -> str:
    """Collapse whitespace and strip the line."""
    return re.sub(r"\s+", " ", text).strip()


def header_key(text: str) -> str:
    """The identity of a running header: letters and digits, case-folded."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def clean_title(text: str) -> str:
    """Drop the dot leaders a contents line leaves on a title."""
    return re.sub(r"[\s.·…]+$", "", norm_space(text)).strip()


def int_to_roman(value: int) -> str:
    """The canonical lowercase roman numeral for a positive integer."""
    remaining = value
    parts: list[str] = []
    for amount, glyph in _ROMAN_GLYPHS:
        while remaining >= amount:
            parts.append(glyph)
            remaining -= amount
    return "".join(parts)


def roman_value(text: str) -> int | None:
    """The integer a roman token names, after the `l`/`i` confusion common in OCR.

    A token is accepted only when it is the canonical numeral for its value, or
    becomes that numeral once a misread `l` is put back to `i`. `xl` stays 40.
    `vili` becomes 8.
    """
    raw = re.sub(r"[^a-z]", "", text.lower())
    if not raw or any(char not in _ROMAN_VALUES for char in raw):
        return None
    for candidate in (raw, raw.replace("l", "i")):
        value = _parse_roman(candidate)
        if value is not None and int_to_roman(value) == candidate and value < 400:
            return value
    return None


def _parse_roman(text: str) -> int | None:
    total = 0
    index = 0
    while index < len(text):
        value = _ROMAN_VALUES.get(text[index])
        if value is None:
            return None
        nxt = _ROMAN_VALUES.get(text[index + 1]) if index + 1 < len(text) else None
        if nxt is not None and nxt > value:
            total += nxt - value
            index += 2
        else:
            total += value
            index += 1
    return total


def norm_printed(token: str) -> str:
    """A printed-page token in the form the mapping stores."""
    stripped = token.strip()
    if re.fullmatch(r"\d+", stripped):
        return str(int(stripped))
    value = roman_value(stripped)
    if value is not None:
        return int_to_roman(value)
    return stripped.lower()


def peel_printed(text: str) -> tuple[str, str]:
    """Split a trailing printed page off a contents line. The page may be empty."""
    numbered = _NUMBERED_PAGE.match(text)
    if numbered:
        body = f"{numbered.group('num')} {clean_title(numbered.group('title'))}"
        return body, norm_printed(numbered.group("page"))
    for pattern in (_DOTTED_PAGE, _SPACED_PAGE, _GLUED_PAGE):
        match = pattern.match(text)
        if match:
            return clean_title(match.group("body")), norm_printed(match.group("page"))
    return text, ""


def page_token(text: str, x_frac: float) -> str | None:
    """A line that is only a printed page, or None when it is a section number."""
    match = _PAGE_ONLY.match(text.strip())
    if match is None:
        return None
    token = match.group(1)
    leaders = bool(re.search(r"[.…·]", text)) or text.strip() != token
    if token.isdigit():
        if leaders or x_frac >= 0.35:
            return str(int(token))
        return None
    value = roman_value(token)
    if value is None:
        return None
    if leaders or x_frac >= 0.35:
        return int_to_roman(value)
    return None


def split_heading(text: str) -> HeadingParts | None:
    """A part, chapter, section, or front-matter heading, or None."""
    if _CAPTION.match(text):
        return None
    part = _PART.match(text)
    if part:
        return _parts("part", _part_number(part.group(1)), part.group(2))
    chapter = _CHAPTER_WORD.match(text)
    if chapter:
        return _parts("chapter", _part_number(chapter.group(1)), chapter.group(2))
    appendix = _APPENDIX.match(text)
    if appendix:
        return _parts("chapter", appendix.group(1).upper(), appendix.group(2))
    numbered = _NUMBERED.match(text)
    if numbered:
        number = numbered.group(1)
        return _parts(_depth(number), number, numbered.group(2))
    letters = re.sub(r"[^a-z]", "", text.lower())
    if letters in _FRONT:
        return HeadingParts("chapter", "", clean_title(text))
    return None


def _parts(level: str, number: str, title: str) -> HeadingParts:
    cleaned = clean_title(title)
    if not cleaned and level == "part":
        cleaned = f"Part {number}"
    if not cleaned and level == "chapter" and number:
        cleaned = f"Chapter {number}"
    return HeadingParts(level, number, cleaned)


def _part_number(token: str) -> str:
    if token.isdigit():
        return str(int(token))
    return token.upper()


def _depth(number: str) -> str:
    dots = number.count(".")
    if dots <= 0:
        return "chapter"
    if dots == 1:
        return "section"
    return "subsection"


def jaccard(left: str, right: str) -> float:
    """Token overlap of two titles. A one-letter token does not count."""
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    shared = left_tokens & right_tokens
    return len(shared) / len(left_tokens | right_tokens)


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}


def printed_delta(left: str, right: str) -> int | None:
    """The absolute distance between two printed labels, or None when they differ in kind."""
    if left.isdigit() and right.isdigit():
        return abs(int(left) - int(right))
    left_roman = roman_value(left)
    right_roman = roman_value(right)
    if left_roman is not None and right_roman is not None:
        return abs(left_roman - right_roman)
    return None
