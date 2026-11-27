"""Figure, table, and equation numbers read from captions, footers, and display numbers."""

import re
from collections import Counter

from llb.core.contracts.data_prep.book_structure import ItemEvidence, ItemKind, NumberedItem
from llb.prep.book.structure.constants import FOOTER_BAND, LIST_CAPTION_MIN
from llb.prep.book.structure.lines import Line
from llb.prep.book.structure.text import norm_space

_FIG = re.compile(r"(?:figure|fig\.)\s*([A-Za-z]?\d+(?:[.\-]\d+)*)", re.IGNORECASE)
_TAB = re.compile(r"\btable\s+([A-Za-z]?\d+(?:[.\-]\d+)*)", re.IGNORECASE)
_EQ = re.compile(r"(?:equation|eq\.)\s*([A-Za-z]?\d+(?:[.\-]\d+)*)", re.IGNORECASE)
_CAPTION_FIG = re.compile(r"^(?:figure|fig\.)\s*([A-Za-z]?\d+(?:[.\-]\d+)*)", re.IGNORECASE)
_CAPTION_TAB = re.compile(r"^table\s+([A-Za-z]?\d+(?:[.\-]\d+)*)", re.IGNORECASE)
_CAPTION_EQ = re.compile(r"^(?:equation|eq\.)\s*([A-Za-z]?\d+(?:[.\-]\d+)*)", re.IGNORECASE)
_DISPLAY = re.compile(r"^\(([A-Za-z]?\d+(?:[.\-]\d+)*)\)$")
# A lowercase word straight after the number is a sentence running on, not a caption title.
_PROSE_AFTER = re.compile(r"\s+[a-z]")


def inventory_of(lines: list[Line], printed_of: dict[int, str | None]) -> list[NumberedItem]:
    """One row per number per page. A caption wins over the same number in the footer."""
    found: list[NumberedItem] = []
    seen: set[tuple[str, str, int]] = set()
    listed = _list_pages(lines)
    ordered = sorted(lines, key=lambda line: (line.pdf_page, line.y0, line.x0))
    for line in ordered:
        for kind, number, evidence in _items(line):
            if evidence == "caption" and line.pdf_page in listed:
                continue
            key = (kind, number, line.pdf_page)
            if key in seen:
                continue
            seen.add(key)
            found.append(_item(line, kind, number, evidence, printed_of))
    return found


def caption_figure_number(text: str) -> str | None:
    """The printed figure number when ``text`` opens as a figure caption, else none."""
    return _opening(_CAPTION_FIG, text)


def caption_table_number(text: str) -> str | None:
    """The printed table number when ``text`` opens as a table caption, else none."""
    return _opening(_CAPTION_TAB, text)


def caption_equation_number(text: str) -> str | None:
    """The printed equation number when ``text`` opens as an equation caption, else none."""
    return _opening(_CAPTION_EQ, text)


def _opening(pattern: re.Pattern[str], text: str) -> str | None:
    matched = pattern.match(norm_space(text))
    if matched is None:
        return None
    return matched.group(1)


def _items(line: Line) -> list[tuple[ItemKind, str, ItemEvidence]]:
    text = norm_space(line.text)
    if line.y_frac >= FOOTER_BAND:
        return (
            _found(text, _FIG, "figure", "page-footer")
            + _found(text, _TAB, "table", "page-footer")
            + _found(text, _EQ, "equation", "page-footer")
        )
    caption = (
        _caption(text, _CAPTION_FIG, "figure")
        + _caption(text, _CAPTION_TAB, "table")
        + _caption(text, _CAPTION_EQ, "equation")
    )
    if caption:
        return caption
    display = _DISPLAY.match(text)
    if display:
        return [("equation", display.group(1), "display-number")]
    return []


def _list_pages(lines: list[Line]) -> set[int]:
    """Pages whose caption lines are a list of figures or tables, not the items themselves."""
    counts: Counter[int] = Counter()
    for line in lines:
        if line.y_frac >= FOOTER_BAND:
            continue
        text = norm_space(line.text)
        if _CAPTION_FIG.match(text) or _CAPTION_TAB.match(text) or _CAPTION_EQ.match(text):
            counts[line.pdf_page] += 1
    return {pdf for pdf, count in counts.items() if count >= LIST_CAPTION_MIN}


def _caption(
    text: str, pattern: re.Pattern[str], kind: ItemKind
) -> list[tuple[ItemKind, str, ItemEvidence]]:
    """A line that opens with a numbered label, unless prose runs on from the number.

    `Figure 3-1 shows the transfer.` cites the figure; it is not the figure's caption, and
    taking it as one would anchor the figure on the sentence. The kind is not consulted.
    """
    matched = pattern.match(text)
    if matched is None or _PROSE_AFTER.match(text, matched.end()) is not None:
        return []
    return [(kind, matched.group(1), "caption")]


def _found(
    text: str,
    pattern: re.Pattern[str],
    kind: ItemKind,
    evidence: ItemEvidence,
) -> list[tuple[ItemKind, str, ItemEvidence]]:
    return [(kind, match.group(1), evidence) for match in pattern.finditer(text)]


def _item(
    line: Line,
    kind: ItemKind,
    number: str,
    evidence: ItemEvidence,
    printed_of: dict[int, str | None],
) -> NumberedItem:
    return NumberedItem(
        kind=kind,
        number=number,
        pdf_page=line.pdf_page,
        printed=printed_of.get(line.pdf_page),
        block_id=line.block_id,
        evidence=evidence,
    )
