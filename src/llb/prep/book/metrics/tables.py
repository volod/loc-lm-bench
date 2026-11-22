"""Ordered tree-edit distance on a table, and the grids a reading's text can hold.

The tree is the table node, then each row, then each cell. Inserting or deleting a node costs
one, and relabelling a cell costs one. Similarity is one minus that distance over the two sizes.
A lane that cannot express a grid does not get a zero here; the caller reports the metric absent.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL = re.compile(r"<t([dh])\b([^>]*)>(.*?)</t\1>", re.IGNORECASE | re.DOTALL)
_SPAN = re.compile(r"""colspan\s*=\s*["']?(\d+)""", re.IGNORECASE)
_RSPAN = re.compile(r"""rowspan\s*=\s*["']?(\d+)""", re.IGNORECASE)
_RULE = re.compile(r":?-{3,}:?")
_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Cell:
    """One cell label and its span. The label is the text the tree edit relabels."""

    text: str
    colspan: int = 1
    rowspan: int = 1


def cell_text(text: str) -> str:
    """The label compared during a relabel: trimmed, internal whitespace collapsed."""
    return " ".join(text.split())


def table_size(rows: Sequence[Sequence[Cell]]) -> int:
    """Table node, plus one node per row and per cell."""
    return 1 + len(rows) + sum(len(row) for row in rows)


def table_distance(
    reference: Sequence[Sequence[Cell]], hypothesis: Sequence[Sequence[Cell]]
) -> int:
    """Ordered tree-edit distance. An empty side costs the other side's whole size."""
    if not reference and not hypothesis:
        return 0
    if not reference:
        return table_size(hypothesis)
    if not hypothesis:
        return table_size(reference)
    return _row_distance(reference, hypothesis)


def table_similarity(
    reference: Sequence[Sequence[Cell]], hypothesis: Sequence[Sequence[Cell]]
) -> float:
    """``1 - distance / (size(reference) + size(hypothesis))``."""
    total = table_size(reference) + table_size(hypothesis)
    if total == 0:
        return 1.0
    return 1.0 - table_distance(reference, hypothesis) / total


def parse_table_text(text: str) -> tuple[tuple[Cell, ...], ...] | None:
    """A Markdown or HTML table, or None when the text does not hold a grid."""
    if "<table" in text.lower():
        return _html(text)
    return _markdown(text)


def _markdown(text: str) -> tuple[tuple[Cell, ...], ...] | None:
    rows: list[tuple[Cell, ...]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "|" not in stripped:
            continue
        cells = [cell_text(cell) for cell in stripped.strip("|").split("|")]
        if cells and all(_RULE.fullmatch(cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(tuple(Cell(cell) for cell in cells))
    if not rows:
        return None
    return tuple(rows)


def _html(text: str) -> tuple[tuple[Cell, ...], ...] | None:
    rows: list[tuple[Cell, ...]] = []
    for row in _ROW.findall(text):
        cells = tuple(_html_cell(attrs, body) for _, attrs, body in _CELL.findall(row))
        if cells:
            rows.append(cells)
    if not rows:
        return None
    return tuple(rows)


def _html_cell(attrs: str, body: str) -> Cell:
    colspan = _span(_SPAN, attrs)
    rowspan = _span(_RSPAN, attrs)
    return Cell(cell_text(_TAG.sub(" ", body)), colspan, rowspan)


def _span(pattern: re.Pattern[str], attrs: str) -> int:
    match = pattern.search(attrs)
    if match is None:
        return 1
    return max(1, int(match.group(1)))


def _row_distance(reference: Sequence[Sequence[Cell]], hypothesis: Sequence[Sequence[Cell]]) -> int:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    for index in range(1, rows):
        cost[index][0] = cost[index - 1][0] + _insert(reference[index - 1])
    for index in range(1, cols):
        cost[0][index] = cost[0][index - 1] + _insert(hypothesis[index - 1])
    for left in range(1, rows):
        for right in range(1, cols):
            substitute = cost[left - 1][right - 1] + _cells(
                reference[left - 1], hypothesis[right - 1]
            )
            cost[left][right] = min(
                cost[left - 1][right] + _insert(reference[left - 1]),
                cost[left][right - 1] + _insert(hypothesis[right - 1]),
                substitute,
            )
    return cost[-1][-1]


def _insert(row: Sequence[Cell]) -> int:
    return 1 + len(row)


def _cells(reference: Sequence[Cell], hypothesis: Sequence[Cell]) -> int:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    for index in range(1, rows):
        cost[index][0] = index
    for index in range(1, cols):
        cost[0][index] = index
    for left in range(1, rows):
        for right in range(1, cols):
            relabel = 0 if reference[left - 1] == hypothesis[right - 1] else 1
            cost[left][right] = min(
                cost[left - 1][right] + 1,
                cost[left][right - 1] + 1,
                cost[left - 1][right - 1] + relabel,
            )
    return cost[-1][-1]
