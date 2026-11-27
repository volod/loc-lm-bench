"""Formula and table structure recovered from an engine's native output.

The ledger keeps this grid. Rendering decides Markdown, HTML, or the page crop.
Nothing here derives, converts, or calculates a formula.
"""

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from llb.core.contracts.data_prep.book_blocks import FormulaReading, TableCell, TableReading

if TYPE_CHECKING:
    from llb.prep.book.metrics.tables import Cell

_TAG = re.compile(r"\\tag\s*\*?\s*\{([^{}]*)\}")
_DISPLAY_WRAP = re.compile(r"^\\\[([\s\S]*)\\\]$")
_LEAD = re.compile(r"^\((?P<num>\d{1,4}(?:[-\u2013\u2014]\d{1,4})*)\)\s+")
_TRAIL = re.compile(r"\s+\((?P<num>\d{1,4}(?:[-\u2013\u2014]\d{1,4})*)\)\s*$")
_LEAD_PAREN = re.compile(r"^\((?P<num>[^()]*)\)\s+")
_TAIL_TEXT = re.compile(r"\\text\s*\{\s*\((?P<num>[^()]*)\)\s*\}\s*$")
_TAIL_PAREN = re.compile(r"\((?P<num>[^()]*)\)\s*$")
_SPACING = re.compile(r"\\(?:quad|qquad|,|;|:|!| )")
_NUMBER = re.compile(r"\d{1,4}(?:-\d{1,4})*[A-Za-z]?")
_SLOT = re.compile(r"(?:\\quad|&)\s*$")
_TRAILING_BREAK = re.compile(r"(?:\s*\\\\)+\s*$")
_CAPTION = re.compile(r"^Table\b", re.IGNORECASE)


def formula_from_text(text: str, *, display: bool) -> FormulaReading | None:
    """A formula reading. A trailing or leading parenthesized number is the equation number."""
    body = _unwrap_display(_TRAILING_BREAK.sub("", text.strip()).strip())
    if not body:
        return None
    number, side, body = _number(body)
    if not body:
        return None
    return FormulaReading(latex=body, display=display, equation_number=number, number_side=side)


def looks_like_caption(text: str) -> bool:
    """True when a line is a printed table caption."""
    return _CAPTION.match(text.strip()) is not None


def table_from_text(text: str, caption: str | None = None) -> TableReading | None:
    """A Markdown or HTML table. None when the text does not hold a grid."""
    from llb.prep.book.metrics.tables import parse_table_text

    parsed = parse_table_text(text)
    if parsed is None:
        return None
    rows = [[_from_cell(cell) for cell in row] for row in parsed]
    if "<table" in text.lower():
        headers = _html_header_rows(text)
    else:
        rows, headers = _repeated_header(rows)
    return _finish(rows, headers, caption)


def table_from_grid(
    rows: Sequence[Sequence[object]], caption: str | None = None, *, header_rows: int | None = None
) -> TableReading | None:
    """Rows of engine cells. Dict cells may name spans and ``column_header``."""
    built: list[list[TableCell]] = []
    raw_rows: list[Sequence[object]] = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            continue
        cells = [_engine_cell(cell) for cell in row]
        if cells:
            built.append(cells)
            raw_rows.append(row)
    if not built:
        return None
    headers = header_rows if header_rows is not None else _flagged_headers(raw_rows)
    return _finish(built, headers, caption)


def table_from_flat(cells: Sequence[object], caption: str | None = None) -> TableReading | None:
    """Docling's flat cell list, placed by each cell's start row and column."""
    placed = [cell for cell in cells if isinstance(cell, dict)]
    if not placed:
        return None
    buckets: dict[int, list[dict[str, object]]] = {}
    for cell in placed:
        buckets.setdefault(_int(cell.get("start_row_offset_idx")), []).append(cell)
    ordered: list[list[object]] = []
    for index in range(max(buckets) + 1):
        row = sorted(
            buckets.get(index, []), key=lambda item: _int(item.get("start_col_offset_idx"))
        )
        if row:
            ordered.append(list(row))
    return table_from_grid(ordered, caption)


Position = tuple[int, int]
CellIndex = tuple[int, int]


def occupancy(table: TableReading) -> dict[Position, CellIndex]:
    """Each expanded grid position, mapped to the ``(row, index in row)`` of the cell covering it."""
    taken: dict[Position, CellIndex] = {}
    for row_index, row in enumerate(table.rows):
        column = 0
        for cell_index, cell in enumerate(row):
            while (row_index, column) in taken:
                column += 1
            for down in range(cell.rowspan):
                for across in range(cell.colspan):
                    taken.setdefault((row_index + down, column + across), (row_index, cell_index))
            column += cell.colspan
    return taken


def html_holds(table: TableReading) -> bool:
    """False when a span overlaps another cell or a cell nests a table."""
    if _nests_table(table):
        return False
    width, height = _span_size(table)
    occupied = [[False] * width for _ in range(height)]
    for row_index, row in enumerate(table.rows):
        if not _place_row(occupied, row_index, row):
            return False
    return True


def _nests_table(table: TableReading) -> bool:
    return any("<table" in cell.text.lower() for row in table.rows for cell in row)


def _span_size(table: TableReading) -> tuple[int, int]:
    width = max(sum(cell.colspan for cell in row) for row in table.rows)
    height = len(table.rows)
    for index, row in enumerate(table.rows):
        for cell in row:
            height = max(height, index + cell.rowspan)
    return width, height


def _place_row(occupied: list[list[bool]], row_index: int, row: list[TableCell]) -> bool:
    column = 0
    for cell in row:
        column = _free_column(occupied, row_index, column)
        if not _claim(occupied, row_index, column, cell.rowspan, cell.colspan):
            return False
        column += cell.colspan
    return True


def _free_column(occupied: list[list[bool]], row: int, start: int) -> int:
    column = start
    width = len(occupied[row])
    while column < width and occupied[row][column]:
        column += 1
    return column


def _claim(occupied: list[list[bool]], row: int, column: int, rowspan: int, colspan: int) -> bool:
    if column + colspan > len(occupied[0]):
        return False
    cells = [(row + down, column + across) for down in range(rowspan) for across in range(colspan)]
    if any(occupied[y][x] for y, x in cells):
        return False
    for y, x in cells:
        occupied[y][x] = True
    return True


def cells_of(table: TableReading) -> "tuple[tuple[Cell, ...], ...]":
    """The scoring grid for one native table."""
    from llb.prep.book.metrics.tables import Cell

    return tuple(
        tuple(Cell(cell.text, cell.colspan, cell.rowspan) for cell in row) for row in table.rows
    )


def _unwrap_display(body: str) -> str:
    """Drop a display-math wrapper. The equation number lives inside it."""
    wrapped = _DISPLAY_WRAP.match(body)
    if wrapped is not None:
        return wrapped.group(1).strip()
    if body.startswith("$$") and body.endswith("$$") and len(body) > 4:
        return body[2:-2].strip()
    return body


def _repeated_header(rows: list[list[TableCell]]) -> tuple[list[list[TableCell]], int]:
    """Two header rows whose corners repeat, with a colspan where a label repeats sideways.

    A Markdown export of a merged header has no separator line. The corner labels are
    copied into both rows, and the spanning label is written once per column it covers.
    """
    matched = _matching_header(rows)
    if matched is None:
        return rows, 1
    top, nxt = matched
    vertical = _vertical_header(top, nxt)
    collapsed = _span_header(top, vertical)
    kept = [cell for cell, down in zip(nxt, vertical) if not down]
    return [collapsed, kept, *rows[2:]], 2


def _matching_header(rows: list[list[TableCell]]) -> tuple[list[TableCell], list[TableCell]] | None:
    """The two header rows when their corners repeat and one label spans sideways."""
    if len(rows) < 2:
        return None
    top = list(rows[0])
    nxt = list(rows[1])
    if len(top) != len(nxt) or len(top) < 2:
        return None
    if top[0].text != nxt[0].text or top[-1].text != nxt[-1].text:
        return None
    if not _sideways_repeat(top):
        return None
    return top, nxt


def _sideways_repeat(top: list[TableCell]) -> bool:
    return any(
        top[index].text and top[index].text == top[index + 1].text for index in range(len(top) - 1)
    )


def _vertical_header(top: Sequence[TableCell], nxt: Sequence[TableCell]) -> list[bool]:
    return [
        bool(top[index].text) and top[index].text == nxt[index].text for index in range(len(top))
    ]


def _span_header(top: Sequence[TableCell], vertical: Sequence[bool]) -> list[TableCell]:
    """Rowspan on a repeated corner, colspan on a label written once per column."""
    collapsed: list[TableCell] = []
    index = 0
    while index < len(top):
        if vertical[index]:
            collapsed.append(top[index].model_copy(update={"rowspan": 2}))
            index += 1
            continue
        end = _colspan_end(top, vertical, index)
        collapsed.append(top[index].model_copy(update={"colspan": end - index}))
        index = end
    return collapsed


def _colspan_end(top: Sequence[TableCell], vertical: Sequence[bool], index: int) -> int:
    end = index + 1
    while end < len(top) and _same_span(top, vertical, index, end):
        end += 1
    return end


def _same_span(top: Sequence[TableCell], vertical: Sequence[bool], index: int, end: int) -> bool:
    return bool(not vertical[end] and top[end].text == top[index].text and top[index].text)


def _number(body: str) -> tuple[str | None, Literal["left", "right"] | None, str]:
    tag = _TAG.search(body)
    if tag:
        number = tag.group(1).strip() or None
        return number, "right" if number else None, _TAG.sub("", body).strip()
    lead = _LEAD.match(body)
    if lead:
        return lead.group("num"), "left", body[lead.end() :].strip()
    trail = _TRAIL.search(body)
    if trail and trail.start() > 0:
        return trail.group("num"), "right", body[: trail.start()].strip()
    spaced = _spaced(body)
    if spaced is not None:
        return spaced
    return None, None, body


def _spaced(body: str) -> tuple[str, Literal["left", "right"], str] | None:
    """A printed number whose digits the engine separated, at either end of the formula."""
    lead = _LEAD_PAREN.match(body)
    if lead is not None:
        number = _compact_number(lead.group("num"))
        rest = body[lead.end() :].strip()
        if number and rest:
            return number, "left", rest
    tail = _TAIL_TEXT.search(body) or _TAIL_PAREN.search(body)
    if tail is None or tail.start() == 0:
        return None
    number = _compact_number(tail.group("num"))
    if number is None and _SLOT.search(body[: tail.start()]):
        number = _compact_number(tail.group("num"), marks=True)
    if not number:
        return None
    return number, "right", body[: tail.start()].strip()


def _compact_number(raw: str, *, marks: bool = False) -> str | None:
    """Digits of an equation number. Spacing commands and brace-wrapped dashes collapse.

    ``marks`` also reads ``\\cdot`` and ``\\div`` as the dash of a number that sits in the
    equation-number slot. A product in the formula body is left alone.
    """
    cleaned = _SPACING.sub("", raw)
    cleaned = cleaned.replace(" ", "").replace("{", "").replace("}", "")
    cleaned = cleaned.translate({0x2013: "-", 0x2014: "-"})
    if marks:
        cleaned = cleaned.replace(r"\cdot", "-").replace(r"\div", "-")
    if _NUMBER.fullmatch(cleaned):
        return cleaned
    return None


def _from_cell(cell: "Cell") -> TableCell:
    return TableCell(text=cell.text, colspan=cell.colspan, rowspan=cell.rowspan)


def _engine_cell(cell: object) -> TableCell:
    if not isinstance(cell, dict):
        return TableCell(text="" if cell is None else str(cell))
    return TableCell(
        text=str(cell.get("text") or ""),
        colspan=max(1, _int(cell.get("col_span", cell.get("colspan")), 1)),
        rowspan=max(1, _int(cell.get("row_span", cell.get("rowspan")), 1)),
    )


def _flagged_headers(rows: Sequence[Sequence[object]]) -> int:
    count = 0
    for row in rows:
        flags = [
            bool(cell.get("column_header"))
            for cell in row
            if isinstance(cell, dict) and "column_header" in cell
        ]
        if not flags or not all(flags):
            break
        count += 1
    return count or 1


def _html_header_rows(text: str) -> int:
    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", text, flags=re.IGNORECASE | re.DOTALL)
    count = 0
    for row in rows:
        cells = re.findall(r"<t([dh])\b", row, flags=re.IGNORECASE)
        if cells and all(tag.lower() == "h" for tag in cells):
            count += 1
            continue
        break
    if count > 1:
        return count
    if rows and _merged_header(rows[0]):
        return 2
    return count or 1


def _merged_header(row: str) -> bool:
    """A first row of data cells that spans is the top of a two-row header."""
    attrs = re.findall(r"<td\b([^>]*)>", row, flags=re.IGNORECASE)
    if len(attrs) < 2:
        return False
    return any(_span_at_least(attr, name) for attr in attrs for name in ("rowspan", "colspan"))


def _span_at_least(attrs: str, name: str) -> bool:
    found = re.search(rf"{name}\s*=\s*['\"]?(\d+)", attrs, flags=re.IGNORECASE)
    return found is not None and int(found.group(1)) >= 2


def _finish(rows: list[list[TableCell]], header_rows: int, caption: str | None) -> TableReading:
    cleaned = " ".join(caption.split()) if caption and caption.strip() else None
    table = TableReading(rows=rows, header_rows=header_rows, caption=cleaned, faithful=True)
    if html_holds(table):
        return table
    return table.model_copy(update={"faithful": False})


def _int(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)
