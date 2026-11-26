"""Separate table failures. A similarity score hides which one happened.

Each name fires on its own planted change: a changed cell, a header that lost a
row or a column, an empty cell that disappeared, or a caption that does not match.
"""

from llb.core.contracts.data_prep.book_blocks import TableCell, TableReading

ALTERED_CELL = "altered-cell"
COLLAPSED_HEADER = "collapsed-header"
LOST_EMPTY_CELL = "lost-empty-cell"
WRONG_CAPTION = "wrong-caption"


def table_faults(reference: TableReading, hypothesis: TableReading) -> tuple[str, ...]:
    """The faults in ``hypothesis``, in a fixed order. Empty when the grids agree."""
    found: list[str] = []
    if _caption(reference.caption) != _caption(hypothesis.caption):
        found.append(WRONG_CAPTION)
    if _collapsed(reference, hypothesis):
        found.append(COLLAPSED_HEADER)
    if _lost_empty(reference, hypothesis):
        found.append(LOST_EMPTY_CELL)
    if _altered(reference, hypothesis):
        found.append(ALTERED_CELL)
    return tuple(found)


def _caption(caption: str | None) -> str:
    return " ".join((caption or "").split())


def _altered(reference: TableReading, hypothesis: TableReading) -> bool:
    if not _same_shape(reference, hypothesis):
        return False
    pairs = zip(_flat(reference), _flat(hypothesis), strict=True)
    return any(left.text != right.text for left, right in pairs)


def _collapsed(reference: TableReading, hypothesis: TableReading) -> bool:
    shorter = hypothesis.header_rows < reference.header_rows
    limit = min(reference.header_rows, len(reference.rows), len(hypothesis.rows))
    fewer = any(len(hypothesis.rows[index]) < len(reference.rows[index]) for index in range(limit))
    if not shorter and not fewer:
        return False
    return _rows_equal(
        reference.rows[reference.header_rows :], hypothesis.rows[hypothesis.header_rows :]
    )


def _lost_empty(reference: TableReading, hypothesis: TableReading) -> bool:
    if len(reference.rows) != len(hypothesis.rows):
        return False
    lost = False
    for left, right in zip(reference.rows, hypothesis.rows, strict=True):
        if _row_key(left) == _row_key(right):
            continue
        if not any(not cell.text.strip() for cell in left):
            return False
        if _without_empty(left) != _row_key(right):
            return False
        lost = True
    return lost


def _same_shape(reference: TableReading, hypothesis: TableReading) -> bool:
    if reference.header_rows != hypothesis.header_rows or len(reference.rows) != len(
        hypothesis.rows
    ):
        return False
    return all(
        _row_key(left) == _row_key(right) or _spans(left) == _spans(right)
        for left, right in zip(reference.rows, hypothesis.rows, strict=True)
    ) and all(
        len(left) == len(right) for left, right in zip(reference.rows, hypothesis.rows, strict=True)
    )


def _spans(row: list[TableCell]) -> list[tuple[int, int]]:
    return [(cell.colspan, cell.rowspan) for cell in row]


def _rows_equal(left: list[list[TableCell]], right: list[list[TableCell]]) -> bool:
    if len(left) != len(right):
        return False
    return all(_row_key(one) == _row_key(other) for one, other in zip(left, right, strict=True))


def _flat(table: TableReading) -> list[TableCell]:
    return [cell for row in table.rows for cell in row]


def _row_key(row: list[TableCell]) -> list[tuple[str, int, int]]:
    return [(cell.text, cell.colspan, cell.rowspan) for cell in row]


def _without_empty(row: list[TableCell]) -> list[tuple[str, int, int]]:
    return [(cell.text, cell.colspan, cell.rowspan) for cell in row if cell.text.strip()]
