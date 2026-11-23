"""The reading string a character-error comparison sees.

Blocks contribute in reading order. A table contributes its cell labels, a figure contributes
nothing, and an equation number is appended when the block text does not already contain it.
Whitespace collapses to single spaces so a line break is not an error by itself.
"""

from collections.abc import Iterable, Sequence

from llb.prep.book.metrics.reference import RefBlock
from llb.prep.book.metrics.tables import Cell

_SILENT = frozenset({"figure", "ornament", "neighbour-page-bleed"})


def collapsed(text: str) -> str:
    """Trimmed text with internal whitespace collapsed."""
    return " ".join(text.split())


def block_reading(
    kind: str,
    text: str,
    equation_number: str | None,
    cells: Sequence[str] | None,
) -> str:
    """The characters one block adds to the page reading."""
    if kind in _SILENT:
        body = ""
    elif kind == "table" and cells is not None:
        body = " ".join(cells)
    else:
        body = text
    if equation_number and equation_number not in body:
        body = f"{body} {equation_number}".strip()
    return collapsed(body)


def reference_reading(block: RefBlock) -> str:
    """The reference block's reading, using its table when it has one."""
    cells = None
    if block.table is not None:
        cells = [cell.text for row in block.table.rows for cell in row]
    return block_reading(block.kind, block.text, block.equation_number, cells)


def page_reading(parts: Iterable[tuple[int, str]]) -> str:
    """Join ``(reading_order, text)`` parts, skipping empty ones."""
    ordered = [text for _, text in sorted(parts) if text]
    return " ".join(ordered)


def cell_labels(rows: Sequence[Sequence[Cell]]) -> list[str]:
    """Cell labels in row-major order."""
    return [cell.text for row in rows for cell in row]
