"""Reading the MARKDOWN tables in the delivered docs as data, so a swap can be costed against them.

A measured baseline in this repo lives in two places: the run artifact it came out of, and the table
in the delivered docs that publishes the row. The artifact is JSON and the table is prose, but the
table is the thing a reader trusts and the thing a generation swap makes wrong, so a report that
listed only artifacts would name the cheap half of the re-measurement cost.

Only tables that DECLARE a model column are read. A markdown table has no schema, so the header row
is the only statement of what its first column means, and a scan that searched every cell of every
table would resolve a model named in a prose comparison the same way it resolves a measured row.
The header is that declaration: `| model | ... |` says this table's rows are per model, and nothing
else here has to guess.
"""

from dataclasses import dataclass
from pathlib import Path
import re

# The header cells that declare a column of model identities. Kept explicit rather than fuzzy: a
# header this list does not name is a table whose rows are not per model, and adding one is a
# deliberate act with a doc in front of you.
MODEL_COLUMNS = ("model", "models", "served artifact", "tag")

_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_DIVIDER = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_BACKTICKED = re.compile(r"`([^`]+)`")


@dataclass(frozen=True)
class TableCell:
    """One model identity read out of one table row, with the line an operator would open."""

    path: Path
    line: int
    header_line: int
    column: str
    identity: str


def _cells(line: str) -> list[str]:
    found = _ROW.match(line)
    return [cell.strip() for cell in found.group(1).split("|")] if found else []


def _model_columns(header: list[str]) -> dict[int, str]:
    """Which columns of one header declare model identities, by position."""
    return {index: cell for index, cell in enumerate(header) if cell.casefold() in MODEL_COLUMNS}


def _row_identities(
    path: Path, line: int, header_line: int, cells: list[str], columns: dict[int, str]
) -> list[TableCell]:
    """Every backticked identity one row carries in its declared model columns.

    Backticked, because that is how this repo writes an artifact id in a table and how it writes
    everything else too: an un-ticked cell is a label (`Qwen 3.8, previous`), and reading it as an
    identity would resolve a note rather than a measurement.
    """
    return [
        TableCell(path=path, line=line, header_line=header_line, column=column, identity=identity)
        for index, column in columns.items()
        if index < len(cells)
        for identity in _BACKTICKED.findall(cells[index])
    ]


def read_model_cells(path: Path) -> list[TableCell]:
    """Every model identity every model-column table in one markdown file publishes."""
    lines = path.read_text(encoding="utf-8").splitlines()
    found: list[TableCell] = []
    for index, line in enumerate(lines):
        header = _cells(line)
        if not header or index + 1 >= len(lines) or not _DIVIDER.match(lines[index + 1]):
            continue
        columns = _model_columns(header)
        if not columns:
            continue
        for offset, row in enumerate(lines[index + 2 :], start=index + 3):
            cells = _cells(row)
            if not cells:
                break
            found.extend(_row_identities(path, offset, index + 1, cells, columns))
    return found


def read_doc_tree(root: Path) -> list[TableCell]:
    """Every model identity published by a model-column table anywhere under one docs tree."""
    return [cell for path in sorted(root.rglob("*.md")) for cell in read_model_cells(path)]
