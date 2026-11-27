"""Spans a formula or a table holds open that no character uncertainty names.

The ledger's uncertainty spans align the readings' text. A table's cells and a formula's
symbols are compared on their own structure, so a disagreement there has to be carried to the
primary text before resolve can keep it out of the adopted reading.

- A disputed cell is found in the primary text, row by row, and those characters stay open. A
  cell the primary left blank, did not have, or whose text it does not show opens the whole
  reading: the pipeline abstains on that table rather than publish a grid it cannot ground.
- Two table captions that disagree on a digit, a sign, a decimal, or a symbol open the whole
  reading, because the caption is printed from the structure and is not in the text.
- Inside a formula a letter is a symbol. A letter another formula reading writes differently
  stays open, so `x_i` against `x_j` is not settled by whichever engine is primary.

A span the primary does not cover because another reading inserted it holds the characters on
either side of that point, as the uncertainty masks do.
"""

from difflib import SequenceMatcher

from llb.core.contracts.data_prep.book_blocks import (
    UNCERTAINTY_CLASSES,
    Block,
    RawReading,
    UncertaintyClass,
)
from llb.prep.book.grid import CellIndex, occupancy
from llb.prep.book.ledger.reconcile import mismatch_classes
from llb.prep.book.resolve.masks import insertion_range
from llb.prep.book.resolve.spans import ForcedSpan

_SYMBOL: tuple[UncertaintyClass, ...] = ("symbol",)


def technical_spans(block: Block, primary: RawReading) -> list[ForcedSpan]:
    """Open spans from disputed cells, contested captions, and formula letters."""
    spans: list[ForcedSpan] = []
    spans.extend(_disputed_cells(block, primary))
    spans.extend(_contested_caption(block, primary))
    spans.extend(_formula_letters(block, primary))
    return spans


def cell_offsets(text: str, table_rows: list[list[str]]) -> dict[CellIndex, tuple[int, int]]:
    """Where each non-blank cell sits in the reading's text, walking the grid row by row."""
    offsets: dict[CellIndex, tuple[int, int]] = {}
    cursor = 0
    for row_index, row in enumerate(table_rows):
        for cell_index, cell in enumerate(row):
            value = cell.strip()
            if not value:
                continue
            found = text.find(value, cursor)
            if found < 0:
                continue
            offsets[(row_index, cell_index)] = (found, found + len(value))
            cursor = found + len(value)
    return offsets


def _disputed_cells(block: Block, primary: RawReading) -> list[ForcedSpan]:
    if primary.table is None or not block.cell_disputes:
        return []
    cells = occupancy(primary.table)
    rows = [[cell.text for cell in row] for row in primary.table.rows]
    offsets = cell_offsets(primary.text, rows)
    spans: list[ForcedSpan] = []
    for dispute in block.cell_disputes:
        if not any(_is(primary, source.engine, source.variant) for source in dispute.sources):
            continue
        index = cells.get((dispute.row, dispute.column))
        located = offsets.get(index) if index is not None else None
        if located is None:
            return _whole(primary, tuple(dispute.classes))
        spans.append((located[0], located[1], tuple(dispute.classes)))
    return spans


def caption_contest(block: Block, primary: RawReading) -> tuple[UncertaintyClass, ...]:
    """The dangerous classes on which another table reading prints a different caption."""
    if primary.table is None or not primary.table.caption:
        return ()
    for other in block.readings:
        if other is primary or other.table is None or not other.table.caption:
            continue
        classes = _differing_classes(primary.table.caption, other.table.caption)
        if classes:
            return classes
    return ()


def _differing_classes(left: str, right: str) -> tuple[UncertaintyClass, ...]:
    """The dangerous classes of the slices two strings differ on, in the ledger's order."""
    found: set[UncertaintyClass] = set()
    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, start, end, other_start, other_end in matcher.get_opcodes():
        if tag != "equal":
            found.update(mismatch_classes(left[start:end], right[other_start:other_end]))
    return tuple(name for name in UNCERTAINTY_CLASSES if name in found)


def _contested_caption(block: Block, primary: RawReading) -> list[ForcedSpan]:
    classes = caption_contest(block, primary)
    return _whole(primary, classes) if classes else []


def _formula_letters(block: Block, primary: RawReading) -> list[ForcedSpan]:
    if block.kind != "formula" or primary.formula is None:
        return []
    spans: list[ForcedSpan] = []
    for other in block.readings:
        if other is not primary and other.formula is not None:
            spans.extend(_letter_spans(primary.text, other.text))
    return spans


def _letter_spans(primary: str, other: str) -> list[ForcedSpan]:
    """Primary characters two formula readings differ on where the difference holds a letter."""
    spans: list[ForcedSpan] = []
    matcher = SequenceMatcher(a=primary, b=other, autojunk=False)
    for tag, start, end, other_start, other_end in matcher.get_opcodes():
        blob = primary[start:end] + other[other_start:other_end]
        if tag == "equal" or not any(char.isalpha() for char in blob):
            continue
        held = (start, end) if end > start else insertion_range(len(primary), start)
        if held is not None:
            spans.append((held[0], held[1], _SYMBOL))
    return spans


def _whole(primary: RawReading, classes: tuple[UncertaintyClass, ...]) -> list[ForcedSpan]:
    if not primary.text:
        return []
    return [(0, len(primary.text), classes or _SYMBOL)]


def _is(reading: RawReading, engine: str, variant: str) -> bool:
    return reading.engine == engine and reading.variant == variant
