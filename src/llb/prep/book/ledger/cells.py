"""Disputed table cells, classified and kept with the reading that produced each one.

A letter-only difference stays on the raw readings. A difference the ledger already
treats as dangerous, or that the substitution walk names, becomes a cell dispute.

Two grids are compared position by position after spans are expanded, so a merged cell
is compared with every cell it covers rather than shifting the rest of its row. A
position only one grid fills is compared with a blank: a column one engine omitted is a
dispute wherever the other engine read a digit, a sign, or a symbol there. ``row`` and
``column`` are that expanded position.
"""

from collections.abc import Sequence
from itertools import combinations

from llb.core.contracts.data_prep.book_blocks import (
    CellDispute,
    CellSource,
    RawReading,
    TableReading,
)
from llb.prep.book.grid import CellIndex, occupancy
from llb.prep.book.ledger.reconcile import mismatch_classes
from llb.prep.book.metrics.align import alignment
from llb.prep.book.metrics.confusion import (
    CLASS_EQUATION,
    equation_numbers,
    normalize_equation,
    step_class,
)


def cell_disputes(readings: Sequence[RawReading]) -> list[CellDispute]:
    """One dispute per dangerous cell disagreement, in pair order."""
    structured = [reading for reading in readings if reading.table is not None]
    found: list[CellDispute] = []
    seen: set[tuple[object, ...]] = set()
    for left, right in combinations(structured, 2):
        found.extend(_pair(left, right, seen))
    return found


def _pair(left: RawReading, right: RawReading, seen: set[tuple[object, ...]]) -> list[CellDispute]:
    disputes: list[CellDispute] = []
    assert left.table is not None and right.table is not None
    left_cells = occupancy(left.table)
    right_cells = occupancy(right.table)
    compared: set[tuple[CellIndex | None, CellIndex | None]] = set()
    matched: set[tuple[int, CellIndex]] = set()
    for position in sorted(set(left_cells) | set(right_cells)):
        pair = (left_cells.get(position), right_cells.get(position))
        if pair in compared:
            continue
        compared.add(pair)
        one = _text(left.table, pair[0])
        other = _text(right.table, pair[1])
        if one == other:
            matched.update(_sides(pair))
            continue
        if _merge_only(pair, one, other, matched):
            continue
        dispute = _cell(left, right, position[0], position[1], one, other)
        if dispute is None:
            continue
        key = _key(dispute)
        if key not in seen:
            seen.add(key)
            disputes.append(dispute)
    return disputes


def _sides(pair: tuple[CellIndex | None, CellIndex | None]) -> set[tuple[int, CellIndex]]:
    return {(side, index) for side, index in enumerate(pair) if index is not None}


def _merge_only(
    pair: tuple[CellIndex | None, CellIndex | None],
    one: str,
    other: str,
    matched: set[tuple[int, CellIndex]],
) -> bool:
    """A merged cell whose text already matched, against the blank the other grid left beside it."""
    left, right = pair
    if not other and left is not None and (0, left) in matched:
        return True
    return not one and right is not None and (1, right) in matched


def _text(table: TableReading, index: CellIndex | None) -> str:
    if index is None:
        return ""
    row, cell = index
    return table.rows[row][cell].text


def _cell(
    left: RawReading, right: RawReading, row: int, column: int, one: str, other: str
) -> CellDispute | None:
    if one == other:
        return None
    classes = mismatch_classes(one, other)
    substitutions = _substitutions(one, other)
    if not classes and not substitutions:
        return None
    if not classes:
        classes = ["symbol"]
    return CellDispute(
        row=row,
        column=column,
        classes=classes,
        substitutions=substitutions,
        sources=[
            CellSource(engine=left.engine, variant=left.variant, text=one),
            CellSource(engine=right.engine, variant=right.variant, text=other),
        ],
    )


def _substitutions(left: str, right: str) -> list[str]:
    names: list[str] = []
    left_numbers = equation_numbers(left.strip())
    right_numbers = equation_numbers(right.strip())
    if left_numbers == [left.strip()] and right_numbers == [right.strip()]:
        if normalize_equation(left) != normalize_equation(right):
            names.append(CLASS_EQUATION)
    for op, left_index, right_index in alignment(left, right):
        name = step_class(op, left_index, right_index, left, right)
        if name and name not in names:
            names.append(name)
    return names


def _key(dispute: CellDispute) -> tuple[object, ...]:
    sources = tuple((item.engine, item.variant, item.text) for item in dispute.sources)
    return (
        dispute.row,
        dispute.column,
        tuple(dispute.classes),
        tuple(dispute.substitutions),
        sources,
    )
