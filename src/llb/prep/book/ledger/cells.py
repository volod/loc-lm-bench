"""Disputed table cells, classified and kept with the reading that produced each one.

A letter-only difference stays on the raw readings. A difference the ledger already
treats as dangerous, or that the substitution walk names, becomes a cell dispute.
"""

from collections.abc import Sequence
from itertools import combinations

from llb.core.contracts.data_prep.book_blocks import CellDispute, CellSource, RawReading
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
    rows = min(len(left.table.rows), len(right.table.rows))
    for row_index in range(rows):
        left_row = left.table.rows[row_index]
        right_row = right.table.rows[row_index]
        for column in range(min(len(left_row), len(right_row))):
            dispute = _cell(
                left, right, row_index, column, left_row[column].text, right_row[column].text
            )
            if dispute is None:
                continue
            key = _key(dispute)
            if key not in seen:
                seen.add(key)
                disputes.append(dispute)
    return disputes


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
