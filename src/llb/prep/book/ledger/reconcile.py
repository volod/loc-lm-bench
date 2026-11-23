"""Character alignment of the raw readings on one block.

A disagreement on a digit, a sign, a decimal point, or a symbol becomes an uncertainty span that
names every reading's slice. Nothing in this module picks a winner: two readings of `0` against
one of `O` still produce the span, and a letter-only disagreement is left on the raw readings
rather than voted into a third string.
"""

import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from itertools import combinations

from llb.core.contracts.data_prep.book_blocks import (
    UNCERTAINTY_CLASSES,
    RawReading,
    UncertaintyClass,
    UncertaintySlice,
    UncertaintySpan,
)

# Hyphen-minus, plus, minus sign, en dash, em dash.
SIGN_CHARACTERS = frozenset("-+\u2212\u2013\u2014")
_COMMA_DECIMAL = re.compile(r"\d,\d")


def reconcile(readings: Sequence[RawReading]) -> list[UncertaintySpan]:
    """Spans for every dangerous pairwise disagreement, in reading order of the pairs."""
    ordered = [item for _, item in sorted(enumerate(readings), key=_sort_key)]
    spans: list[UncertaintySpan] = []
    seen: set[tuple[object, ...]] = set()
    for left, right in combinations(ordered, 2):
        for span in _pair(left, right):
            key = _identity(span)
            if key not in seen:
                seen.add(key)
                spans.append(span)
    return spans


def _sort_key(item: tuple[int, RawReading]) -> tuple[str, str, int]:
    index, reading = item
    return (reading.engine, reading.variant, index)


def _pair(left: RawReading, right: RawReading) -> list[UncertaintySpan]:
    matcher = SequenceMatcher(a=left.text, b=right.text, autojunk=False)
    spans = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        classes = mismatch_classes(left.text[i1:i2], right.text[j1:j2])
        if not classes:
            continue
        spans.append(
            UncertaintySpan(
                classes=classes,
                slices=[_slice(left, i1, i2), _slice(right, j1, j2)],
            )
        )
    return spans


def _slice(reading: RawReading, start: int, end: int) -> UncertaintySlice:
    return UncertaintySlice(
        engine=reading.engine,
        variant=reading.variant,
        start=start,
        end=end,
        text=reading.text[start:end],
    )


def mismatch_classes(left: str, right: str) -> list[UncertaintyClass]:
    """The dangerous classes touched by this mismatched pair of slices, in a fixed order."""
    blob = left + right
    found: list[UncertaintyClass] = []
    if _digit(blob):
        found.append("digit")
    if any(char in SIGN_CHARACTERS for char in blob):
        found.append("sign")
    if _decimal(blob):
        found.append("decimal")
    if any(_symbol(char) for char in blob):
        found.append("symbol")
    return [name for name in UNCERTAINTY_CLASSES if name in found]


def _digit(blob: str) -> bool:
    return any(char.isdigit() for char in blob)


def _decimal(blob: str) -> bool:
    return "." in blob or _COMMA_DECIMAL.search(blob) is not None


def _symbol(char: str) -> bool:
    if char.isspace() or char in SIGN_CHARACTERS or char in ".,":
        return False
    return not (char.isascii() and char.isalnum())


def _identity(span: UncertaintySpan) -> tuple[object, ...]:
    slices = tuple(
        (item.engine, item.variant, item.start, item.end, item.text) for item in span.slices
    )
    return (tuple(span.classes), slices)
