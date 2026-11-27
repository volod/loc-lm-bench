"""Where every reading agrees, and where an uncertainty span touches the primary reading.

Agreement is the characters every reading shares with the primary. A single reading has nothing
to agree with. An uncertainty span stays on the primary characters its own slice names; a
disagreement between two other readings does not punch a hole in a primary that was not part of it.
A primary slice of no width is a character another reading has and the primary lacks, such as the
second `0` of `0.05` against `0.5`. It holds the primary characters on either side of that point,
so a value the primary shortened is not adopted as read.
"""

from collections.abc import Sequence
from difflib import SequenceMatcher

from llb.core.contracts.data_prep.book_blocks import (
    UNCERTAINTY_CLASSES,
    RawReading,
    UncertaintyClass,
    UncertaintySlice,
    UncertaintySpan,
)


def agreed_mask(primary: RawReading, readings: Sequence[RawReading]) -> list[bool]:
    """True where every other reading equals the primary. One reading agrees with nothing."""
    if len(readings) < 2:
        return [False] * len(primary.text)
    mask = [True] * len(primary.text)
    skipped = False
    for other in readings:
        if other is primary and not skipped:
            skipped = True
            continue
        matcher = SequenceMatcher(a=primary.text, b=other.text, autojunk=False)
        for tag, start, end, _other_start, _other_end in matcher.get_opcodes():
            if tag == "equal":
                continue
            for index in range(start, end):
                mask[index] = False
    return mask


def uncertain_mask(primary: RawReading, spans: Sequence[UncertaintySpan]) -> list[bool]:
    """True on primary characters named by an uncertainty slice of this reading."""
    mask = [False] * len(primary.text)
    for span in spans:
        for sliver in span.slices:
            held = _held(primary, sliver)
            if held is None:
                continue
            for index in range(*held):
                mask[index] = True
    return mask


def classes_covering(
    primary: RawReading, spans: Sequence[UncertaintySpan], start: int, end: int
) -> list[UncertaintyClass]:
    """Uncertainty classes of every primary slice that overlaps `[start, end)`, in fixed order."""
    found: list[UncertaintyClass] = []
    for span in spans:
        for sliver in span.slices:
            held = _held(primary, sliver)
            if held is None or held[1] <= start or held[0] >= end:
                continue
            found.extend(span.classes)
    return [name for name in UNCERTAINTY_CLASSES if name in found]


def insertion_range(length: int, point: int) -> tuple[int, int] | None:
    """The primary characters on either side of a point where another reading has more."""
    start = max(0, point - 1)
    end = min(length, point + 1)
    if end <= start:
        return None
    return start, end


def _held(primary: RawReading, sliver: UncertaintySlice) -> tuple[int, int] | None:
    """The primary characters a slice of this reading holds open."""
    if not _slice_is_primary(primary, sliver):
        return None
    if sliver.end > sliver.start:
        return sliver.start, sliver.end
    return insertion_range(len(primary.text), sliver.start)


def _slice_is_primary(primary: RawReading, sliver: UncertaintySlice) -> bool:
    if sliver.engine != primary.engine or sliver.variant != primary.variant:
        return False
    if sliver.start < 0 or sliver.end > len(primary.text) or sliver.end < sliver.start:
        return False
    return primary.text[sliver.start : sliver.end] == sliver.text
