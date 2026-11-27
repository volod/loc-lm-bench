"""Adoption check: characters a correction fixed, and characters it made wrong.

The character counts walk the alignment of the reference against the text before correction
and against the text after. A reference character is wrong when it is substituted or deleted.
An insertion the hypothesis adds is its own error. The class counts are the quality harness's
confusion walk: how many of each dangerous substitution disappeared, and how many appeared.
Script events are zero unless the caller supplies the formula-structure counts.
"""

from dataclasses import dataclass

from llb.prep.book.metrics.align import alignment
from llb.prep.book.metrics.confusion import confusion_counts
from llb.prep.book.metrics.constants import CONFUSION_CLASSES


@dataclass(frozen=True, slots=True)
class AdoptionReport:
    """One before/after reading against a reference. Both maps name every confusion class."""

    errors_fixed: int
    tokens_harmed: int
    classes_fixed: dict[str, int]
    classes_harmed: dict[str, int]


def adoption_report(
    reference: str,
    before: str,
    after: str,
    *,
    script_events_before: int = 0,
    script_events_after: int = 0,
) -> AdoptionReport:
    """Errors removed by `after`, and reference characters `after` newly gets wrong.

    `tokens_harmed` counts characters. A digit inside a numeral is the token that matters,
    and a word split would hide it.
    """
    before_wrong, before_inserted = _errors(reference, before)
    after_wrong, after_inserted = _errors(reference, after)
    fixed = len(before_wrong - after_wrong) + max(0, before_inserted - after_inserted)
    harmed = len(after_wrong - before_wrong) + max(0, after_inserted - before_inserted)
    classes_fixed, classes_harmed = _classes(
        reference, before, after, script_events_before, script_events_after
    )
    return AdoptionReport(fixed, harmed, classes_fixed, classes_harmed)


def _errors(reference: str, hypothesis: str) -> tuple[set[int], int]:
    wrong: set[int] = set()
    inserted = 0
    for op, left, _right in alignment(reference, hypothesis):
        if op == "eq":
            continue
        if op == "ins":
            inserted += 1
            continue
        wrong.add(left)
    return wrong, inserted


def _classes(
    reference: str,
    before: str,
    after: str,
    script_before: int,
    script_after: int,
) -> tuple[dict[str, int], dict[str, int]]:
    before_counts = confusion_counts(reference, before, script_before)
    after_counts = confusion_counts(reference, after, script_after)
    fixed = {name: max(0, before_counts[name] - after_counts[name]) for name in CONFUSION_CLASSES}
    harmed = {name: max(0, after_counts[name] - before_counts[name]) for name in CONFUSION_CLASSES}
    return fixed, harmed
