"""Admit a proposal only when every changed character is its declared class.

A rewrite is any diff outside that class, any class the guard does not list, or a diff wider
than the per-edit cap. Insertion and deletion are admitted only when that character is itself
the class. Visual agreement is required for every admitted class: the crop was sent, and the
proposal says the crop shows the replacement. Language alone never admits an edit.
"""

import re
from dataclasses import dataclass

from llb.core.contracts.data_prep.book_blocks import UncertaintyClass
from llb.core.contracts.data_prep.book_resolve import RefusalReason
from llb.prep.book.correct.constants import (
    ADMITTED_CLASSES,
    CLASS_BRACKET,
    CLASS_DECIMAL,
    CLASS_DIGIT,
    CLASS_EQUATION,
    CLASS_INDEX,
    CLASS_LATIN_GREEK,
    CLASS_SIGN,
    CLASS_UNIT,
    EDIT_BUDGET,
    MAX_CHANGES_PER_EDIT,
)
from llb.prep.book.correct.proposal import Proposal
from llb.prep.book.metrics.align import alignment
from llb.prep.book.metrics.confusion import is_script_character, step_class
from llb.prep.book.metrics.constants import (
    CLASS_BRACKET as METRIC_BRACKET,
)
from llb.prep.book.metrics.constants import (
    CLASS_DECIMAL as METRIC_DECIMAL,
)
from llb.prep.book.metrics.constants import (
    CLASS_FIVE_S,
    CLASS_MINUS,
    CLASS_ONE_LI,
    CLASS_ZERO_O,
)
from llb.prep.book.metrics.constants import (
    CLASS_LATIN_GREEK as METRIC_LATIN_GREEK,
)
from llb.prep.book.metrics.constants import (
    CLASS_UNIT as METRIC_UNIT,
)

_DIGITS = frozenset("0123456789")
_DASHES = frozenset("-\u2212\u2010\u2011\u2013\u2014")
_DIGIT_PAIRS = frozenset((CLASS_ZERO_O, CLASS_ONE_LI, CLASS_FIVE_S))
_EQUATION = re.compile(r"^\(\d{1,4}(?:[-\u2212\u2010\u2011\u2013\u2014]\d{1,4})+\)$")
_OPERATION = {
    CLASS_SIGN: CLASS_MINUS,
    CLASS_DECIMAL: METRIC_DECIMAL,
    CLASS_UNIT: METRIC_UNIT,
    CLASS_BRACKET: METRIC_BRACKET,
    CLASS_LATIN_GREEK: METRIC_LATIN_GREEK,
}

REASON_REWRITE: RefusalReason = "rewrite"
REASON_BUDGET: RefusalReason = "budget"
REASON_NO_VISUAL: RefusalReason = "no-visual"
REASON_UNSUPPORTED: RefusalReason = "unsupported"


@dataclass(frozen=True, slots=True)
class Refusal:
    """A proposal the guard will not apply. `raw_text` is the primary slice."""

    proposal: Proposal
    reason: RefusalReason
    raw_text: str


@dataclass(frozen=True, slots=True)
class Judgement:
    """Admitted proposals, in start order, and every refusal that names a real span."""

    admitted: tuple[Proposal, ...]
    refused: tuple[Refusal, ...]


def judge(text: str, proposals: list[Proposal], *, saw_crop: bool) -> Judgement:
    """Split proposals into the budget that may be applied and the refusals."""
    admitted: list[Proposal] = []
    refused: list[Refusal] = []
    ordered = sorted(proposals, key=lambda item: (item.start, item.end, item.replacement))
    for proposal in ordered:
        raw = _slice(text, proposal)
        if raw is None or raw == proposal.replacement:
            continue
        reason = _reason(raw, proposal, saw_crop)
        if reason is not None:
            refused.append(Refusal(proposal, reason, raw))
            continue
        if _overlaps(proposal, admitted):
            refused.append(Refusal(proposal, REASON_UNSUPPORTED, raw))
            continue
        admitted.append(proposal)
    kept = admitted[:EDIT_BUDGET]
    for extra in admitted[EDIT_BUDGET:]:
        refused.append(Refusal(extra, REASON_BUDGET, text[extra.start : extra.end]))
    return Judgement(tuple(kept), tuple(refused))


def marker_classes(confusion_class: str) -> tuple[UncertaintyClass, ...]:
    """The uncertainty class a refusal of this proposal shows on the marker."""
    if confusion_class in (CLASS_DIGIT, CLASS_EQUATION):
        return ("digit",)
    if confusion_class == CLASS_SIGN:
        return ("sign",)
    if confusion_class == CLASS_DECIMAL:
        return ("decimal",)
    return ("symbol",)


def _slice(text: str, proposal: Proposal) -> str | None:
    if proposal.start < 0 or proposal.end > len(text) or proposal.end < proposal.start:
        return None
    return text[proposal.start : proposal.end]


def _reason(raw: str, proposal: Proposal, saw_crop: bool) -> RefusalReason | None:
    fits = _fits(proposal.confusion_class, raw, proposal.replacement)
    if proposal.confusion_class not in ADMITTED_CLASSES or not fits:
        return REASON_REWRITE
    if not proposal.evidence.strip():
        return REASON_UNSUPPORTED
    if not saw_crop or not proposal.visual:
        return REASON_NO_VISUAL
    return None


def _fits(confusion_class: str, raw: str, replacement: str) -> bool:
    if confusion_class == CLASS_EQUATION:
        return _equation(raw, replacement)
    ops = _changes(raw, replacement)
    if not ops or len(ops) > MAX_CHANGES_PER_EDIT:
        return False
    if confusion_class == CLASS_DIGIT:
        return all(_digit(op, raw, replacement, left, right) for op, left, right in ops)
    if confusion_class == CLASS_INDEX:
        return all(_index(op, raw, replacement, left, right) for op, left, right in ops)
    expected = _OPERATION[confusion_class]
    return all(step_class(op, left, right, raw, replacement) == expected for op, left, right in ops)


def _changes(raw: str, replacement: str) -> list[tuple[str, int, int]]:
    return [(op, left, right) for op, left, right in alignment(raw, replacement) if op != "eq"]


def _digit(op: str, raw: str, replacement: str, left: int, right: int) -> bool:
    if op != "sub":
        return False
    kind = step_class(op, left, right, raw, replacement)
    if kind in _DIGIT_PAIRS:
        return True
    return raw[left] in _DIGITS and replacement[right] in _DIGITS


def _index(op: str, raw: str, replacement: str, left: int, right: int) -> bool:
    if op == "sub":
        return is_script_character(raw[left]) and is_script_character(replacement[right])
    char = raw[left] if op == "del" else replacement[right]
    return is_script_character(char)


def _equation(raw: str, replacement: str) -> bool:
    if raw == replacement:
        return False
    if _EQUATION.fullmatch(raw) is None or _EQUATION.fullmatch(replacement) is None:
        return False
    ops = _changes(raw, replacement)
    if not ops or len(ops) > MAX_CHANGES_PER_EDIT:
        return False
    return all(_equation_op(op, raw, replacement, left, right) for op, left, right in ops)


def _equation_op(op: str, raw: str, replacement: str, left: int, right: int) -> bool:
    if op == "sub":
        pair = {raw[left], replacement[right]}
        return pair <= _DIGITS or pair <= _DASHES
    char = raw[left] if op == "del" else replacement[right]
    return char in _DIGITS


def _overlaps(proposal: Proposal, kept: list[Proposal]) -> bool:
    return any(proposal.start < item.end and item.start < proposal.end for item in kept)
