"""Adopt-or-reject, per axiom class and per lane.

Two decisions, deliberately kept apart. A LANE decision asks whether the whole gate is worth its
cost on this corpus -- the objective delta on the commonly-answered items, read with the same
paired machinery every other adoption verdict in the repo uses. A CLASS decision asks the narrower
question the acceptance gate turns on: did THIS constraint catch more wrong answers than the
correct ones it refused, on enough items for the reading to hold?

A class that never fired is `not-measured`, never `adopt`. Absence of a rejection is absence of
evidence, and enabling a constraint on that basis is exactly what the sign-off boundary exists to
prevent; a class that fired and did not clear its own false rejections is recorded as
`measured-not-adopted`, which is a finding, not a gap.
"""

from collections.abc import Mapping
from typing import Any

from typing_extensions import TypedDict

from llb.eval import common as eval_common
from llb.eval.answer_validation.labelling import LABEL_FALSE_REJECTION, RefusalLabel
from llb.eval.paired_cases import CaseRows
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison, separates

DECISION_ADOPT = "adopt"
DECISION_RETAIN = "retain"
DECISION_NOT_ADOPTED = "measured-not-adopted"
DECISION_NOT_MEASURED = "not-measured"


class ClassVerdict(TypedDict):
    """Adopt-or-reject for ONE axiom class, on what it caught against what it refused wrongly."""

    axiom_class: str
    n_rejected: int
    n_catches: int
    n_false_rejections: int
    # The same two counts under the SHIPPED surface-token proxy alone, so the run that re-labels
    # its refusals reports the new reading BESIDE the old rather than in place of it.
    n_catches_contains: int
    n_false_rejections_contains: int
    catch_rate: float  # catches per case of the gated lane
    false_rejection_rate: float  # false rejections per case of the gated lane
    net: PairedComparison  # per-item (catch - false rejection), against a zero lane
    decision: str
    reason: str


def class_verdicts(
    rows: CaseRows,
    labels: Mapping[str, RefusalLabel],
    index_sets: list[list[int]],
    confidence: float,
) -> list[ClassVerdict]:
    """Adopt-or-reject per axiom class, decided on the gated lane's own rejections.

    A class that never fired is `not-measured`, not `adopted`: absence of a rejection is absence of
    evidence, and shipping a constraint on that basis is what the sign-off boundary exists to stop.

    `labels` is the one catch / false-rejection reading of each refusal
    (`llb.eval.answer_validation.labelling`), passed in rather than recomputed so this verdict and
    the refusal table beside it can never disagree about what a rejection was.
    """
    classes = sorted({name for row in rows for name in row.get("validation_classes", [])})
    return [_class_verdict(name, rows, labels, index_sets, confidence) for name in classes]


def _tally(values: list[float]) -> tuple[int, int]:
    """(catches, false rejections) over one class's per-case values."""
    return sum(1 for value in values if value > 0), sum(1 for value in values if value < 0)


def _class_verdict(
    name: str,
    rows: CaseRows,
    labels: Mapping[str, RefusalLabel],
    index_sets: list[list[int]],
    confidence: float,
) -> ClassVerdict:
    """One class's reading: what it caught, what it wrongly refused, and whether that clears zero.

    Both readings of the SAME rejections are carried -- the inflection-tolerant labels the verdict
    turns on, and what the shipped surface-token proxy alone gives -- so a run measured after the
    re-labelling stays comparable to one recorded before it.
    """
    net = [_class_value(row, name, labels) for row in rows]
    catches, false_rejections = _tally(net)
    shipped_catches, shipped_false = _tally(
        [_class_value(row, name, labels, shipped=True) for row in rows]
    )
    comparison = paired_comparison(net, [0.0] * len(net), index_sets, confidence)
    decision, reason = _class_decision(
        name, catches, false_rejections, separates(comparison, confidence)
    )
    return {
        "axiom_class": name,
        "n_rejected": catches + false_rejections,
        "n_catches": catches,
        "n_false_rejections": false_rejections,
        "n_catches_contains": shipped_catches,
        "n_false_rejections_contains": shipped_false,
        "catch_rate": round(catches / len(rows), 4) if rows else 0.0,
        "false_rejection_rate": round(false_rejections / len(rows), 4) if rows else 0.0,
        "net": comparison,
        "decision": decision,
        "reason": reason,
    }


def _class_value(
    row: Mapping[str, Any],
    name: str,
    labels: Mapping[str, RefusalLabel],
    *,
    shipped: bool = False,
) -> float:
    """+1 when this class caught a wrong answer, -1 when it refused a correct one, else 0.

    With `shipped`, the reading is the one the surface-token proxy alone gives -- the same number
    the artifact would have carried before the refusals were re-labelled.
    """
    if str(row.get("status")) != eval_common.ONTOLOGY_VIOLATION:
        return 0.0
    if name not in row.get("validation_classes", []):
        return 0.0
    label = labels.get(str(row.get("item_id")))
    if label is None:
        return 1.0
    correct = label.shipped_label == LABEL_FALSE_REJECTION if shipped else label.correct
    return -1.0 if correct else 1.0


def _class_decision(
    name: str, catches: int, false_rejections: int, separated: bool
) -> tuple[str, str]:
    if catches + false_rejections == 0:
        return DECISION_NOT_MEASURED, f"{name} refused nothing on this item set"
    if catches > false_rejections and separated:
        return (
            DECISION_ADOPT,
            f"{name} caught {catches} and wrongly refused {false_rejections}; the paired net "
            "clears zero",
        )
    return (
        DECISION_NOT_ADOPTED,
        f"{name} caught {catches} and wrongly refused {false_rejections}"
        + ("" if separated else "; the paired net does not clear zero"),
    )


def lane_decision(
    lane: str, objective: PairedComparison, separated: bool, regressed: bool
) -> tuple[str, str]:
    """The whole-gate reading: adopt only on a gain the paired interval puts clear of zero."""
    delta = objective["delta"]
    if separated:
        return DECISION_ADOPT, f"{lane} gains {delta['mean']:+.3f} objective, clear of zero"
    if regressed:
        return DECISION_RETAIN, f"{lane} loses {delta['mean']:+.3f} objective, clear of zero"
    return DECISION_RETAIN, f"{lane} moves the objective {delta['mean']:+.3f}, not clear of zero"
