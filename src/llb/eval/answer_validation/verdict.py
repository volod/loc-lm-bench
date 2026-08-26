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
from llb.eval.answer_validation.constants import REFERENCE_CORRECT_COLUMN
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
    catch_rate: float  # catches per case of the gated lane
    false_rejection_rate: float  # false rejections per case of the gated lane
    net: PairedComparison  # per-item (catch - false rejection), against a zero lane
    decision: str
    reason: str


def class_verdicts(
    rows: CaseRows, index_sets: list[list[int]], confidence: float
) -> list[ClassVerdict]:
    """Adopt-or-reject per axiom class, decided on the gated lane's own rejections.

    A class that never fired is `not-measured`, not `adopted`: absence of a rejection is absence of
    evidence, and shipping a constraint on that basis is what the sign-off boundary exists to stop.
    """
    classes = sorted({name for row in rows for name in row.get("validation_classes", [])})
    verdicts: list[ClassVerdict] = []
    for name in classes:
        net = [_class_value(row, name) for row in rows]
        catches = sum(1 for value in net if value > 0)
        false_rejections = sum(1 for value in net if value < 0)
        comparison = paired_comparison(net, [0.0] * len(net), index_sets, confidence)
        decision, reason = _class_decision(
            name, catches, false_rejections, separates(comparison, confidence)
        )
        verdicts.append(
            {
                "axiom_class": name,
                "n_rejected": catches + false_rejections,
                "n_catches": catches,
                "n_false_rejections": false_rejections,
                "catch_rate": round(catches / len(rows), 4) if rows else 0.0,
                "false_rejection_rate": round(false_rejections / len(rows), 4) if rows else 0.0,
                "net": comparison,
                "decision": decision,
                "reason": reason,
            }
        )
    return verdicts


def _class_value(row: Mapping[str, Any], name: str) -> float:
    """+1 when this class caught a wrong answer, -1 when it refused a correct one, else 0."""
    if str(row.get("status")) != eval_common.ONTOLOGY_VIOLATION:
        return 0.0
    if name not in row.get("validation_classes", []):
        return 0.0
    return -1.0 if float(row.get(REFERENCE_CORRECT_COLUMN, 0.0) or 0.0) >= 1.0 else 1.0


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
