"""What every axiom check returns, and the two shapes every one of them builds.

The population an axiom was evaluated over is what makes its result a BASE RATE rather than a
count: "3 violations" says nothing until you know whether the axiom applied to 4 subjects or 4000.
So every check reports `checked` beside `violating`, and `unchecked` -- a unit with no asserted
type to test -- stays its own column and is never counted as either.
"""

from dataclasses import dataclass, field

from llb.prep.ontology.axioms.models import Axiom, Violation, ViolationFact
from llb.prep.ontology.models import SROFact


@dataclass
class Outcome:
    """What one axiom found on one ledger."""

    checked: int = 0
    supporting: int = 0
    violating: int = 0
    unchecked: int = 0
    violations: list[Violation] = field(default_factory=list)
    examples: list[ViolationFact] = field(default_factory=list)


def as_fact(fact: SROFact) -> ViolationFact:
    """Render an SRO fact in the violation shape, keeping its exact evidence span."""
    return ViolationFact(
        subject=fact.subject,
        relation=fact.relation,
        object=fact.object,
        evidence=fact.evidence,
    )


def violation(axiom: Axiom, subject: str, detail: str, facts: list[ViolationFact]) -> Violation:
    """One broken axiom, naming every fact a reviewer needs to adjudicate it."""
    return Violation(
        axiom_id=axiom.axiom_id,
        kind=axiom.kind,
        relation=axiom.relation,
        subject=subject,
        detail=detail,
        facts=facts,
    )
