"""What the answer gate returns for one answer, and how a lane names its setting.

The verdict is deliberately thin: the gate DECIDES nothing about the run, it reports which signed
axioms the declared triples broke and what evidence stands behind each. The status, the repair, and
the columns are decided one layer up, so the same verdict object serves the run-time lane, the
fixture harness, and the study.
"""

from dataclasses import dataclass, field

from llb.eval.answer_validation.constants import MAX_REPORTED_VIOLATIONS
from llb.prep.ontology.axioms.models import Violation


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """The ontology gate's reading of one declared answer.

    `checked_triples` is the population the verdict rests on: an envelope that declared no triple
    was not validated, it was UNCHECKABLE, and reporting those two as the same "passed" would let a
    model buy a clean gate by declining to type its claims.
    """

    violations: list[Violation] = field(default_factory=list)
    checked_claims: int = 0
    checked_triples: int = 0
    scoped_facts: int = 0

    @property
    def ok(self) -> bool:
        """No signed axiom the declared triples broke. An unchecked envelope is also `ok`."""
        return not self.violations

    @property
    def checkable(self) -> bool:
        """Whether the envelope declared anything the gate could test."""
        return self.checked_triples > 0

    @property
    def classes(self) -> list[str]:
        """The axiom CLASSES broken, sorted -- what the per-class adopt/reject verdict keys on."""
        return sorted({violation.kind for violation in self.violations})

    @property
    def axiom_ids(self) -> list[str]:
        """The individual axioms broken, sorted."""
        return sorted({violation.axiom_id for violation in self.violations})

    def detail(self) -> str:
        """One line naming the broken constraints, for the repair reprompt and the logs."""
        return "; ".join(
            f"{violation.axiom_id}: {violation.detail}"
            for violation in self.violations[:MAX_REPORTED_VIOLATIONS]
        )
