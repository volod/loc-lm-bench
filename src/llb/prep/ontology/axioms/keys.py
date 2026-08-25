"""The one key that makes an in-repo violation and a reasoner entailment comparable.

Both sides find the same contradiction from different directions, so neither the surface spelling
of a name nor the order the two offending facts happen to arrive in can be part of the identity.
The key folds every endpoint with `normalize_name` and sorts the pairs.
"""

from llb.prep.ontology.axioms.models import Violation
from llb.prep.ontology.naming import normalize_name

PAIR_SEPARATOR = ">"
PAIRS_SEPARATOR = "|"


def comparison_key(axiom_id: str, pairs: list[tuple[str, str]]) -> str:
    """`<axiom id>:<folded endpoint pairs, sorted>` -- stable across both implementations."""
    folded = sorted(
        f"{normalize_name(left)}{PAIR_SEPARATOR}{normalize_name(right)}" for left, right in pairs
    )
    return f"{axiom_id}:{PAIRS_SEPARATOR.join(folded)}"


def pair_key(violation: Violation) -> str:
    """The same key computed from an in-repo violation's facts."""
    return comparison_key(
        violation.axiom_id, [(fact.subject, fact.object) for fact in violation.facts]
    )
