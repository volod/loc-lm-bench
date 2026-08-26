"""What type an endpoint may carry: `domain`, `range`, `disjoint_types`.

These are the only classes that can be UNCHECKED. A fact endpoint the extractor never typed is the
`MISC` fact-only node `graph/build.py` creates; a type constraint has nothing to test there, and
calling that a pass or a failure would both be inventions. Absence of a type is absence of
evidence, so it gets its own column.
"""

from llb.prep.ontology.axioms.checks.base import Outcome, as_fact, violation
from llb.prep.ontology.axioms.ledger import Ledger, TypeAssertion
from llb.prep.ontology.axioms.models import Axiom

SUBJECT = "subject"
OBJECT = "object"


def check_type_constraint(axiom: Axiom, ledger: Ledger, key: str) -> Outcome:
    """Shared body of the domain and range classes."""
    outcome = Outcome()
    allowed = set(axiom.entity_types)
    for fact in ledger.relation_facts(axiom.relation or ""):
        endpoint = fact.subject if key == SUBJECT else fact.object
        assertions = ledger.types_of(endpoint)
        if not assertions:
            outcome.unchecked += 1
            continue
        outcome.checked += 1
        if any(a.entity_type in allowed for a in assertions):
            outcome.supporting += 1
            outcome.examples.append(as_fact(fact))
            continue
        outcome.violating += 1
        found = "/".join(sorted(a.entity_type for a in assertions))
        outcome.violations.append(
            violation(
                axiom,
                endpoint,
                f"{key} {endpoint!r} is typed {found} under {axiom.relation!r}, outside the "
                f"allowed {sorted(allowed)}",
                [as_fact(fact), assertions[0].as_fact()],
            )
        )
    return outcome


def check_domain(axiom: Axiom, ledger: Ledger) -> Outcome:
    """The subject's entity type must be in the relation's declared domain."""
    return check_type_constraint(axiom, ledger, SUBJECT)


def check_range(axiom: Axiom, ledger: Ledger) -> Outcome:
    """The object's entity type must be in the relation's declared range."""
    return check_type_constraint(axiom, ledger, OBJECT)


def _matching(assertions: list[TypeAssertion], pair: tuple[str, str]) -> list[TypeAssertion]:
    return [a for a in assertions if a.entity_type in pair]


def check_disjoint_types(axiom: Axiom, ledger: Ledger) -> Outcome:
    """One name cannot carry both types of a disjoint pair.

    The denominator is the names carrying at least one member of the pair -- the population the
    axiom could possibly apply to -- not every typed name in the corpus.
    """
    outcome = Outcome()
    left, right = axiom.entity_types[0], axiom.entity_types[1]
    for _key, assertions in ledger.typed_names():
        matched = _matching(assertions, (left, right))
        if not matched:
            continue
        outcome.checked += 1
        if len({a.entity_type for a in matched}) == 1:
            outcome.supporting += 1
            outcome.examples.append(matched[0].as_fact())
            continue
        outcome.violating += 1
        first = next(a for a in matched if a.entity_type == left)
        second = next(a for a in matched if a.entity_type == right)
        outcome.violations.append(
            violation(
                axiom,
                first.name,
                f"{first.name!r} is asserted as both {left} and {right}, which the axiom "
                "declares disjoint",
                [first.as_fact(), second.as_fact()],
            )
        )
    return outcome
