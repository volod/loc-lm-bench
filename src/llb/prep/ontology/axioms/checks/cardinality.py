"""How many values one endpoint may carry: `functional`, `inverse_functional`, `max_cardinality`.

All three ask the same question of a different side and a different bound, so they share one body.
A bound of 1 is the PAIRWISE case -- every conflicting pair of values is one decision a reviewer
takes, so it emits one violation per pair with both facts' spans. A bound above 1 is a single
group decision, so it emits one violation carrying every value over the bound.
"""

from itertools import combinations

from llb.prep.ontology.axioms.checks.base import Outcome, as_fact, violation
from llb.prep.ontology.axioms.ledger import Ledger
from llb.prep.ontology.axioms.models import Axiom, Violation
from llb.prep.ontology.models import SROFact
from llb.prep.ontology.naming import normalize_name

SUBJECT = "subject"
OBJECT = "object"


def _group(facts: list[SROFact], key: str) -> dict[str, list[SROFact]]:
    """Group facts by the folded subject or object, preserving ledger order."""
    groups: dict[str, list[SROFact]] = {}
    for fact in facts:
        value = fact.subject if key == SUBJECT else fact.object
        groups.setdefault(normalize_name(value), []).append(fact)
    return groups


def _distinct(facts: list[SROFact], key: str) -> list[SROFact]:
    """One representative fact per distinct value of the other endpoint, in ledger order."""
    seen: dict[str, SROFact] = {}
    for fact in facts:
        value = fact.object if key == SUBJECT else fact.subject
        seen.setdefault(normalize_name(value), fact)
    return list(seen.values())


def _pairwise(
    axiom: Axiom, anchor: str, labels: tuple[str, str], distinct: list[SROFact]
) -> list[Violation]:
    anchor_label, value_label = labels
    return [
        violation(
            axiom,
            anchor,
            f"{anchor_label} {anchor!r} carries two different {value_label}s "
            f"under {axiom.relation!r}",
            [as_fact(left), as_fact(right)],
        )
        for left, right in combinations(distinct, 2)
    ]


def check_cardinality(axiom: Axiom, ledger: Ledger, key: str, bound: int) -> Outcome:
    """Shared body of the functional, inverse-functional, and max-cardinality classes."""
    outcome = Outcome()
    labels = (SUBJECT, OBJECT) if key == SUBJECT else (OBJECT, SUBJECT)
    for group in _group(ledger.relation_facts(axiom.relation or ""), key).values():
        outcome.checked += 1
        distinct = _distinct(group, key)
        anchor = group[0].subject if key == SUBJECT else group[0].object
        if len(distinct) <= bound:
            outcome.supporting += 1
            outcome.examples.append(as_fact(group[0]))
            continue
        outcome.violating += 1
        if bound == 1:
            outcome.violations += _pairwise(axiom, anchor, labels, distinct)
            continue
        outcome.violations.append(
            violation(
                axiom,
                anchor,
                f"{labels[0]} {anchor!r} carries {len(distinct)} {labels[1]}s under "
                f"{axiom.relation!r}, above the bound of {bound}",
                [as_fact(fact) for fact in distinct],
            )
        )
    return outcome


def check_functional(axiom: Axiom, ledger: Ledger) -> Outcome:
    """At most one object per subject."""
    return check_cardinality(axiom, ledger, SUBJECT, 1)


def check_inverse_functional(axiom: Axiom, ledger: Ledger) -> Outcome:
    """At most one subject per object."""
    return check_cardinality(axiom, ledger, OBJECT, 1)


def check_max_cardinality(axiom: Axiom, ledger: Ledger) -> Outcome:
    """At most N objects per subject."""
    return check_cardinality(axiom, ledger, SUBJECT, axiom.max_count or 1)
