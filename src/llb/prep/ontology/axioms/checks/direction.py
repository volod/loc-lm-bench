"""Which way a relation may run: `symmetric`, `asymmetric`, `irreflexive`.

All three read the relation's directed edge set, so they share `_directed`. Self-loops are excluded
from the symmetric and asymmetric populations because they are irreflexivity's business -- counting
one fact under two classes would double-report the same problem.
"""

from llb.prep.ontology.axioms.checks.base import Outcome, as_fact, violation
from llb.prep.ontology.axioms.ledger import Ledger
from llb.prep.ontology.axioms.models import Axiom
from llb.prep.ontology.models import SROFact
from llb.prep.ontology.naming import normalize_name


def _directed(facts: list[SROFact]) -> dict[tuple[str, str], SROFact]:
    """First fact per directed endpoint pair, self-loops excluded."""
    edges: dict[tuple[str, str], SROFact] = {}
    for fact in facts:
        src, dst = normalize_name(fact.subject), normalize_name(fact.object)
        if src != dst:
            edges.setdefault((src, dst), fact)
    return edges


def check_irreflexive(axiom: Axiom, ledger: Ledger) -> Outcome:
    """Nothing stands in this relation to itself."""
    outcome = Outcome()
    for fact in ledger.relation_facts(axiom.relation or ""):
        outcome.checked += 1
        if normalize_name(fact.subject) != normalize_name(fact.object):
            outcome.supporting += 1
            outcome.examples.append(as_fact(fact))
            continue
        outcome.violating += 1
        outcome.violations.append(
            violation(
                axiom,
                fact.subject,
                f"{fact.subject!r} stands in {axiom.relation!r} to itself",
                [as_fact(fact)],
            )
        )
    return outcome


def check_symmetric(axiom: Axiom, ledger: Ledger) -> Outcome:
    """An asserted edge needs its counterpart; a one-way assertion is a gap in the ledger."""
    outcome = Outcome()
    edges = _directed(ledger.relation_facts(axiom.relation or ""))
    for (src, dst), fact in edges.items():
        outcome.checked += 1
        if (dst, src) in edges:
            outcome.supporting += 1
            outcome.examples.append(as_fact(fact))
            continue
        outcome.violating += 1
        outcome.violations.append(
            violation(
                axiom,
                fact.subject,
                f"{axiom.relation!r} is symmetric, but the ledger asserts only "
                f"{fact.subject!r} -> {fact.object!r}",
                [as_fact(fact)],
            )
        )
    return outcome


def check_asymmetric(axiom: Axiom, ledger: Ledger) -> Outcome:
    """Both directions cannot hold between two distinct endpoints."""
    outcome = Outcome()
    edges = _directed(ledger.relation_facts(axiom.relation or ""))
    seen: set[tuple[str, str]] = set()
    for (src, dst), fact in edges.items():
        pair = (src, dst) if src < dst else (dst, src)
        if pair in seen:
            continue
        seen.add(pair)
        outcome.checked += 1
        mirror = edges.get((dst, src))
        if mirror is None:
            outcome.supporting += 1
            outcome.examples.append(as_fact(fact))
            continue
        outcome.violating += 1
        outcome.violations.append(
            violation(
                axiom,
                fact.subject,
                f"{axiom.relation!r} is asymmetric, but the ledger asserts it in both "
                f"directions between {fact.subject!r} and {fact.object!r}",
                [as_fact(fact), as_fact(mirror)],
            )
        )
    return outcome
