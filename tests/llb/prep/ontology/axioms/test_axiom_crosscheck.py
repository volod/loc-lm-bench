"""The OWL reasoner must reach the in-repo checker's verdict on the committed fixture.

Marked `heavy_env`: it needs the optional `[ontology]` extra (rdflib + owlrl), which GitHub CI's
base install does not carry. A disagreement here is a bug in the pure-Python checker -- the
reasoner is the reference for what an OWL construct means, never the other way round.
"""

import pytest

from llb.prep.ontology.axioms.checker import check_ledger
from llb.prep.ontology.axioms.constants import REASONER_KINDS
from llb.prep.ontology.axioms.crosscheck import crosscheck_report
from llb.prep.ontology.axioms.ledger import Ledger
from llb.prep.ontology.axioms.models import AxiomSet, ValidationReport

pytestmark = pytest.mark.heavy_env


def _report(axiom_set: AxiomSet, ledger: Ledger) -> ValidationReport:
    check = check_ledger(axiom_set, ledger, "fixture", "tests/fixtures/ontology")
    return ValidationReport(
        axioms_source="samples/ontology/axioms_uk_v1.ttl",
        axioms_version=axiom_set.version,
        n_axioms=len(axiom_set.axioms),
        n_signed=0,
        ledgers=[check.report],
    )


def test_reasoner_and_checker_report_the_same_violations(
    axiom_set: AxiomSet, fixture_ledger: Ledger
) -> None:
    result = crosscheck_report(axiom_set, [fixture_ledger], _report(axiom_set, fixture_ledger))
    assert result.ran
    assert result.checker_only == []
    assert result.reasoner_only == []
    assert result.agrees


def test_the_cross_check_covers_the_inconsistency_condition_classes(
    axiom_set: AxiomSet, fixture_ledger: Ledger
) -> None:
    result = crosscheck_report(axiom_set, [fixture_ledger], _report(axiom_set, fixture_ledger))
    assert set(result.kinds) == set(REASONER_KINDS)


def test_a_checker_that_misses_a_violation_is_caught(
    axiom_set: AxiomSet, fixture_ledger: Ledger
) -> None:
    # Drop one confirmed violation from the checker's side; the reasoner must still find it.
    report = _report(axiom_set, fixture_ledger)
    report.ledgers[0].violations = [
        v for v in report.ledgers[0].violations if v.axiom_id != "func-diie"
    ]
    result = crosscheck_report(axiom_set, [fixture_ledger], report)
    assert result.reasoner_only and not result.agrees


def test_the_committed_turtle_parses_under_a_real_rdf_library() -> None:
    from rdflib import Graph

    from tests.llb.prep.ontology.axioms.conftest import CANDIDATE_TURTLE

    assert len(Graph().parse(source=str(CANDIDATE_TURTLE), format="turtle")) > 0
