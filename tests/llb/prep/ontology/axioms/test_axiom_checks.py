"""Each axiom class finds its planted violation in the committed fixture -- and nothing else.

The fixture carries exactly one planted violation per class over 14 facts, so the assertions here
are exact counts rather than "at least one": an axiom that over-fires is as wrong as one that
misses, and only an exact count catches the first.
"""

from llb.prep.ontology.axioms.checker import check_ledger, collect_evidence
from llb.prep.ontology.axioms.constants import AXIOM_KINDS, N_AXIOM_EXAMPLES
from llb.prep.ontology.axioms.ledger import TYPE_RELATION, Ledger
from llb.prep.ontology.axioms.models import AxiomSet
from llb.prep.ontology.axioms.report import reading, render_report
from llb.prep.ontology.models import DocExtraction

PLANTED = {
    "func-diie": "functional",
    "inv-func-avtor": "inverse_functional",
    "domain-napysav": "domain",
    "range-diie": "range",
    "disjoint-person-org": "disjoint_types",
    "sym-mezhuie-z": "symmetric",
    "asym-mistyt": "asymmetric",
    "irrefl-ye": "irreflexive",
    "maxcard-ye": "max_cardinality",
}


def _check(axiom_set: AxiomSet, ledger: Ledger):
    return check_ledger(axiom_set, ledger, "fixture", "tests/fixtures/ontology")


def test_one_violation_per_axiom_class(axiom_set: AxiomSet, fixture_ledger: Ledger) -> None:
    report = _check(axiom_set, fixture_ledger).report
    assert {v.axiom_id for v in report.violations} == set(PLANTED)
    assert len(report.violations) == len(PLANTED)
    assert {v.kind for v in report.violations} == set(AXIOM_KINDS)


def test_every_violation_cites_grounded_spans(
    axiom_set: AxiomSet, fixture_ledger: Ledger, fixture_doc_text: str
) -> None:
    report = _check(axiom_set, fixture_ledger).report
    for violation in report.violations:
        assert violation.facts, violation.axiom_id
        for fact in violation.facts:
            span = fact.evidence
            assert fixture_doc_text[span.char_start : span.char_end] == span.text


def test_pairwise_classes_cite_both_offending_facts(
    axiom_set: AxiomSet, fixture_ledger: Ledger
) -> None:
    report = _check(axiom_set, fixture_ledger).report
    by_id = {v.axiom_id: v for v in report.violations}
    for axiom_id in ("func-diie", "inv-func-avtor", "asym-mistyt", "disjoint-person-org"):
        assert len(by_id[axiom_id].facts) == 2, axiom_id


def test_type_violations_cite_the_contradicting_type_assertion(
    axiom_set: AxiomSet, fixture_ledger: Ledger
) -> None:
    report = _check(axiom_set, fixture_ledger).report
    by_id = {v.axiom_id: v for v in report.violations}
    domain = by_id["domain-napysav"]
    assert [fact.relation for fact in domain.facts] == ["написав", TYPE_RELATION]
    assert domain.facts[1].object == "ORG"


def test_max_cardinality_cites_every_value_over_the_bound(
    axiom_set: AxiomSet, fixture_ledger: Ledger
) -> None:
    report = _check(axiom_set, fixture_ledger).report
    by_id = {v.axiom_id: v for v in report.violations}
    assert len(by_id["maxcard-ye"].facts) == 3


def test_an_untyped_endpoint_is_unchecked_not_a_violation(axiom_set: AxiomSet) -> None:
    # A fact-only endpoint has no asserted type, so a type constraint has nothing to test there.
    untyped = DocExtraction.model_validate(
        {
            "doc_id": "d.md",
            "entities": [],
            "facts": [
                {
                    "subject": "Патент",
                    "relation": "діє",
                    "object": "двадцять років",
                    "evidence": {
                        "doc_id": "d.md",
                        "char_start": 0,
                        "char_end": 6,
                        "text": "Патент",
                    },
                }
            ],
        }
    )
    report = _check(axiom_set, Ledger([untyped])).report
    stat = next(s for s in report.stats if s.axiom_id == "range-diie")
    assert (stat.checked, stat.violating, stat.unchecked) == (0, 0, 1)
    assert not report.violations


def test_a_zero_is_reported_as_a_finding_not_a_blank(
    axiom_set: AxiomSet, fixture_ledger: Ledger
) -> None:
    report = _check(axiom_set, fixture_ledger).report
    absent = next(s for s in report.stats if s.axiom_id == "func-maie-naselennia")
    held = next(s for s in report.stats if s.axiom_id == "irrefl-mistyt")
    assert reading(absent) == "did not apply here (no fact carries this relation)"
    assert reading(held) == f"held on all {held.checked} units"
    assert reading(absent) in render_report(_report(axiom_set, fixture_ledger))


def _report(axiom_set: AxiomSet, ledger: Ledger):
    from llb.prep.ontology.axioms.models import ValidationReport

    check = _check(axiom_set, ledger)
    return ValidationReport(
        axioms_source="samples/ontology/axioms_uk_v1.ttl",
        axioms_version=axiom_set.version,
        n_axioms=len(axiom_set.axioms),
        n_signed=len(axiom_set.signed),
        ledgers=[check.report],
        evidence=collect_evidence(axiom_set, [check]),
    )


def test_evidence_rows_carry_both_sides_for_the_sign_off_lane(
    axiom_set: AxiomSet, fixture_ledger: Ledger
) -> None:
    check = _check(axiom_set, fixture_ledger)
    rows = {row.axiom_id: row for row in collect_evidence(axiom_set, [check])}
    assert set(rows) == {axiom.axiom_id for axiom in axiom_set.axioms}
    functional = rows["func-diie"]
    assert functional.contradicting and functional.supporting
    assert len(functional.supporting) <= N_AXIOM_EXAMPLES
    assert functional.gloss and "owl:FunctionalProperty" in functional.turtle


def test_relation_matching_folds_case_and_whitespace(
    axiom_set: AxiomSet, fixture_extractions: list[DocExtraction]
) -> None:
    shouted = fixture_extractions[0].model_copy(deep=True)
    for fact in shouted.facts:
        fact.relation = f"  {fact.relation.upper()} "
    report = _check(axiom_set, Ledger([shouted])).report
    assert {v.axiom_id for v in report.violations} == set(PLANTED)
