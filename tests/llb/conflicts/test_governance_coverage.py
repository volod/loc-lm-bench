"""The precondition behind a zero policy-choice delta: could this run have produced another one?

A zero delta has two opposite readings -- the corpus carries dated revisions the two policies agree
on, or it carries no governance dates at all and `superseded_by` could never have been derived --
and the delta alone cannot tell them apart. These tests pin the distinction end to end:

1. the counts (`document_coverage`, `pair_orderability`) measure the two levels the precondition
   can be missing at, and `compare_editions` is the same orderability test detection promotes on;
2. the READING beside a zero delta follows the counts: structural with no orderable pair, evidence
   about the corpus's knowledge with one;
3. the delta itself never moves, on either corpus -- the coverage is a second reading beside it,
   not an input to it.

The two corpora are the acceptance gate: byte-identical bodies, one carrying `effective_date` front
matter and one carrying none, so the ONLY thing that differs between the two runs is the
precondition.
"""

import json

import pytest

from llb.conflicts.audit import AuditParams, run_audit
from llb.conflicts.constants import REL_DUPLICATE, TIER_CLAIM, TIER_HASH
from llb.conflicts.governance_coverage import (
    COVERAGE_SCHEMA_VERSION,
    coverage_reading,
    document_coverage,
    governance_coverage,
    has_orderable_pair,
    pair_orderability,
)
from llb.conflicts.models import AuditResult, ClaimRef, Finding
from llb.conflicts.policy_projection import project_policies
from llb.conflicts.report import render_report, write_audit
from llb.conflicts.report_projection import coverage_sentence
from llb.conflicts.resolution_policy import POLICY_CONSERVATIVE, POLICY_PREFER_NEWER

BODY = "the same claim stated once and ingested twice\n"
POLICY_PAIR = (POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)


def _row(index: int, *, a_governance: dict, b_governance: dict) -> Finding:
    return Finding(
        relation=REL_DUPLICATE,
        tier=TIER_CLAIM,
        a=ClaimRef("left.md", 0, 10, "a" * 10, chunk_id="left.md#0001", governance=a_governance),
        b=ClaimRef(
            "right.md",
            index * 100,
            index * 100 + 10,
            "b" * 10,
            chunk_id=f"right.md#{index:04d}",
            governance=b_governance,
        ),
        score=0.9,
        evidence="model",
    )


def _projected(findings: list[Finding], coverage: dict) -> AuditResult:
    """One audit result carrying both readings: the policy delta and its precondition."""
    result = AuditResult(effort=TIER_CLAIM, corpus_root="corpus", n_docs=2, findings=findings)
    result.policy_projection = project_policies(result.rows(), list(POLICY_PAIR))
    result.governance_coverage = coverage
    return result


def _corpus(root, *, dated: bool):
    """Two documents with identical bodies -- a hash-tier duplicate pair, dated or not."""
    root.mkdir(parents=True)
    for index, name in enumerate(("first.md", "second.md")):
        front = f"---\neffective_date: 202{index + 3}-01-01\n---\n" if dated else ""
        (root / name).write_text(front + BODY, encoding="utf-8")
    return root


def _audit(root):
    return run_audit(root, AuditParams(effort=TIER_HASH))


# --- the counts ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("governance", "dated", "by_field"),
    [
        ({"effective_date": "2024-01-01"}, 1, {"effective_date": 1, "version": 0}),
        ({"version": "v2"}, 1, {"effective_date": 0, "version": 1}),
        ({"effective_date": "2024-01-01", "version": "v2"}, 1, {"effective_date": 1, "version": 1}),
        ({"language": "uk"}, 0, {"effective_date": 0, "version": 0}),
        ({"effective_date": "   ", "version": None}, 0, {"effective_date": 0, "version": 0}),
    ],
)
def test_document_coverage_counts_the_fields_an_edition_comparison_can_use(
    governance, dated, by_field
):
    """Per field as well as in total: a corpus with `version` everywhere is orderable too."""
    coverage = document_coverage([governance])

    assert coverage["documents"] == 1
    assert coverage["dated_documents"] == dated
    assert coverage["documents_by_field"] == by_field


@pytest.mark.parametrize(
    ("left", "right", "orderable"),
    [
        ({"effective_date": "2021-01-01"}, {"effective_date": "2024-01-01"}, 1),
        ({"version": "v1"}, {"version": "v2"}, 1),
        # Dated on both sides and still not orderable: one edition, seen twice.
        ({"effective_date": "2024-01-01"}, {"effective_date": "2024-01-01"}, 0),
        # A date on one side orders nothing -- which is why the DOCUMENT count is not enough.
        ({"effective_date": "2024-01-01"}, {}, 0),
        ({}, {}, 0),
    ],
)
def test_a_pair_is_orderable_only_when_both_sides_carry_a_field_that_differs(
    left, right, orderable
):
    """The stricter count, and the one the reading turns on: what `compare_editions` can order."""
    result = AuditResult(
        effort=TIER_CLAIM,
        corpus_root="corpus",
        n_docs=2,
        findings=[_row(1, a_governance=left, b_governance=right)],
    )
    coverage = pair_orderability(result.rows())

    assert coverage["returned_pairs"] == 1
    assert coverage["orderable_pairs"] == orderable
    assert coverage["orderable_share"] == float(orderable)
    assert has_orderable_pair(coverage) is bool(orderable)


def test_a_run_that_returned_nothing_has_no_share_rather_than_a_zero_one():
    """`None` and `0.0` are opposite answers, the same distinction `moved_share` draws."""
    empty = pair_orderability([])

    assert (empty["returned_pairs"], empty["orderable_pairs"]) == (0, 0)
    assert empty["orderable_share"] is None
    assert not has_orderable_pair(empty)


# --- the reading beside the delta ----------------------------------------------------------------


def test_a_zero_delta_with_no_orderable_pair_reads_as_structural():
    """The failure mode: an operator told the choice is free by a run that could not have differed."""
    undated = _projected(
        [_row(1, a_governance={}, b_governance={})],
        governance_coverage([{}, {}], [{"a": {}, "b": {}}]),
    )
    report = render_report(undated)

    assert "**no difference on this corpus**" in report
    assert "0 of 2 documents with `effective_date` or `version`" in report
    assert "0 of 1 returned pair orderable by `compare_editions`" in report
    assert "the zero above is STRUCTURAL" in report
    assert "Fixable at INGESTION" in report


def test_a_zero_delta_with_orderable_pairs_present_reads_as_the_policies_agreeing():
    """The other reading: dated pairs were returned and both policies settled them the same way."""
    older, newer = {"effective_date": "2021-01-01"}, {"effective_date": "2024-01-01"}
    findings = [_row(1, a_governance=older, b_governance=newer)]
    rows = AuditResult(effort=TIER_CLAIM, corpus_root="c", n_docs=2, findings=findings).rows()
    report = render_report(_projected(findings, governance_coverage([older, newer], rows)))

    assert "**no difference on this corpus**" in report
    assert "2 of 2 documents with `effective_date` or `version`" in report
    assert "1 of 1 returned pair orderable by `compare_editions`" in report
    assert "the zero above is about this corpus's KNOWLEDGE" in report
    assert "STRUCTURAL" not in report


def test_the_coverage_prints_only_beside_a_delta():
    """One policy is no choice, so there is no zero to explain -- and no flag prints nothing."""
    rows = [_row(1, a_governance={}, b_governance={})]
    coverage = governance_coverage([{}, {}], [{"a": {}, "b": {}}])
    result = AuditResult(effort=TIER_CLAIM, corpus_root="corpus", n_docs=2, findings=rows)
    result.governance_coverage = coverage

    assert "governance coverage" not in render_report(result), "no projection, no precondition"
    result.policy_projection = project_policies(result.rows(), [POLICY_CONSERVATIVE])
    assert "governance coverage" not in render_report(result), "one policy is not a choice"
    result.policy_projection = project_policies(result.rows(), list(POLICY_PAIR))
    assert "governance coverage" in render_report(result)


def test_the_report_and_the_cli_quote_one_precondition_through_one_helper():
    """Both surfaces prefix the same sentence, so they cannot disagree about which zero it is."""
    findings = [_row(1, a_governance={}, b_governance={})]
    coverage = governance_coverage([{}, {}], [{"a": {}, "b": {}}])
    result = _projected(findings, coverage)
    sentence = coverage_sentence(result.policy_projection, coverage)

    assert f"- {sentence}" in render_report(result).splitlines()
    assert coverage_sentence(project_policies(result.rows(), [POLICY_CONSERVATIVE]), coverage) == ""
    assert coverage_sentence(result.policy_projection, {}) == "", "nothing measured, nothing said"


def test_a_non_zero_delta_carries_the_counts_without_a_zero_reading():
    """The counts still ride with a delta that moved rows; only the ZERO needs explaining."""
    reading = coverage_reading(governance_coverage([{}], []), zero_delta=False)

    assert reading.startswith("governance coverage: ")
    assert "STRUCTURAL" not in reading and "KNOWLEDGE" not in reading


# --- the acceptance gate: two corpora, different coverage, the same delta -------------------------


def test_two_corpora_differing_only_in_dates_report_different_coverage_and_the_same_delta(tmp_path):
    """The gate: identical bodies, one corpus dated and one not, audited the same way.

    The delta is zero on both -- a hash-tier duplicate resolves the same way under either policy --
    so a report that printed only the delta would say "the choice is free here" twice and mean two
    different things. The coverage is what separates them, and it must not disturb the delta.
    """
    dated = _audit(_corpus(tmp_path / "dated", dated=True))
    undated = _audit(_corpus(tmp_path / "undated", dated=False))

    assert dated.governance_coverage["schema_version"] == COVERAGE_SCHEMA_VERSION
    assert dated.governance_coverage["dated_documents"] == 2
    assert undated.governance_coverage["dated_documents"] == 0
    assert (
        dated.governance_coverage["orderable_pairs"],
        dated.governance_coverage["returned_pairs"],
    ) == (1, 1)
    assert undated.governance_coverage["orderable_pairs"] == 0
    assert undated.governance_coverage["orderable_share"] == 0.0, "returned a pair, ordered none"

    deltas = [
        project_policies(result.rows(), list(POLICY_PAIR))["deltas"] for result in (dated, undated)
    ]
    assert deltas[0] == deltas[1], "the coverage is a reading beside the delta, never an input"
    assert deltas[0][0]["moved_rows"] == 0 and deltas[0][0]["review_rows"] == 0

    readings = [
        coverage_reading(result.governance_coverage, zero_delta=True) for result in (dated, undated)
    ]
    assert "KNOWLEDGE" in readings[0] and "STRUCTURAL" in readings[1]


def test_the_coverage_reaches_summary_json_on_every_run(tmp_path):
    """Recorded whether or not a policy was named: it is detection-side and costs nothing."""
    result = _audit(_corpus(tmp_path / "dated", dated=True))
    paths = write_audit(tmp_path / "audit", result)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

    assert summary["governance_coverage"] == result.governance_coverage
    assert "policy_projection" not in summary, "no policy was named; the precondition still is"
