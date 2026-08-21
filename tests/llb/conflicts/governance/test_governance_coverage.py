"""The precondition behind a zero policy-choice delta: could this run have produced another one?

A zero delta has two opposite readings -- the corpus carries dated revisions the two policies agree
on, or it carries no governance dates at all and `superseded_by` could never have been derived --
and the delta alone cannot tell them apart. These tests pin the distinction end to end:

1. the counts (`document_coverage`, `document_pair_orderability`, `pair_orderability`) measure the
   three levels the precondition can be missing at, and `compare_editions` is the same orderability
   test detection promotes on -- the document-pair count is asserted against enumerating that
   function over every pair, because it is computed from key multisets instead;
2. the READING beside a zero delta follows the counts, and names the STAGE: an ingestion gap with
   nothing orderable anywhere, a retrieval miss when the corpus has orderable document pairs and
   the returned rows have none, evidence about the corpus's knowledge when a returned pair orders;
3. the delta itself never moves, on either corpus -- the coverage is a second reading beside it,
   not an input to it.

The two corpora are the acceptance gate: byte-identical bodies, one carrying `effective_date` front
matter and one carrying none, so the ONLY thing that differs between the two runs is the
precondition.
"""

import json
from itertools import combinations, combinations_with_replacement

import pytest

from llb.conflicts.audit import AuditParams, run_audit
from llb.conflicts.constants import REL_DUPLICATE, TIER_CLAIM, TIER_HASH
from llb.conflicts.governance.editions import compare_editions
from llb.conflicts.governance.coverage import (
    COVERAGE_SCHEMA_VERSION,
    coverage_reading,
    document_coverage,
    document_pair_orderability,
    governance_coverage,
    has_orderable_document_pair,
    has_orderable_pair,
    pair_orderability,
)
from llb.conflicts.models import AuditResult, ClaimRef, Finding
from llb.conflicts.resolution.projection import project_policies
from llb.conflicts.report.render import render_report, write_audit
from llb.conflicts.report.projection import coverage_sentence
from llb.conflicts.resolution.policy import POLICY_CONSERVATIVE, POLICY_PREFER_NEWER

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


def _revision_corpus(root):
    """A dated corpus whose orderable pair is NOT the pair the tiers return.

    Two byte-identical copies of one edition (the duplicate the hash tier finds, and two sides
    carrying the same date order no better than two undated ones) beside a third document holding a
    later edition of something else -- so the corpus supplies orderable document pairs and the
    returned row is not one of them.
    """
    root.mkdir(parents=True)
    for name in ("first.md", "second.md"):
        (root / name).write_text(f"---\neffective_date: 2024-01-01\n---\n{BODY}", encoding="utf-8")
    (root / "third.md").write_text(
        "---\neffective_date: 2026-01-01\n---\na later edition of another claim\n", encoding="utf-8"
    )
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
    ("governances", "pairs", "orderable"),
    [
        ([], 0, 0),
        ([{"effective_date": "2024-01-01"}], 0, 0),
        # Dated end to end and nothing to order: one edition recorded on both documents.
        ([{"effective_date": "2024-01-01"}] * 2, 1, 0),
        ([{"effective_date": "2021-01-01"}, {"effective_date": "2024-01-01"}], 1, 1),
        ([{"version": "v1"}, {"version": "v2"}], 1, 1),
        # A field each orders nothing: `compare_editions` needs the SAME field on both sides.
        ([{"effective_date": "2024-01-01"}, {"version": "v2"}], 1, 0),
        # The version fallback carries a pair the dates cannot.
        (
            [
                {"effective_date": "2024-01-01", "version": "v1"},
                {"effective_date": "2024-01-01", "version": "v2"},
            ],
            1,
            1,
        ),
        # Orderable BOTH ways is still one pair -- the inclusion-exclusion term, not a double count.
        (
            [
                {"effective_date": "2021-01-01", "version": "v1"},
                {"effective_date": "2024-01-01", "version": "v2"},
                {"effective_date": "2026-01-01", "version": "v3"},
            ],
            3,
            3,
        ),
        # Two copies of one edition beside a revision: 3 pairs, and the 2 crossing the revision.
        (
            [
                {"effective_date": "2024-01-01"},
                {"effective_date": "2024-01-01"},
                {"effective_date": "2026-01-01"},
            ],
            3,
            2,
        ),
        ([{"language": "uk"}, {}], 1, 0),
    ],
)
def test_document_pair_orderability_counts_what_the_corpus_itself_could_have_supplied(
    governances, pairs, orderable
):
    """The middle count: measured on the corpus alone, with no candidate list and no store."""
    coverage = document_pair_orderability(governances)

    assert coverage["document_pairs"] == pairs
    assert coverage["orderable_document_pairs"] == orderable
    assert has_orderable_document_pair(coverage) is bool(orderable)


_GOVERNANCE_POOL = (
    {},
    {"language": "uk"},
    {"effective_date": "2024-01-01"},
    {"effective_date": "2021-05-05"},
    {"version": "v1"},
    {"version": "v2"},
    {"effective_date": "2024-01-01", "version": "v1"},
    {"effective_date": "2021-05-05", "version": "v2"},
    {"effective_date": "not-a-date", "version": "v1"},
)


@pytest.mark.parametrize("size", [2, 3, 4])
def test_the_document_pair_count_equals_enumerating_compare_editions_over_every_pair(size):
    """The count is derived from key multisets, so it is pinned against the quadratic truth.

    Every corpus of `size` documents drawable from a pool covering each way the fields can be
    present, absent, blank, shared, or unparseable -- the count must agree with the function it is
    a precondition for on all of them, or the audit is measuring a different orderability.
    """
    for corpus in combinations_with_replacement(_GOVERNANCE_POOL, size):
        enumerated = sum(
            1
            for left, right in combinations(corpus, 2)
            if compare_editions(left, right).newer_side is not None
        )
        coverage = document_pair_orderability(list(corpus))

        assert coverage["document_pairs"] == size * (size - 1) // 2
        assert coverage["orderable_document_pairs"] == enumerated, corpus


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


def test_a_zero_delta_with_nothing_orderable_anywhere_reads_as_an_ingestion_gap():
    """The failure mode: an operator told the choice is free by a run that could not have differed."""
    undated = _projected(
        [_row(1, a_governance={}, b_governance={})],
        governance_coverage([{}, {}], [{"a": {}, "b": {}}]),
    )
    report = render_report(undated)

    assert "**no difference on this corpus**" in report
    assert "0 of 2 documents with `effective_date` or `version`" in report
    assert "0 of 1 document pair and 0 of 1 returned pair orderable by `compare_editions`" in report
    assert "the zero above is STRUCTURAL" in report
    assert "Fixable at INGESTION" in report
    assert "RETRIEVAL" not in report


def test_a_zero_delta_on_a_dated_corpus_whose_rows_order_none_names_retrieval_instead():
    """The reading this count exists for: the same structural zero, the opposite fix.

    The corpus carries a revision `compare_editions` orders, and the pair the audit RETURNED is two
    copies of one edition -- so nothing about the ingestion is wrong and re-ingesting would change
    nothing. The stage that lost the orderable pair is the one that chose the candidates.
    """
    one_edition = {"effective_date": "2024-01-01"}
    revision = {"effective_date": "2026-01-01"}
    findings = [_row(1, a_governance=one_edition, b_governance=one_edition)]
    rows = AuditResult(effort=TIER_CLAIM, corpus_root="c", n_docs=3, findings=findings).rows()
    report = render_report(
        _projected(findings, governance_coverage([one_edition, one_edition, revision], rows))
    )

    assert "**no difference on this corpus**" in report
    assert "3 of 3 documents with `effective_date` or `version`" in report
    assert "2 of 3 document pairs and 0 of 1 returned pair orderable" in report
    assert "the stage that lost the orderable pair is RETRIEVAL, not ingestion" in report
    assert "Fixable at INGESTION" not in report


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


# --- the acceptance gate: three corpora, different coverage, the same delta -----------------------


def test_three_corpora_read_as_three_stages_and_the_delta_never_moves(tmp_path):
    """The gate the document-pair count exists for: one zero delta, three different fixes.

    All three audits return a hash-tier duplicate, which resolves the same way under either policy,
    so all three deltas are zero and a report printing only the delta would say "the choice is free
    here" three times. The coverage separates them by the STAGE the orderable pair was lost at.
    """
    dated = _audit(_corpus(tmp_path / "dated", dated=True))
    undated = _audit(_corpus(tmp_path / "undated", dated=False))
    revision = _audit(_revision_corpus(tmp_path / "revision"))
    results = (dated, undated, revision)

    coverage = revision.governance_coverage
    assert coverage["dated_documents"] == 3, "dated end to end, and audited at the same effort"
    assert (coverage["orderable_document_pairs"], coverage["document_pairs"]) == (2, 3)
    assert (coverage["orderable_pairs"], coverage["returned_pairs"]) == (0, 1)
    assert undated.governance_coverage["orderable_document_pairs"] == 0
    assert dated.governance_coverage["orderable_document_pairs"] == 1

    deltas = [project_policies(result.rows(), list(POLICY_PAIR))["deltas"] for result in results]
    assert deltas[0] == deltas[1] == deltas[2], "the coverage reads beside the delta, never into it"
    assert deltas[0][0]["moved_rows"] == 0 and deltas[0][0]["review_rows"] == 0

    knowledge, ingestion, retrieval = (
        coverage_reading(result.governance_coverage, zero_delta=True) for result in results
    )
    assert "KNOWLEDGE" in knowledge and "STRUCTURAL" not in knowledge
    assert "Fixable at INGESTION" in ingestion and "RETRIEVAL" not in ingestion
    assert "RETRIEVAL, not ingestion" in retrieval and "INGESTION" not in retrieval


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
