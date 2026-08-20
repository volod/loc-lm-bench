"""One definition of "actionable", shared by what the audit COUNTS and what it READS first.

The claim-tier precision block calls every non-`complementary` verdict a row an operator must act
on. If the report's ordering promotes a narrower set, the audit quotes a precision figure over rows
it then buries: a `subsumed_by` row sorts below any coexisting fact the model happened to score
higher, which is the exact reading a precision figure is published to support.

Because the ordering also writes `findings.jsonl`, these tests pin the other half of the contract:
the resolution lane must resolve the same rows to the same overlay whatever order it reads them in.
"""

import json

from llb.conflicts.grouping.census import finding_sort_key
from llb.conflicts.claim.precision import AdjudicatedRow
from llb.conflicts.constants import (
    REL_COMPLEMENTARY,
    REL_CONTRADICTS,
    REL_DUPLICATE,
    REL_SUBSUMED_BY,
    REL_SUBSUMES,
    REL_SUPERSEDED_BY,
    RELATIONS,
    TIER_CLAIM,
    is_actionable,
)
from llb.conflicts.models import AuditResult, ClaimRef, Finding
from llb.conflicts.resolution.overlay import overlay_from_plan
from llb.conflicts.report.render import render_report, write_audit
from llb.conflicts.resolution.policy import POLICY_CONSERVATIVE, build_plan

ACTIONABLE_RELATIONS = tuple(relation for relation in RELATIONS if relation != REL_COMPLEMENTARY)


def _finding(relation: str, score: float, index: int = 0) -> Finding:
    return Finding(
        relation=relation,
        tier=TIER_CLAIM,
        a=ClaimRef("a.md", index * 20, index * 20 + 10, "a" * 10, chunk_id=f"a.md#{index}"),
        b=ClaimRef("b.md", index * 20, index * 20 + 10, "b" * 10, chunk_id=f"b.md#{index}"),
        score=score,
        evidence="model",
    )


def _row(relation: str) -> AdjudicatedRow:
    return AdjudicatedRow(
        rank=1, left_key="a", right_key="b", score=0.5, relation=relation, parsed=True
    )


def test_every_relation_but_complementary_is_work_to_do():
    assert [relation for relation in RELATIONS if is_actionable(relation)] == list(
        ACTIONABLE_RELATIONS
    )
    assert not is_actionable(REL_COMPLEMENTARY) and not is_actionable(None)
    assert is_actionable("something_the_vocabulary_does_not_know"), (
        "an unrecognized verdict is someone's problem, not a coexisting fact"
    )


def test_the_precision_block_and_the_ordering_agree_relation_for_relation():
    for relation in RELATIONS:
        counted = _row(relation).actionable
        read_first = finding_sort_key(_finding(relation, 0.5))[0] == 0
        assert counted == read_first, f"{relation} is counted and read differently"


def test_a_decision_row_leads_a_higher_scored_coexisting_row():
    """The measured case: a `subsumed_by` row under complementary rows scored 1.000."""
    findings = [
        _finding(REL_COMPLEMENTARY, 1.0, index=0),
        _finding(REL_COMPLEMENTARY, 1.0, index=1),
        _finding(REL_SUBSUMED_BY, 0.95, index=2),
    ]
    assert [f.relation for f in sorted(findings, key=finding_sort_key)][0] == REL_SUBSUMED_BY


def test_the_report_and_the_findings_file_lead_with_the_same_row(tmp_path):
    findings = [
        _finding(REL_COMPLEMENTARY, 1.0, index=0),
        _finding(REL_SUBSUMED_BY, 0.95, index=1),
        _finding(REL_DUPLICATE, 0.90, index=2),
        _finding(REL_CONTRADICTS, 0.80, index=3),
        _finding(REL_SUPERSEDED_BY, 0.70, index=4),
        _finding(REL_SUBSUMES, 0.60, index=5),
    ]
    result = AuditResult(effort=TIER_CLAIM, corpus_root=str(tmp_path), n_docs=2, findings=findings)
    paths = write_audit(tmp_path / "audit", result)

    rows = [
        json.loads(line)
        for line in paths["findings"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["relation"] == REL_SUBSUMED_BY, "the top decision leads the machine file"
    assert rows[-1]["relation"] == REL_COMPLEMENTARY, "coexistence sinks whatever it scored"

    report_rows = [
        line
        for line in render_report(result).split("### Rows", 1)[1].splitlines()
        if line.startswith("| G")
    ]
    assert f"`{REL_SUBSUMED_BY}`" in report_rows[0]
    assert f"`{REL_COMPLEMENTARY}`" in report_rows[-1], "the two artifacts read in one order"


def test_the_resolution_overlay_is_the_same_whatever_order_the_rows_arrive_in(tmp_path):
    """The ordering writes `findings.jsonl`, so a reordering must not move a single corpus byte."""
    findings = [
        _finding(REL_DUPLICATE, 0.9, index=0),
        _finding(REL_COMPLEMENTARY, 1.0, index=1),
        _finding(REL_SUBSUMED_BY, 0.5, index=2),
    ]
    rows = [finding.payload() for finding in sorted(findings, key=finding_sort_key)]
    plan = build_plan(rows, POLICY_CONSERVATIVE, str(tmp_path))
    reversed_plan = build_plan(list(reversed(rows)), POLICY_CONSERVATIVE, str(tmp_path))

    assert overlay_from_plan(plan) == overlay_from_plan(reversed_plan)
    assert plan["action_counts"] == reversed_plan["action_counts"]
    assert {item["finding_id"] for item in plan["items"]} == {
        item["finding_id"] for item in reversed_plan["items"]
    }
