"""The decision table is ranked by what each decision costs, without moving a single group id.

Group ids are derived from `findings.jsonl` in file order and are the join key `groups.json` and
`plan.json` share, so the table's ORDER and the group's NAME have to come apart: an operator reads
the table to decide what to fund first, while a machine reads the id to address the same rows in
three artifacts. These tests pin both halves -- and the failure that motivated the ranking, which is
that a saturating score leaves the reading order resting on a document-id tiebreak.
"""

import json

from llb.conflicts.grouping.census import group_findings
from llb.conflicts.constants import (
    REL_COMPLEMENTARY,
    REL_CONTRADICTS,
    TIER_CLAIM,
)
from llb.conflicts.grouping.artifact import group_summaries
from llb.conflicts.models import AuditResult, ClaimRef, Finding
from llb.conflicts.report.render import render_report, write_audit
from llb.conflicts.report.findings import stake_key
from llb.conflicts.resolution.policy import POLICY_CONSERVATIVE, build_plan


def _fan(unit: str, rows: int, relation: str, score: float, offset: int = 0) -> list[Finding]:
    """`rows` findings sharing one left chunk -- one decision of the given size."""
    return [
        Finding(
            relation=relation,
            tier=TIER_CLAIM,
            a=ClaimRef("left.md", offset, offset + 10, "a" * 10, chunk_id=unit),
            b=ClaimRef(
                "right.md",
                offset + index * 20 + 1000,
                offset + index * 20 + 1010,
                "b" * 10,
                chunk_id=f"{unit}#right#{index}",
            ),
            score=score,
            evidence="model",
        )
        for index in range(rows)
    ]


def _groups_of(report: str) -> list[str]:
    table = report.split("### Decision groups", 1)[1].split("### Rows", 1)[0]
    return [line.split("|")[1].strip() for line in table.splitlines() if line.startswith("| G")]


def _result(findings: list[Finding]) -> AuditResult:
    return AuditResult(effort=TIER_CLAIM, corpus_root="corpus", n_docs=2, findings=findings)


def _label_of(findings: list[Finding], predicate) -> str:
    """The file-order id of the one group matching `predicate` -- never assumed from the fixture."""
    return next(group.label for group in group_findings(findings) if predicate(group))


def test_the_biggest_decision_leads_even_without_the_top_scored_row():
    """The measured shape: several groups tie at the top score, so size has to break the tie."""
    findings = _fan("small", 2, REL_COMPLEMENTARY, 1.0) + _fan(
        "big", 20, REL_COMPLEMENTARY, 1.0, offset=100
    )
    ranked = _groups_of(render_report(_result(findings)))
    biggest = _label_of(findings, lambda group: len(group.findings) == 20)
    assert ranked[0] == biggest and len(ranked) == 2, "the 20-row decision leads"


def test_work_outranks_size():
    findings = _fan("big", 20, REL_COMPLEMENTARY, 1.0) + _fan(
        "small", 2, REL_CONTRADICTS, 0.1, offset=100
    )
    ranked = _groups_of(render_report(_result(findings)))
    actionable = _label_of(findings, lambda group: group.decide_rows == 2)
    assert ranked[0] == actionable, "two rows someone must act on outrank twenty that coexist"


def test_the_ranking_is_total_so_a_full_tie_falls_back_to_the_id():
    findings = _fan("one", 3, REL_COMPLEMENTARY, 0.5) + _fan(
        "two", 3, REL_COMPLEMENTARY, 0.5, offset=100
    )
    groups = group_findings(findings)
    assert sorted(groups, key=stake_key) == sorted(groups, key=lambda group: group.index)


def test_ranking_the_table_does_not_renumber_the_groups(tmp_path):
    """The id is a join key into `groups.json` and `plan.json`; the table order must not touch it."""
    findings = _fan("small", 1, REL_COMPLEMENTARY, 1.0) + _fan(
        "big", 9, REL_COMPLEMENTARY, 0.2, offset=100
    )
    result = _result(findings)
    paths = write_audit(tmp_path / "audit", result)
    sidecar = json.loads(paths["groups"].read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in paths["findings"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    report_order = _groups_of(paths["report"].read_text(encoding="utf-8"))
    assert report_order[0] == "G2", "ranked by stake"
    assert [group["group_id"] for group in sidecar["groups"]] == ["G1", "G2"], "ids in file order"
    sidecar_order = [
        group["group_id"] for group in sorted(sidecar["groups"], key=lambda group: group["rank"])
    ]
    assert sidecar_order == report_order, "the sidecar carries the report's own order"
    assert [(group["decide_rows"], group["rank"]) for group in sidecar["groups"]] == [
        (0, 2),
        (0, 1),
    ]
    plan = build_plan(rows, POLICY_CONSERVATIVE, "corpus")
    assert [(decision["group_id"], decision["rank"]) for decision in plan["decisions"]] == [
        (group["group_id"], group["rank"]) for group in sidecar["groups"]
    ], "the plan copies the audit rank instead of deriving another order"
    plan_order = [
        decision["group_id"]
        for decision in sorted(plan["decisions"], key=lambda decision: decision["rank"])
    ]
    assert plan_order == report_order
    assert group_summaries(rows) == sidecar["groups"], "the file still derives the same ids"


def test_the_row_table_stays_in_file_order(tmp_path):
    findings = _fan("small", 1, REL_COMPLEMENTARY, 1.0) + _fan(
        "big", 4, REL_COMPLEMENTARY, 0.2, offset=100
    )
    paths = write_audit(tmp_path / "audit", _result(findings))
    report = paths["report"].read_text(encoding="utf-8")
    labels = [
        line.split("|")[1].strip()
        for line in report.split("### Rows", 1)[1].splitlines()
        if line.startswith("| G")
    ]
    assert labels == ["G1"] + ["G2"] * 4, "rows read in the order a resolution lane consumes them"
