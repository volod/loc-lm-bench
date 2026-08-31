"""Decision groups as machine-readable output: the audit sidecar and the grouped resolution plan.

The inflation this guards against is downstream of detection: six rows quoting one stale chunk are
one call an operator makes once, and a plan that lists six escalations asks for the review the
report was changed to stop asking for. The group id must therefore mean the same thing in
`groups.json`, `plan.json`, and `resolution_review.jsonl` -- while `findings.jsonl` keeps every row,
because the overlay and its rollback are built row by row.
"""

import json

import pytest

from llb.conflicts.constants import (
    FINDINGS_FILE,
    GROUPS_FILE,
    REVIEW_RECORDS_FILE,
    TIER_CLAIM,
)
from llb.conflicts.grouping.artifact import (
    GROUPS_SCHEMA_VERSION,
    group_key,
    group_summaries,
    groups_document,
)
from llb.conflicts.tiers.hashing import finding_id
from llb.conflicts.models import AuditResult, ClaimRef, Finding
from llb.conflicts.report.render import write_audit
from llb.conflicts.resolution.io import create_resolution_artifacts, load_findings
from llb.conflicts.resolution.policy import (
    ACTION_ESCALATE,
    ACTION_KEEP_BOTH,
    POLICY_CONSERVATIVE,
    build_plan,
)

SHARED_CHUNK = "left.md#recursive#0003"


def _finding(index: int, relation: str = "contradicts") -> Finding:
    """One of a fan of rows that all quote the same left chunk -- one decision, many rows."""
    return Finding(
        relation=relation,
        tier=TIER_CLAIM,
        a=ClaimRef("left.md", 0, 10, "a" * 10, chunk_id=SHARED_CHUNK),
        b=ClaimRef(
            "right.md",
            index * 100,
            index * 100 + 10,
            "b" * 10,
            chunk_id=f"right.md#recursive#{index:04d}",
        ),
        score=0.9 - index / 100,
        evidence="model",
    )


def _concentrated(rows: int = 6) -> list[Finding]:
    return [_finding(index) for index in range(rows)]


def _write_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "left.md").write_text("a" * 10_000, encoding="utf-8")
    (corpus / "right.md").write_text("b" * 10_000, encoding="utf-8")
    return corpus


def _audit(tmp_path, findings: list[Finding]):
    result = AuditResult(
        effort=TIER_CLAIM, corpus_root=str(tmp_path / "corpus"), n_docs=2, findings=findings
    )
    return write_audit(tmp_path / "audit", result)


def test_the_sidecar_addresses_the_rows_it_was_written_beside(tmp_path):
    paths = _audit(tmp_path, _concentrated())
    sidecar = json.loads(paths["groups"].read_text(encoding="utf-8"))
    rows, source_sha = load_findings(paths["findings"])

    assert paths["groups"].name == GROUPS_FILE
    assert sidecar["schema_version"] == GROUPS_SCHEMA_VERSION
    assert sidecar["source_findings_sha256"] == source_sha, "pinned to the rows on disk"
    assert sidecar["census"]["findings"] == 6 and sidecar["census"]["groups"] == 1
    group = sidecar["groups"][0]
    assert group["group_id"] == "G1" and group["rows"] == 6
    assert group["shared_units"] == [SHARED_CHUNK]
    assert group["documents"] == ["left.md", "right.md"]
    assert group["finding_ids"] == [finding_id(row) for row in rows]
    assert group["group_key"] == group_key(group["finding_ids"])


def test_group_keys_survive_member_and_group_reordering():
    first_group = [_finding(0).payload(), _finding(1).payload()]
    second_group = [
        Finding(
            relation="contradicts",
            tier=TIER_CLAIM,
            a=ClaimRef("third.md", 0, 10, "c" * 10, chunk_id="third.md#recursive#0001"),
            b=ClaimRef("fourth.md", 0, 10, "d" * 10, chunk_id="fourth.md#recursive#0001"),
            score=0.7,
            evidence="model",
        ).payload()
    ]

    before_rows = first_group + second_group
    after_rows = second_group + list(reversed(first_group))
    before_sidecar = groups_document(before_rows, findings_sha256="before")
    after_sidecar = groups_document(after_rows, findings_sha256="after")
    before = before_sidecar["groups"]
    after = after_sidecar["groups"]

    before_by_key = {summary["group_key"]: summary for summary in before}
    after_by_key = {summary["group_key"]: summary for summary in after}
    assert before_by_key.keys() == after_by_key.keys()
    assert before_by_key.keys() == {
        group_key([finding_id(row) for row in first_group]),
        group_key([finding_id(row) for row in second_group]),
    }
    assert {
        key: (before_by_key[key]["group_id"], after_by_key[key]["group_id"])
        for key in before_by_key
    } == {
        before[0]["group_key"]: ("G1", "G2"),
        before[1]["group_key"]: ("G2", "G1"),
    }

    before_plan = build_plan(before_rows, POLICY_CONSERVATIVE, "corpus")
    after_plan = build_plan(after_rows, POLICY_CONSERVATIVE, "corpus")
    assert {decision["group_key"] for decision in before_plan["decisions"]} == before_by_key.keys()
    assert {decision["group_key"] for decision in after_plan["decisions"]} == after_by_key.keys()


def test_the_resolution_lane_derives_the_same_groups_without_the_sidecar(tmp_path):
    paths = _audit(tmp_path, _concentrated())
    sidecar = json.loads(paths["groups"].read_text(encoding="utf-8"))
    rows, _ = load_findings(paths["findings"])
    paths["groups"].unlink()

    assert group_summaries(rows) == sidecar["groups"], (
        "the sidecar is an emission of the grouping, not its only source"
    )


def test_a_concentrated_plan_carries_one_decision_naming_its_member_rows(tmp_path):
    rows = [finding.payload() for finding in _concentrated()]
    plan = build_plan(rows, POLICY_CONSERVATIVE, "corpus")

    assert len(plan["items"]) == 6, "the overlay is still built row by row"
    assert {item["group_id"] for item in plan["items"]} == {"G1"}
    assert len(plan["decisions"]) == 1
    decision = plan["decisions"][0]
    assert decision["group_key"] == group_key(decision["finding_ids"])
    assert decision["rank"] == 1
    assert decision["rows"] == 6 and decision["action"] == ACTION_ESCALATE
    assert decision["status"] == "review_required" and decision["review_rows"] == 6
    assert decision["finding_ids"] == [item["finding_id"] for item in plan["items"]]
    assert decision["shared_units"] == [SHARED_CHUNK]


def test_a_group_whose_rows_disagree_is_reported_as_mixed_rather_than_decided(tmp_path):
    rows = [
        _finding(0, relation="duplicate").payload(),
        _finding(1, relation="complementary").payload(),
    ]
    decision = build_plan(rows, POLICY_CONSERVATIVE, "corpus")["decisions"][0]

    assert decision["action"] is None, "no single action the members agreed on"
    assert decision["actions"] == {"drop_duplicate": 1, "keep_both": 1}
    assert decision["status"] == "accepted", "neither row asked for review"


def test_rows_that_share_nothing_are_separate_decisions(tmp_path):
    rows = [
        Finding(
            relation="contradicts",
            tier=TIER_CLAIM,
            a=ClaimRef(f"a{i}.md", 0, 5, "x" * 5, chunk_id=f"a{i}.md#0"),
            b=ClaimRef(f"b{i}.md", 0, 5, "y" * 5, chunk_id=f"b{i}.md#0"),
            score=0.5,
            evidence="model",
        ).payload()
        for i in range(3)
    ]
    plan = build_plan(rows, POLICY_CONSERVATIVE, "corpus")
    assert [decision["group_id"] for decision in plan["decisions"]] == ["G1", "G2", "G3"]
    assert all(decision["rows"] == 1 for decision in plan["decisions"])


def test_the_review_ledger_tells_a_reviewer_the_row_is_one_of_a_group(tmp_path):
    corpus = _write_corpus(tmp_path)
    paths = _audit(tmp_path, _concentrated())
    _, _, out = create_resolution_artifacts(
        paths["findings"], tmp_path / "plan", policy=POLICY_CONSERVATIVE, corpus_root=corpus
    )
    records = [
        json.loads(line)
        for line in out["review"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 6, "a drop still applies to one span, so the row stays the unit"
    assert {record["group_id"] for record in records} == {"G1"}
    assert {record["group_rows"] for record in records} == {6}


def test_one_group_wide_keep_settles_every_row_it_covers(tmp_path):
    corpus = _write_corpus(tmp_path)
    paths = _audit(tmp_path, _concentrated())
    reviewed = tmp_path / REVIEW_RECORDS_FILE
    reviewed.write_text(
        json.dumps({"group_id": "G1", "resolution_decision": ACTION_KEEP_BOTH}) + "\n",
        encoding="utf-8",
    )
    plan, overlay, _ = create_resolution_artifacts(
        paths["findings"],
        tmp_path / "plan",
        policy=POLICY_CONSERVATIVE,
        corpus_root=corpus,
        reviewed=reviewed,
    )
    assert plan["action_counts"] == {ACTION_KEEP_BOTH: 6}
    assert plan["decisions"][0]["status"] == "accepted"
    assert all(item["status"] == "accepted" for item in plan["items"])
    assert not any(
        directive.get("drop") for directive in (overlay.get("documents") or {}).values()
    ), "keeping a group suppresses nothing"


def test_a_group_wide_drop_is_refused_because_it_would_reach_unreviewed_spans(tmp_path):
    corpus = _write_corpus(tmp_path)
    paths = _audit(tmp_path, _concentrated())
    reviewed = tmp_path / REVIEW_RECORDS_FILE
    reviewed.write_text(
        json.dumps({"group_id": "G1", "resolution_decision": "drop_a"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="whole-group review decision"):
        create_resolution_artifacts(
            paths["findings"],
            tmp_path / "plan",
            policy=POLICY_CONSERVATIVE,
            corpus_root=corpus,
            reviewed=reviewed,
        )


def test_grouping_leaves_findings_jsonl_byte_identical(tmp_path):
    paths = _audit(tmp_path, _concentrated())
    rows = [finding.payload() for finding in _concentrated()]
    expected = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    assert paths["findings"].name == FINDINGS_FILE
    assert paths["findings"].read_text(encoding="utf-8") == expected
