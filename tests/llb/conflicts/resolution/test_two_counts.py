"""Two counts of one decision group's work, named apart and printed together.

The audit counts TO DECIDE (relation is not `complementary`) and the resolver counts TO REVIEW
(policy left the row at `review_required`). They diverge in both directions -- an accepted
`drop_duplicate` is to decide and not to review, a semantic-tier duplicate is both -- so an
operator handed the two numbers under one name reconciles a difference that is real.

The fixture here is the mixed group the reconciliation is hardest on: one claim-tier duplicate the
conservative policy accepts, one semantic-tier duplicate it escalates, both quoting the same left
chunk, so they are ONE decision worth 2 to decide and 1 to review.
"""

import json

from llb.conflicts.grouping.census import group_findings
from llb.conflicts.constants import (
    DECIDE_LABEL,
    REL_COMPLEMENTARY,
    REL_DUPLICATE,
    REVIEW_LABEL,
    TIER_CLAIM,
    TIER_SEMANTIC,
    decide_count,
)
from llb.conflicts.models import AuditResult, ClaimRef, Finding
from llb.conflicts.report.render import render_report, write_audit
from llb.conflicts.resolution.io import create_resolution_artifacts
from llb.conflicts.resolution.policy import (
    ACTION_DROP_DUPLICATE,
    ACTION_ESCALATE,
    PLAN_SCHEMA_VERSION,
    POLICY_CONSERVATIVE,
    build_plan,
)

SHARED_CHUNK = "left.md#recursive#0003"


def _row(index: int, *, tier: str, relation: str = REL_DUPLICATE, score: float = 0.9) -> Finding:
    """One row of a mixed group: same left chunk, different tier, so different policy outcomes."""
    return Finding(
        relation=relation,
        tier=tier,
        a=ClaimRef("left.md", 0, 10, "a" * 10, chunk_id=SHARED_CHUNK),
        b=ClaimRef(
            "right.md",
            index * 100,
            index * 100 + 10,
            "b" * 10,
            chunk_id=f"right.md#recursive#{index:04d}",
        ),
        score=score,
        evidence="model",
    )


def _mixed() -> list[Finding]:
    """An accepted duplicate and an escalated candidate in one group: 2 to decide, 1 to review."""
    return [_row(1, tier=TIER_CLAIM), _row(2, tier=TIER_SEMANTIC)]


def _corpus(tmp_path):
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


def test_the_two_counts_of_one_group_are_different_numbers():
    findings = _mixed()
    group = group_findings(findings)[0]
    decision = build_plan([f.payload() for f in findings], POLICY_CONSERVATIVE, "corpus")[
        "decisions"
    ][0]

    assert len(group.findings) == 2, "one decision, not two"
    assert group.decide_rows == 2, "both duplicates are work the vocabulary calls work"
    assert decision["review_rows"] == 1, "only the semantic candidate lacks deletion authority"
    assert decision["decide_rows"] == group.decide_rows, "the plan restates the audit's own count"
    assert decision["actions"] == {ACTION_DROP_DUPLICATE: 1, ACTION_ESCALATE: 1}


def test_the_headline_count_is_named_before_an_operator_meets_the_other_one():
    report = render_report(
        AuditResult(effort=TIER_CLAIM, corpus_root="corpus", n_docs=2, findings=_mixed())
    )
    headline = report.split("## ", 1)[0]

    assert f"- {DECIDE_LABEL}: 2 of 2 rows" in headline
    assert REVIEW_LABEL in headline and "`review_rows`" in headline, (
        "the headline names the count it is not, where that count lives, and what it depends on"
    )
    assert "resolution policy" in headline


def test_the_report_names_its_count_and_names_the_one_it_cannot_compute():
    report = render_report(
        AuditResult(effort=TIER_CLAIM, corpus_root="corpus", n_docs=2, findings=_mixed())
    )
    table = report.split("### Decision groups", 1)[1].split("### Rows", 1)[0]
    header = next(line for line in table.splitlines() if line.startswith("| group"))
    row = next(line for line in table.splitlines() if line.startswith("| G"))

    assert f"| {DECIDE_LABEL} |" in header, "the column is named, not left as bare 'actionable'"
    assert "| actionable |" not in header
    assert row.split("|")[3].strip() == "2", "the audit's count is to decide"
    assert REVIEW_LABEL in table and "review_rows" in table, (
        "the section names the count it does not print and says where it lives"
    )
    assert "`decide_rows`" in table


def test_the_plan_and_the_ledger_carry_both_counts(tmp_path):
    corpus = _corpus(tmp_path)
    paths = _audit(tmp_path, _mixed())
    plan, _, out = create_resolution_artifacts(
        paths["findings"], tmp_path / "plan", policy=POLICY_CONSERVATIVE, corpus_root=corpus
    )
    records = [
        json.loads(line)
        for line in out["review"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert plan["schema_version"] == PLAN_SCHEMA_VERSION
    assert [(d["decide_rows"], d["review_rows"]) for d in plan["decisions"]] == [(2, 1)]
    assert len(records) == 1, "only the escalated row is open"
    assert (records[0]["group_rows"], records[0]["group_decide_rows"]) == (2, 2)
    assert records[0]["group_review_rows"] == 1


def test_the_ledger_is_ranked_on_the_count_a_reviewer_funds(tmp_path):
    """A big to-decide group with one open row must not outrank a smaller all-open group."""
    findings = [
        # G1 by file order: 4 rows, all to decide, but only one of them open for a human.
        _row(1, tier=TIER_CLAIM),
        _row(2, tier=TIER_CLAIM, score=0.8),
        _row(3, tier=TIER_CLAIM, score=0.7),
        _row(4, tier=TIER_SEMANTIC, score=0.6),
    ]
    other = "other.md#recursive#0001"
    findings += [
        Finding(
            relation=REL_DUPLICATE,
            tier=TIER_SEMANTIC,
            a=ClaimRef("other.md", 0, 10, "c" * 10, chunk_id=other),
            b=ClaimRef("far.md", i * 50, i * 50 + 10, "d" * 10, chunk_id=f"far.md#{i}"),
            score=0.5,
            evidence="model",
        )
        for i in range(2)
    ]
    plan = build_plan([f.payload() for f in findings], POLICY_CONSERVATIVE, "corpus")
    by_group = {d["group_id"]: (d["decide_rows"], d["review_rows"]) for d in plan["decisions"]}

    from llb.conflicts.resolution.review import review_jsonl

    order = [
        json.loads(line)["group_id"] for line in review_jsonl(plan).splitlines() if line.strip()
    ]
    leader = max(by_group, key=lambda gid: by_group[gid][1])
    ranked_by_decide = max(by_group, key=lambda gid: by_group[gid][0])

    assert by_group[leader] == (2, 2) and by_group[ranked_by_decide] == (4, 1)
    assert leader != ranked_by_decide, "the fixture separates the two rankings"
    assert order[0] == leader, "the ledger leads with the most review, not the most to decide"
    blocks = [gid for index, gid in enumerate(order) if index == 0 or order[index - 1] != gid]
    assert len(blocks) == len(set(blocks)), "a group still reads as one contiguous block"


def test_decide_count_is_one_implementation_over_a_relation_census():
    assert decide_count({REL_DUPLICATE: 3, REL_COMPLEMENTARY: 5}) == 3
    assert decide_count({REL_COMPLEMENTARY: 9}) == 0
    assert decide_count({"something_the_vocabulary_does_not_know": 2}) == 2
