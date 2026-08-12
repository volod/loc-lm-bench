"""The opt-in TO REVIEW projection: one command earlier, and equal to what the resolver measures.

The audit cannot MEASURE the review count -- that number is a property of a resolution policy the
operator has not chosen yet. It can PROJECT it: `resolve_finding` is a pure function of
`(relation, tier, governance, policy)` and `findings.jsonl` carries all four. These tests pin the
three things that make the projection safe to print:

1. it equals `plan.json`'s `review_rows` group for group under the same policy, so it is a second
   reading of one implementation rather than a second implementation of one number;
2. it is labelled a projection under a NAMED policy everywhere it appears, and changes when the
   policy does;
3. the report is unchanged without the flag, and the report layer never imports the resolution
   vocabulary to render it -- the layering decision the task turned on.

The fixture is the mixed group the reconciliation is hardest on, the same shape
`test_two_counts.py` uses: one claim-tier duplicate the conservative policy accepts plus one
semantic-tier duplicate it escalates -- 2 to decide, 1 to review.
"""

import json
from pathlib import Path

import pytest

from llb.conflicts import policy_projection as projection_module
from llb.conflicts import report as report_module
from llb.conflicts import report_findings as report_findings_module
from llb.conflicts import report_precision as report_precision_module
from llb.conflicts.constants import (
    REL_CONTRADICTS,
    REL_DUPLICATE,
    REL_SUPERSEDED_BY,
    REVIEW_LABEL,
    TIER_CLAIM,
    TIER_SEMANTIC,
)
from llb.conflicts.models import AuditResult, ClaimRef, Finding, Staleness
from llb.conflicts.policy_projection import PROJECTION_KIND, project_review_rows
from llb.conflicts.report import render_report, write_audit
from llb.conflicts.resolution_io import create_resolution_artifacts
from llb.conflicts.resolution_policy import (
    POLICY_CONSERVATIVE,
    POLICY_PREFER_NEWER,
    build_plan,
)

SHARED_CHUNK = "left.md#recursive#0003"


def _row(
    index: int,
    *,
    tier: str,
    relation: str = REL_DUPLICATE,
    score: float = 0.9,
    left: str = "left.md",
    newer_side: str | None = None,
) -> Finding:
    return Finding(
        relation=relation,
        tier=tier,
        a=ClaimRef(left, 0, 10, "a" * 10, chunk_id=SHARED_CHUNK if left == "left.md" else left),
        b=ClaimRef(
            "right.md",
            index * 100,
            index * 100 + 10,
            "b" * 10,
            chunk_id=f"right.md#recursive#{index:04d}",
        ),
        score=score,
        evidence="model",
        staleness=Staleness(newer_side=newer_side, basis="issued_at" if newer_side else None),
    )


def _mixed() -> list[Finding]:
    """One accepted duplicate and one escalated candidate in ONE group: 2 to decide, 1 to review."""
    return [_row(1, tier=TIER_CLAIM), _row(2, tier=TIER_SEMANTIC)]


def _result(findings: list[Finding], policy: str | None = None) -> AuditResult:
    result = AuditResult(effort=TIER_CLAIM, corpus_root="corpus", n_docs=2, findings=list(findings))
    if policy is not None:
        result.policy_projection = project_review_rows(result.rows(), policy)
    return result


def test_the_projection_equals_the_plan_the_same_policy_would_write():
    """The acceptance gate: mixed group, projected count == `plan.json`'s measured `review_rows`."""
    result = _result(_mixed(), POLICY_CONSERVATIVE)
    plan = build_plan(result.rows(), POLICY_CONSERVATIVE, "corpus")

    measured = {str(d["group_id"]): int(d["review_rows"]) for d in plan["decisions"]}
    assert measured == {"G1": 1}, "the fixture mixes an accepted duplicate with an escalated one"
    assert result.policy_projection["groups"] == measured, "projected == measured, group for group"
    assert result.policy_projection["review_rows"] == 1
    assert result.policy_projection["review_groups"] == 1
    assert {str(d["group_id"]): int(d["decide_rows"]) for d in plan["decisions"]} == {"G1": 2}, (
        "and the two counts still differ, which is why the projection is worth printing"
    )


def test_the_projection_equals_the_measured_plan_on_every_policy_and_shape():
    """A policy the projection names is a policy it must agree with -- on rows that separate them."""
    findings = _mixed() + [
        _row(3, tier=TIER_CLAIM, relation=REL_SUPERSEDED_BY, newer_side="a", left="other.md"),
        _row(4, tier=TIER_CLAIM, relation=REL_CONTRADICTS, left="third.md"),
    ]
    rows = _result(findings).rows()

    projected = {
        policy: project_review_rows(rows, policy)["groups"]
        for policy in (POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)
    }
    measured = {
        policy: {
            str(d["group_id"]): int(d["review_rows"])
            for d in build_plan(rows, policy, "corpus")["decisions"]
        }
        for policy in (POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)
    }

    assert projected == measured
    assert projected[POLICY_CONSERVATIVE] != projected[POLICY_PREFER_NEWER], (
        "the fixture separates the policies, so 'a projection under a NAMED policy' is not a "
        "distinction without a difference"
    )


def test_the_projection_survives_the_round_trip_through_findings_jsonl(tmp_path):
    """The artifact an operator actually resolves is the file, so project what the file says."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("left.md", "right.md"):
        (corpus / name).write_text("x" * 10_000, encoding="utf-8")
    paths = write_audit(tmp_path / "audit", _result(_mixed(), POLICY_CONSERVATIVE))
    plan, _, _ = create_resolution_artifacts(
        paths["findings"], tmp_path / "plan", policy=POLICY_CONSERVATIVE, corpus_root=corpus
    )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

    assert summary["policy_projection"]["groups"] == {
        str(d["group_id"]): int(d["review_rows"]) for d in plan["decisions"]
    }
    assert summary["policy_projection"]["kind"] == PROJECTION_KIND


def test_the_report_labels_the_number_a_projection_wherever_it_prints_it():
    report = render_report(_result(_mixed(), POLICY_CONSERVATIVE))
    headline = report.split("## ", 1)[0]
    table = report.split("### Decision groups", 1)[1].split("### Rows", 1)[0]
    header = next(line for line in table.splitlines() if line.startswith("| group"))
    row = next(line for line in table.splitlines() if line.startswith("| G"))

    assert f"{REVIEW_LABEL} (PROJECTED under policy `{POLICY_CONSERVATIVE}`): 1 row" in headline
    assert "not a\nmeasurement" in headline or "not a measurement" in headline
    assert f"| {REVIEW_LABEL} (projected) |" in header
    assert [cell.strip() for cell in row.split("|")][1:5] == ["G1", "2", "2", "1"], (
        "rows, to decide, then the projected review count -- in that order"
    )
    assert "projection under a named policy, never a measurement" in table
    assert POLICY_CONSERVATIVE in table


def test_without_the_flag_the_report_is_the_report_it_always_was():
    plain = render_report(_result(_mixed()))
    projected = render_report(_result(_mixed(), POLICY_CONSERVATIVE))

    assert "PROJECTED" not in plain and "(projected)" not in plain
    assert "an audit cannot print it" in plain, "the unprojected claim is unchanged"
    assert plain != projected
    # Every line the plain report prints is still a line of the projected one, except the two the
    # projection widens: the group table's header/separator and its rows.
    widened = {"| group ", "| --- ", "| G"}
    kept = [
        line
        for line in plain.splitlines()
        if not any(line.startswith(prefix) for prefix in widened)
        and "an audit cannot print it" not in line
    ]
    assert all(line in projected.splitlines() for line in kept)


def test_an_empty_projection_is_not_a_projection_of_zero():
    """A group with nothing open must not read as 'no projection was asked for', or vice versa."""
    accepted = [_row(1, tier=TIER_CLAIM)]
    result = _result(accepted, POLICY_CONSERVATIVE)

    assert result.policy_projection["review_rows"] == 0
    report = render_report(result)
    assert "(PROJECTED under policy `conservative`): 0 rows in 0 decision groups" in report
    assert render_report(_result(accepted)).count("| ") > 0, "the plain report still renders"


def test_the_projection_refuses_a_policy_it_does_not_know():
    with pytest.raises(ValueError, match="unknown resolution policy"):
        project_review_rows(_result(_mixed()).rows(), "whatever-the-operator-typed")


def test_the_report_layer_does_not_import_the_resolution_vocabulary():
    """The layering decision, enforced rather than only written down.

    The projection needs `resolution_policy`; the report must not. It is composed above both (the
    CLI) and handed down as plain JSON, so the detector keeps rendering without a policy and no
    import runs from `conflicts.report*` into `conflicts.resolution_*`.
    """
    for module in (report_module, report_findings_module, report_precision_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        assert not [line for line in imports if "resolution" in line], (
            f"{module.__name__} imports the resolution vocabulary; the projection must be "
            "injected as data instead"
        )
    assert "resolution_policy" in Path(projection_module.__file__).read_text(encoding="utf-8"), (
        "the projection is the one module allowed to hold both layers"
    )
