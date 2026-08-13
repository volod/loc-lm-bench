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
    REL_COMPLEMENTARY,
    REL_CONTRADICTS,
    REL_DUPLICATE,
    REL_SUPERSEDED_BY,
    REVIEW_LABEL,
    TIER_CLAIM,
    TIER_SEMANTIC,
)
from llb.conflicts.models import AuditResult, ClaimRef, Finding, Staleness
from llb.conflicts.policy_projection import (
    PROJECTION_KIND,
    project_policies,
    project_review_rows,
)
from llb.conflicts.report import render_report, write_audit
from llb.conflicts.report_projection import moved_share_phrase, projected_review_lines
from llb.conflicts.resolution_io import create_resolution_artifacts
from llb.conflicts.resolution_policy import (
    POLICIES,
    POLICY_CONSERVATIVE,
    POLICY_PREFER_NEWER,
    build_plan,
    resolve_finding,
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


def _separating() -> list[Finding]:
    """The mixed group plus the rows the two policies actually disagree about.

    Only a DATED supersession separates them: conservative escalates it, prefer-newer suppresses
    the older side. The undated contradiction is the control -- it stays review work under both,
    so a non-zero delta cannot be an artifact of simply having more rows.
    """
    return _mixed() + [
        _row(3, tier=TIER_CLAIM, relation=REL_SUPERSEDED_BY, newer_side="a", left="other.md"),
        _row(4, tier=TIER_CLAIM, relation=REL_CONTRADICTS, left="third.md"),
    ]


def _projected(findings: list[Finding], *policies: str) -> AuditResult:
    """The same result `_result` builds, projected under several policies instead of one."""
    result = _result(findings)
    result.policy_projection = project_policies(result.rows(), list(policies))
    return result


def _group_header(report: str) -> str:
    table = report.split("### Decision groups", 1)[1]
    return next(line for line in table.splitlines() if line.startswith("| group"))


def _measured(rows: list[dict], policy: str) -> dict[str, int]:
    """`plan.json`'s own `review_rows` per group -- what every projected column must equal."""
    return {
        str(d["group_id"]): int(d["review_rows"])
        for d in build_plan(rows, policy, "corpus")["decisions"]
    }


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
    rows = _result(_separating()).rows()

    projected = {
        policy: project_review_rows(rows, policy)["groups"]
        for policy in (POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)
    }
    measured = {
        policy: _measured(rows, policy) for policy in (POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)
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


# --- several policies at once, and the delta between them ----------------------------------------


def test_every_projected_column_equals_the_plan_its_own_policy_would_write():
    """The acceptance gate, one column at a time: N columns are N readings of one implementation."""
    result = _projected(_separating(), POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)
    rows = result.rows()
    projection = result.policy_projection

    assert projection["policies"] == [POLICY_CONSERVATIVE, POLICY_PREFER_NEWER]
    for policy in projection["policies"]:
        assert projection["by_policy"][policy]["groups"] == _measured(rows, policy), policy


def test_the_delta_is_the_difference_and_names_the_groups_it_falls_in():
    result = _projected(_separating(), POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)
    projection = result.policy_projection
    (delta,) = projection["deltas"]
    by_policy = projection["by_policy"]

    assert (delta["baseline"], delta["policy"]) == (POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)
    assert delta["groups"] == {
        group: by_policy[POLICY_PREFER_NEWER]["groups"][group] - count
        for group, count in by_policy[POLICY_CONSERVATIVE]["groups"].items()
    }
    assert delta["review_rows"] == -1, "the dated supersession is the one row the switch settles"
    assert (
        delta["moved_groups"] == [group for group, count in delta["groups"].items() if count] != []
    ), "a corpus-wide total hides which decision the choice actually moves"


def test_the_delta_reads_as_a_share_of_the_rows_the_audit_calls_work():
    """A sign says whether the choice is a choice; only a share says how big a one.

    The fixture is 4 rows, all of them actionable (it carries no `complementary` row), and exactly
    one of them moves. `-1` alone does not distinguish that from one row in a thousand, which is
    the whole reason the share is printed.
    """
    projection = _projected(_separating(), POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)
    (delta,) = projection.policy_projection["deltas"]

    assert delta["actionable_rows"] == 4, "every relation in the fixture but `complementary`"
    assert delta["moved_rows"] == 1, "the dated supersession is the one row the switch settles"
    assert delta["moved_share"] == 0.25
    assert delta["moved_rows"] == abs(delta["review_rows"]), (
        "with no offsetting moves the gross count and the net agree -- which is what makes the "
        "corpus where they disagree worth counting separately"
    )


def test_the_two_shipped_policies_can_only_ever_settle_rows_never_open_them():
    """Why `moved_rows` equals `|review_rows|` on every corpus this repo can currently measure.

    `prefer-newer` differs from `conservative` on exactly one relation and in exactly one
    direction: a dated supersession it settles instead of escalating. No row moves the other way,
    so the gross and the net cannot part under THIS pair -- which is a property of the two
    policies, not of the delta, and stops holding the moment a third policy escalates something
    the baseline accepts.
    """
    rows = _result(_separating()).rows()
    before, after = (
        {
            str(item["finding_id"]): str(item["status"])
            for item in (resolve_finding(row, policy) for row in rows)
        }
        for policy in (POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)
    )
    opened = [fid for fid, status in before.items() if status != after[fid] == "review_required"]
    (delta,) = project_policies(rows, [POLICY_CONSERVATIVE, POLICY_PREFER_NEWER])["deltas"]

    assert opened == [], "the switch never turns an accepted row back into review work"
    assert delta["moved_rows"] == abs(delta["review_rows"])


def test_a_delta_whose_moves_cancel_still_reports_the_rows_it_moved():
    """The branch a policy pair that moved rows BOTH ways would land in, pinned on the renderer.

    The two shipped policies cannot produce it (see above), so this is fixed at the layer that
    would have to render it: a net of zero with rows that moved must not print as `no difference`.
    """
    cancelled = {
        "policies": [POLICY_CONSERVATIVE, POLICY_PREFER_NEWER],
        "by_policy": {policy: {"review_rows": 3, "review_groups": 1} for policy in POLICIES},
        "deltas": [
            {
                "baseline": POLICY_CONSERVATIVE,
                "policy": POLICY_PREFER_NEWER,
                "review_rows": 0,
                "groups": {"G1": 0},
                "moved_groups": [],
                "moved_rows": 2,
                "actionable_rows": 8,
                "moved_share": 0.25,
            }
        ],
    }
    line = projected_review_lines(cancelled)[-1]

    assert "no difference on this corpus" not in line
    assert "**0 rows** to review, cancelling out" in line
    assert "moves **2 of 8 actionable rows (25.0%)**" in line


def test_a_corpus_with_nothing_to_decide_has_no_share_rather_than_a_zero_one():
    """`None` and `0.0` mean opposite things: no work at all, versus work the choice never touches."""
    only_complementary = [_row(1, tier=TIER_CLAIM, relation=REL_COMPLEMENTARY)]
    (delta,) = project_policies(
        _result(only_complementary).rows(), [POLICY_CONSERVATIVE, POLICY_PREFER_NEWER]
    )["deltas"]

    assert (delta["actionable_rows"], delta["moved_rows"]) == (0, 0)
    assert delta["moved_share"] is None
    assert "**0 of 0 actionable rows** moved" in render_report(
        _projected(only_complementary, POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)
    )


def test_the_report_and_the_cli_quote_one_share_through_one_helper():
    """The headline, the zero-delta line, and the CLI must not compute the share three ways."""
    projection = _projected(_separating(), POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)
    (delta,) = projection.policy_projection["deltas"]

    assert moved_share_phrase(delta) == "1 of 4 actionable rows (25.0%)"
    assert f"The choice moves **{moved_share_phrase(delta)}**" in render_report(projection)
    free = _projected(_mixed(), POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)
    (zero,) = free.policy_projection["deltas"]
    assert f"nothing here (**{moved_share_phrase(zero)}** moved)" in render_report(free)
    assert moved_share_phrase(zero).startswith("0 of "), "a free choice still reports its share"


def test_the_baseline_policy_stays_where_a_single_policy_consumer_reads_it():
    """One document shape: the first policy's own fields are the top level, whatever N is."""
    findings = _separating()
    one = _projected(findings, POLICY_CONSERVATIVE).policy_projection
    both = _projected(findings, POLICY_CONSERVATIVE, POLICY_PREFER_NEWER).policy_projection

    flat = ("kind", "policy", "basis", "review_rows", "review_groups", "groups")
    assert {key: one[key] for key in flat} == {key: both[key] for key in flat}
    assert one["schema_version"] == both["schema_version"]
    assert one["deltas"] == [], "one policy is no choice, so there is nothing to compare"
    assert set(both) == set(one), "the multi-policy payload adds values, never keys"


def test_naming_the_policies_the_other_way_round_swaps_the_baseline_and_the_sign():
    findings = _separating()
    forward = _projected(findings, POLICY_CONSERVATIVE, POLICY_PREFER_NEWER).policy_projection
    backward = _projected(findings, POLICY_PREFER_NEWER, POLICY_CONSERVATIVE).policy_projection

    assert forward["by_policy"] == backward["by_policy"], "the columns do not depend on the order"
    assert backward["policy"] == POLICY_PREFER_NEWER, "the first named is the baseline"
    assert backward["deltas"][0]["review_rows"] == -forward["deltas"][0]["review_rows"]


def test_a_single_policy_renders_exactly_the_column_it_always_did():
    single = render_report(_projected(_separating(), POLICY_CONSERVATIVE))
    header = _group_header(single)

    assert f"| {REVIEW_LABEL} (projected) |" in header
    assert "delta" not in header and "policy choice" not in single
    assert single == render_report(_result(_separating(), POLICY_CONSERVATIVE)), (
        "the multi-policy path with one policy is byte-identical to the single-policy path"
    )


def test_two_policies_render_one_labelled_column_each_plus_the_delta():
    report = render_report(_projected(_separating(), POLICY_CONSERVATIVE, POLICY_PREFER_NEWER))
    table = report.split("### Decision groups", 1)[1].split("### How many", 1)[0]
    header = [cell.strip() for cell in _group_header(report).split("|")]
    row = next(line for line in table.splitlines() if line.startswith("| G2 |"))

    assert header[4:7] == [
        f"{REVIEW_LABEL} (projected, `{POLICY_CONSERVATIVE}`)",
        f"{REVIEW_LABEL} (projected, `{POLICY_PREFER_NEWER}`)",
        f"delta (`{POLICY_CONSERVATIVE}` -> `{POLICY_PREFER_NEWER}`)",
    ]
    assert [cell.strip() for cell in row.split("|")][4:7] == ["1", "0", "-1"], (
        "the dated supersession costs a review under one policy and nothing under the other"
    )
    assert f"policy choice `{POLICY_CONSERVATIVE}` -> `{POLICY_PREFER_NEWER}`: **-1 row**" in report
    assert "(G2)" in report, "the headline names the group the choice moves"
    assert "never a measurement" in table and POLICY_PREFER_NEWER in table


def test_a_corpus_the_choice_is_free_on_reports_a_zero_delta_and_says_so():
    """The degenerate case is the common one, so it must read as an answer, not as a blank."""
    report = render_report(_projected(_mixed(), POLICY_CONSERVATIVE, POLICY_PREFER_NEWER))
    projection = _projected(_mixed(), POLICY_CONSERVATIVE, POLICY_PREFER_NEWER).policy_projection

    assert projection["deltas"][0]["moved_groups"] == []
    assert projection["deltas"][0]["review_rows"] == 0
    assert "**no difference on this corpus**" in report
    assert "the policies part on dated supersessions" in report


def test_a_delta_column_never_reads_as_a_count():
    """`+2` is a change and `2` is a count; a table that renders both must not confuse them."""
    swapped = render_report(_projected(_separating(), POLICY_PREFER_NEWER, POLICY_CONSERVATIVE))
    row = next(line for line in swapped.splitlines() if line.startswith("| G2 |"))
    assert [cell.strip() for cell in row.split("|")][4:7] == ["0", "1", "+1"]


def test_the_projection_refuses_a_repeated_policy():
    rows = _result(_mixed()).rows()
    with pytest.raises(ValueError, match="duplicate resolution policy"):
        project_policies(rows, [POLICY_CONSERVATIVE, POLICY_CONSERVATIVE])
    with pytest.raises(ValueError, match="at least one resolution policy"):
        project_policies(rows, [])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("conservative", [POLICY_CONSERVATIVE]),
        ("conservative,prefer-newer", [POLICY_CONSERVATIVE, POLICY_PREFER_NEWER]),
        (" prefer-newer , conservative ", [POLICY_PREFER_NEWER, POLICY_CONSERVATIVE]),
        (None, []),
    ],
)
def test_the_flag_parses_an_ordered_policy_list(value, expected):
    from llb.cli.prep.conflicts import _projected_policies

    assert _projected_policies(value) == expected


@pytest.mark.parametrize(
    "value", ["nonsense", "conservative,nonsense", "a,a", ",", "conservative,"]
)
def test_the_flag_refuses_what_it_cannot_project(value):
    import click

    from llb.cli.prep.conflicts import _projected_policies

    if value == "conservative,":
        assert _projected_policies(value) == [POLICY_CONSERVATIVE], "a trailing comma is a typo,"
        return
    with pytest.raises(click.exceptions.BadParameter):
        _projected_policies(value)


def test_the_report_layer_does_not_import_the_resolution_vocabulary():
    """The layering decision, enforced rather than only written down.

    The projection needs `resolution_policy`; the report must not. It is composed above both (the
    CLI) and handed down as plain JSON, so the detector keeps rendering without a policy and no
    import runs from `conflicts.report*` into `conflicts.resolution_*`. Every `report*` module is
    discovered rather than listed, so a renderer added later is held to the rule by default.
    """
    package = Path(report_module.__file__).parent
    renderers = sorted(package.glob("report*.py"))
    assert {path for path in renderers} >= {
        Path(module.__file__)
        for module in (report_module, report_findings_module, report_precision_module)
    }, "the discovery must find at least the renderers this test was written against"
    for path in renderers:
        source = path.read_text(encoding="utf-8")
        imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        assert not [line for line in imports if "resolution" in line], (
            f"{path.name} imports the resolution vocabulary; the projection must be "
            "injected as data instead"
        )
    assert "resolution_policy" in Path(projection_module.__file__).read_text(encoding="utf-8"), (
        "the projection is the one module allowed to hold both layers"
    )
