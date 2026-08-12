"""Two grouping rules over one row list, and the reasons only one of them can be quoted.

The transitive closure that defines a decision group runs long on a real corpus -- the measured
goods rows collapse 100 findings into 6 groups whose largest chains 51 rows through 23 different
shared chunks -- so the group count under-states the work exactly as the row count over-states it.
These tests pin the alternative rule (one group per shared unit, a row joining every group whose
unit it carries), the invariants that decide which rule the audit quotes, and the rendering that
states the choice.
"""

import json

from llb.conflicts.census import group_indices, row_pair_units
from llb.conflicts.constants import REL_COMPLEMENTARY, REL_CONTRADICTS, REL_DUPLICATE, TIER_CLAIM
from llb.conflicts.granularity import (
    QUOTED_RULE,
    RULE_SHARED_UNIT,
    RULE_TRANSITIVE,
    finding_granularity,
    granularity_of,
    rows_granularity,
    shared_unit_indices,
)
from llb.conflicts.models import AuditResult, ClaimRef, Finding
from llb.conflicts.report import render_report, write_audit
from llb.conflicts.report_granularity import comparison_report

SHARED_CHUNK = "left.md#recursive#0003"


def _ref(doc_id: str, chunk: str | None, start: int = 0) -> ClaimRef:
    return ClaimRef(
        doc_id=doc_id, char_start=start, char_end=start + 10, text="a" * 10, chunk_id=chunk
    )


def _finding(a: ClaimRef, b: ClaimRef, relation: str = REL_CONTRADICTS, score: float = 0.9):
    return Finding(relation=relation, tier=TIER_CLAIM, a=a, b=b, score=score, evidence="model")


def _concentrated(rows: int = 6) -> list[Finding]:
    """`rows` findings that all quote ONE left chunk -- one decision under either rule."""
    return [
        _finding(
            _ref("left.md", SHARED_CHUNK),
            _ref("right.md", f"right.md#recursive#{index:04d}", start=index * 100),
            score=0.9 - index / 100,
        )
        for index in range(rows)
    ]


def _chain(length: int) -> list[Finding]:
    """A-B, B-C, C-D ... : one transitive group, but `length` separate pieces of evidence."""
    refs = [_ref(f"d{i}.md", f"d{i}.md#0", start=i * 100) for i in range(length + 1)]
    return [_finding(refs[i], refs[i + 1], score=0.9 - i / 100) for i in range(length)]


def _result(findings: list[Finding]) -> AuditResult:
    return AuditResult(effort=TIER_CLAIM, corpus_root="corpus", n_docs=2, findings=findings)


def _rows_of(findings: list[Finding]) -> list[dict]:
    return _result(findings).rows()


# --- the rule itself ----------------------------------------------------------------------------


def test_one_shared_chunk_is_one_group_under_both_rules():
    """The fixture the whole comparison is anchored on: a fan is one decision either way."""
    granularity = finding_granularity(_concentrated())
    rules = granularity["rules"]
    assert rules[RULE_TRANSITIVE]["groups"] == 1
    assert rules[RULE_SHARED_UNIT]["groups"] == 1
    assert granularity["decision_range"] == [1, 1], "the range collapses to a point"


def test_a_chain_splits_into_one_group_per_link():
    """Where the two rules part: transitive closure calls a 4-link chain one decision."""
    granularity = finding_granularity(_chain(4))
    assert granularity["rules"][RULE_TRANSITIVE]["groups"] == 1
    assert granularity["rules"][RULE_SHARED_UNIT]["groups"] == 3, "the three interior units"
    assert granularity["decision_range"] == [1, 3]


def test_a_row_joins_every_group_whose_unit_it_carries():
    """The defining property of the rule -- and the reason it cannot be funded group by group."""
    granularity = finding_granularity(_chain(3))
    cover = granularity["rules"][RULE_SHARED_UNIT]
    assert cover["rows_in_multiple_groups"] == 1, "the middle row carries both shared units"
    assert cover["memberships"] == 4 and granularity["rows"] == 3
    assert cover["partition"] is False


def test_the_quoted_rule_is_the_only_partition():
    for findings in (_concentrated(), _chain(4), _concentrated(3) + _chain(2)):
        rules = finding_granularity(findings)["rules"]
        assert rules[QUOTED_RULE]["partition"] is True
        assert rules[QUOTED_RULE]["memberships"] == len(findings)


def test_rows_sharing_nothing_are_their_own_group_under_both_rules():
    findings = [
        _finding(_ref(f"a{i}.md", f"a{i}.md#0"), _ref(f"b{i}.md", f"b{i}.md#0")) for i in range(3)
    ]
    rules = finding_granularity(findings)["rules"]
    assert rules[RULE_TRANSITIVE]["groups"] == rules[RULE_SHARED_UNIT]["groups"] == 3
    assert rules[RULE_SHARED_UNIT]["partition"] is True, "a cover with no overlap is a partition"


def test_two_units_joining_the_same_rows_are_one_group():
    """Two chunks that only ever appear together are one piece of evidence, not two decisions."""
    left, right = _ref("l.md", "l.md#0"), _ref("r.md", "r.md#0")
    findings = [_finding(left, right), _finding(left, right, relation=REL_DUPLICATE)]
    assert finding_granularity(findings)["rules"][RULE_SHARED_UNIT]["groups"] == 1


def test_a_document_tier_row_groups_on_its_document_under_both_rules():
    copies = [_ref(name, None) for name in ("one.md", "copy.md", "reformatted.md")]
    findings = [
        _finding(copies[0], copies[1], relation=REL_DUPLICATE),
        _finding(copies[1], copies[2], relation=REL_DUPLICATE),
    ]
    rules = finding_granularity(findings)["rules"]
    assert rules[RULE_TRANSITIVE]["groups"] == 1, "three copies of one document are one decision"
    assert rules[RULE_SHARED_UNIT]["groups"] == 1, "`copy.md` is the one shared unit"


def test_no_shared_unit_group_spans_two_quoted_groups():
    """The invariant that makes the per-group split add up: the cover refines the partition."""
    findings = _concentrated(4) + _chain(3)
    pairs = [row_pair_units(row) for row in _rows_of(findings)]
    quoted = group_indices(pairs)
    owner = {index: at for at, members in enumerate(quoted) for index in members}
    for members in shared_unit_indices(pairs):
        assert len({owner[index] for index in members}) == 1
    split = finding_granularity(findings)["quoted_group_split"]
    total = sum(int(entry["shared_unit_groups"]) for entry in split)
    assert total == finding_granularity(findings)["rules"][RULE_SHARED_UNIT]["groups"]


def test_the_split_names_the_chain_inside_a_quoted_group():
    granularity = finding_granularity(_concentrated(4) + _chain(3))
    by_id = {entry["group_id"]: entry for entry in granularity["quoted_group_split"]}
    assert {entry["shared_unit_groups"] for entry in by_id.values()} == {1, 2}


def test_an_empty_run_reports_both_rules_at_zero():
    granularity = granularity_of([])
    assert granularity["decision_range"] == [0, 0]
    assert granularity["rules"][RULE_SHARED_UNIT]["partition"] is True


# --- the rules read the same over objects and over rows -------------------------------------------


def test_the_rules_read_findings_rows_and_finding_objects_identically():
    findings = _concentrated(4) + _chain(3)
    assert rows_granularity(_rows_of(findings)) == finding_granularity(findings)


# --- what the audit prints ------------------------------------------------------------------------


def test_the_report_states_which_rule_it_quotes_and_why():
    report = render_report(_result(_chain(4)))
    section = report.split("### How many decisions the row count is", 1)[1].split("### Rows", 1)[0]
    assert f"**The audit quotes `{QUOTED_RULE}`**" in section
    assert "only PARTITION of the rows" in section
    assert "**1 to 3**" in section, "the decision range, both ends group counts"
    assert "| `transitive` (quoted) | 1 |" in section
    assert "| `shared_unit` | 3 |" in section


def test_the_report_names_the_longest_chain():
    report = render_report(_result(_chain(4)))
    assert "G1's 4 rows run through 3 distinct shared units" in report


def test_the_report_says_so_when_the_two_rules_agree():
    report = render_report(_result(_concentrated()))
    assert "the two rules agree and the range is a point" in report


def test_the_summary_carries_both_rules(tmp_path):
    result = _result(_chain(4))
    summary = json.loads(write_audit(tmp_path, result)["summary"].read_text(encoding="utf-8"))
    assert summary["group_granularity"] == finding_granularity(result.findings)
    assert summary["group_granularity"]["quoted_rule"] == QUOTED_RULE
    assert summary["finding_census"]["groups"] == 1, "the quoted census is unchanged"


def test_grouping_granularity_never_reaches_findings_jsonl(tmp_path):
    """The second rule is a reading; the rows and the group ids a resolution lane joins on stay."""
    result = _result(_chain(4))
    paths = write_audit(tmp_path, result)
    sidecar = json.loads(paths["groups"].read_text(encoding="utf-8"))
    assert [group["group_id"] for group in sidecar["groups"]] == ["G1"]
    assert len(paths["findings"].read_text(encoding="utf-8").splitlines()) == 4


# --- the cross-run comparison ---------------------------------------------------------------------


def test_the_comparison_report_carries_a_row_per_run():
    entries = [
        {
            "label": name,
            "source": f"{name}/findings.jsonl",
            "granularity": finding_granularity(rows),
        }
        for name, rows in (("fan", _concentrated()), ("chain", _chain(4)))
    ]
    report = comparison_report(entries)
    assert "| `fan` | 6 | 1 | 1 | 1 - 1 | 0 |" in report
    assert "| `chain` | 4 | 1 | 3 | 1 - 3 | 2 |" in report, "both interior rows join two groups"
    assert "## fan" in report and "## chain" in report


def test_the_cli_recomputes_both_rules_over_a_run_directory(tmp_path):
    from typer.testing import CliRunner

    from llb.cli.app import app
    from llb.main import main  # noqa: F401  -- registers every command module

    run_dir = tmp_path / "audit"
    write_audit(run_dir, _result(_chain(4)))
    out_dir = tmp_path / "granularity"
    result = CliRunner().invoke(
        app,
        ["compare-conflict-granularity", "--run", str(run_dir), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.output
    data = json.loads((out_dir / "granularity.json").read_text(encoding="utf-8"))
    assert data[0]["granularity"]["decision_range"] == [1, 3]
    assert "audit" in (out_dir / "granularity.md").read_text(encoding="utf-8")


def test_a_relation_mix_does_not_change_either_rule():
    """Grouping is about units; a relation is what the group is FOR, never what joins it."""
    findings = _chain(3)
    mixed = [
        Finding(
            relation=REL_COMPLEMENTARY if index else REL_CONTRADICTS,
            tier=finding.tier,
            a=finding.a,
            b=finding.b,
            score=finding.score,
            evidence=finding.evidence,
        )
        for index, finding in enumerate(findings)
    ]
    assert finding_granularity(mixed)["rules"] == finding_granularity(findings)["rules"]
