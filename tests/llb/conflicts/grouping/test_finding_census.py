"""The distinct-unit census and the decision grouping that keeps a row count honest.

The failure this guards against is arithmetic, not detection: on a concentrated corpus six rows can
be one stale chunk seen from six sides, and a report that prints only "6 findings" invites an
operator to fund six reviews. Every count the audit prints must therefore carry the units behind it,
and the findings table must show the decision -- while `findings.jsonl` keeps every row, because the
resolution lane consumes rows rather than decisions.
"""

import json

from llb.conflicts.grouping.census import (
    census_phrase,
    finding_census,
    finding_sort_key,
    group_findings,
    relation_census,
)
from llb.conflicts.constants import REL_CONTRADICTS, REL_DUPLICATE, TIER_CLAIM
from llb.conflicts.models import AuditResult, ClaimRef, Finding
from llb.conflicts.report.render import render_report, write_audit

SHARED_CHUNK = "left.md#recursive#0003"


def _ref(doc_id: str, chunk: str | None, start: int = 0) -> ClaimRef:
    return ClaimRef(
        doc_id=doc_id, char_start=start, char_end=start + 10, text="a" * 10, chunk_id=chunk
    )


def _finding(a: ClaimRef, b: ClaimRef, relation: str = REL_CONTRADICTS, score: float = 0.9):
    return Finding(relation=relation, tier=TIER_CLAIM, a=a, b=b, score=score, evidence="model")


def _concentrated(rows: int = 6) -> list[Finding]:
    """`rows` findings that all quote ONE left chunk against one right document -- one decision."""
    return [
        _finding(
            _ref("left.md", SHARED_CHUNK),
            _ref("right.md", f"right.md#recursive#{index:04d}", start=index * 100),
            score=0.9 - index / 100,
        )
        for index in range(rows)
    ]


def _result(findings: list[Finding]) -> AuditResult:
    return AuditResult(effort=TIER_CLAIM, corpus_root="corpus", n_docs=2, findings=findings)


def test_rows_sharing_one_chunk_are_one_group():
    census = finding_census(_concentrated())
    assert census["findings"] == 6
    assert census["groups"] == 1 and census["largest_group"] == 6
    assert census["documents"] == 2 and census["document_pairs"] == 1
    assert census["chunk_units"] == 7, "one left chunk plus six right chunks"


def test_rows_sharing_nothing_stay_separate_decisions():
    findings = [
        _finding(_ref(f"a{i}.md", f"a{i}.md#0"), _ref(f"b{i}.md", f"b{i}.md#0")) for i in range(3)
    ]
    census = finding_census(findings)
    assert census["groups"] == 3 and census["largest_group"] == 1
    assert census["document_pairs"] == 3


def test_grouping_is_transitive_over_a_shared_unit():
    """A-B and B-C are one decision about B, not two independent findings."""
    a, b, c = (_ref(f"{name}.md", f"{name}.md#0") for name in "abc")
    groups = group_findings([_finding(a, b), _finding(b, c)])
    assert len(groups) == 1 and len(groups[0].findings) == 2
    assert groups[0].shared_units == ("b.md#0",)


def test_a_tier_without_chunks_groups_on_the_document():
    """Hash and lexical findings carry no chunk, so their unit is the document they compare."""
    copies = [_ref(name, None) for name in ("one.md", "copy.md", "reformatted.md")]
    findings = [
        _finding(copies[0], copies[1], relation=REL_DUPLICATE),
        _finding(copies[1], copies[2], relation=REL_DUPLICATE),
    ]
    census = finding_census(findings)
    assert census["groups"] == 1, "three copies of one document are one decision"
    assert census["chunk_units"] == 3


def test_relation_census_counts_units_per_relation():
    findings = _concentrated(3) + [
        _finding(_ref("x.md", "x.md#0"), _ref("y.md", "y.md#0"), relation=REL_DUPLICATE)
    ]
    census = relation_census(findings)
    assert census[REL_CONTRADICTS]["findings"] == 3 and census[REL_CONTRADICTS]["groups"] == 1
    assert census[REL_DUPLICATE]["findings"] == 1 and census[REL_DUPLICATE]["groups"] == 1


def test_every_printed_count_carries_its_units():
    report = render_report(_result(_concentrated()))
    units = "on 2 documents / 1 document pair / 7 chunk units, in 1 group (largest 6)"
    assert f"- findings: 6 {units}" in report, "the headline count"
    assert f"| `{REL_CONTRADICTS}` | 6 | 2 | 1 | 7 | 1 |" in report, "the relation table row"
    assert f"**{census_phrase(finding_census(_concentrated()))}.**" in report, "the findings list"


def test_the_findings_table_shows_one_decision_not_six_rows():
    report = render_report(_result(_concentrated()))
    groups = report.split("### Decision groups", 1)[1].split("### Rows", 1)[0]
    group_rows = [line for line in groups.splitlines() if line.startswith("| G")]
    assert len(group_rows) == 1
    assert f"| G1 | 6 | 6 | `{REL_CONTRADICTS}` x6 | `{SHARED_CHUNK}` |" in group_rows[0]
    rows = [line for line in report.split("### Rows", 1)[1].splitlines() if line.startswith("| G")]
    assert len(rows) == 6, "grouping is a rendering of every row, not a filter over them"


def test_a_clean_corpus_prints_a_bare_count_and_no_groups():
    report = render_report(_result([]))
    assert "- findings: 0\n" in report
    assert "### Decision groups" not in report


def test_findings_jsonl_keeps_every_row_byte_identical(tmp_path):
    """The resolution lane consumes rows; grouping must never reach `findings.jsonl`."""
    result = _result(_concentrated())
    paths = write_audit(tmp_path, result)
    expected = "".join(
        json.dumps(finding.payload(), ensure_ascii=False) + "\n"
        for finding in sorted(result.findings, key=finding_sort_key)
    )
    assert paths["findings"].read_text(encoding="utf-8") == expected


def test_the_summary_carries_the_same_census_as_the_report(tmp_path):
    result = _result(_concentrated())
    summary = json.loads(write_audit(tmp_path, result)["summary"].read_text(encoding="utf-8"))
    assert summary["n_findings"] == 6, "the row count stays the row count"
    assert summary["finding_census"] == finding_census(result.findings)
    assert summary["relation_census"] == relation_census(result.findings)
    assert summary["relations"] == {REL_CONTRADICTS: 6}, "the relation counts stay unchanged"
