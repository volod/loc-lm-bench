"""Task records are `NNNN-<group>-<task-id>.md`, numbered without gaps, and every one is indexed."""

from llb.core.paths import PROJECT_ROOT
from llb.quality.record_names import RECORDS_DIR, record_findings

SPEC = """# Spec

## Capability Registry

| # | Id | Status | Evaluation | Where |
| --- | --- | --- | --- | --- |
| 1 | `book-recovery` | planned | measured | plan |
"""


def _tree(tmp_path, records: dict[str, str], index_links: list[str]):
    (tmp_path / "docs/design").mkdir(parents=True)
    (tmp_path / "docs/design/spec.md").write_text(SPEC, encoding="utf-8")
    records_dir = tmp_path / RECORDS_DIR
    records_dir.mkdir(parents=True)
    (records_dir / "template.md").write_text("# Template\n", encoding="utf-8")
    index = "# Task Records\n\n" + "".join(f"- [{name}]({name})\n" for name in index_links)
    (records_dir / "README.md").write_text(index, encoding="utf-8")
    for name, text in records.items():
        (records_dir / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_the_shipped_records_are_named_numbered_and_indexed():
    assert record_findings(PROJECT_ROOT) == []


def test_a_well_formed_directory_has_no_findings(tmp_path):
    names = ["0001-adhoc-template-sync.md", "0002-book-recovery-tool-survey.md"]
    root = _tree(tmp_path, {name: "# r\n" for name in names}, names)
    assert record_findings(root) == []


def test_each_broken_rule_is_named(tmp_path):
    names = [
        "tool-survey.md",  # no sequence
        "0001-adhoc-sync.md",
        "0001-book-recovery-ledger.md",  # duplicate sequence
        "0004-unknown-group-task.md",  # unknown group, and 0002-0003 skipped
        "0005-book-recovery-.md",  # trailing hyphen leaves no task id
    ]
    root = _tree(tmp_path, {name: "# r\n" for name in names}, names[:-1])
    findings = "\n".join(record_findings(root))
    assert "tool-survey.md: name is not NNNN-<group>-<task-id>.md" in findings
    assert "sequence 0001 is also" in findings
    assert "0004-unknown-group-task.md: group is not `adhoc`" in findings
    assert "sequence 0002 is skipped" in findings and "sequence 0003 is skipped" in findings
    assert "0005-book-recovery-.md" in findings


def test_an_unindexed_record_is_refused(tmp_path):
    root = _tree(tmp_path, {"0001-adhoc-sync.md": "# r\n"}, [])
    assert record_findings(root) == [
        f"{RECORDS_DIR / '0001-adhoc-sync.md'}: not linked from {RECORDS_DIR / 'README.md'}"
    ]
