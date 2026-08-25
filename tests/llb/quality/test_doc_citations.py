"""A page that cites a run directory stops being checkable the moment that directory is deleted.

The shipped-tree assertion is the point of the tool. The synthetic cases pin the two refused forms
and, just as importantly, the three shapes that must stay legal: a `$DATA_DIR` TEMPLATE documenting
where a command writes, a run label trailing a description inside a table row, and a fenced block
quoting a command someone actually ran.
"""

import subprocess

from llb.core.paths import PROJECT_ROOT
from llb.quality.doc_citations import RULE_DEFINING_DOCS, path_citations


def _repo(tmp_path, files: dict[str, str]):
    for name, text in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def test_the_shipped_docs_cite_no_run_directory_and_no_bare_label():
    """The CI-able assertion: every measured result in the docs is readable without the bundle."""
    assert path_citations(PROJECT_ROOT) == []


def test_both_refused_forms_are_named(tmp_path):
    """A run-directory path and a bare run label are each reported with their line."""
    root = _repo(
        tmp_path,
        {
            "page.md": (
                "Artifact: `$DATA_DIR/run-eval/20260728T065519.474285Z-2f08bcd131d7/`.\n"
                "\n"
                "Re-read of `20260815T-bare-id-squad-cos060` moved nothing.\n"
            )
        },
    )
    findings = path_citations(root)

    assert [finding.split(": ", 1)[1].split(" -> ")[0] for finding in findings] == [
        "run-directory path",
        "bare run label",
    ]
    assert findings[0].startswith("page.md:1:")
    assert findings[1].startswith("page.md:3:")


def test_a_template_a_table_row_and_a_fenced_command_stay_legal(tmp_path):
    """The three shapes that carry no host-local claim: they document, tabulate, and quote."""
    root = _repo(
        tmp_path,
        {
            "page.md": (
                "Writes `$DATA_DIR/corpus-conflicts/<run>/report.md`.\n"
                "\n"
                "| run | rows |\n"
                "| --- | ---: |\n"
                "| `20260815T-bare-id-squad-cos060` | 250 |\n"
                "\n"
                "```bash\n"
                "ls $DATA_DIR/run-eval/20260728T065519.474285Z-2f08bcd131d7/\n"
                "```\n"
            )
        },
    )

    assert path_citations(root) == []


def test_only_the_documents_that_state_the_rule_are_exempt():
    """The exemption exists so the rule can quote what it forbids -- not to excuse a page."""
    assert [str(doc) for doc in RULE_DEFINING_DOCS] == [
        "docs/guides/development/heavy-runs-and-evidence.md",
    ]
    for doc in RULE_DEFINING_DOCS:
        assert (PROJECT_ROOT / doc).exists()
