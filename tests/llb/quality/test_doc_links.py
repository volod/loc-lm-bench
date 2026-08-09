"""The docs tree is navigated by link now, so a link that does not land is a lost topic.

The shipped-tree assertion is the point of the tool; the synthetic cases pin the two ways a link
fails and the anchor rules a moved section relies on -- a heading keeps its fragment when its LEVEL
changes, which is what lets a section move from one file to another without breaking inbound links.
"""

import subprocess

from llb.core.paths import PROJECT_ROOT
from llb.quality.doc_links import anchors, broken_links, heading_anchor


def _repo(tmp_path, files: dict[str, str]):
    for name, text in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def test_the_shipped_docs_tree_has_no_broken_link():
    """The CI-able assertion: every relative link in the repo's Markdown resolves."""
    assert broken_links(PROJECT_ROOT) == []


def test_a_heading_keeps_its_anchor_when_its_level_changes(tmp_path):
    """Why a section can move between files: the fragment does not depend on the level."""
    deep = tmp_path / "deep.md"
    deep.write_text("##### Cap-fitting boundary surface\n", encoding="utf-8")
    shallow = tmp_path / "shallow.md"
    shallow.write_text("# Cap-fitting boundary surface\n", encoding="utf-8")

    assert anchors(deep) == anchors(shallow) == {"cap-fitting-boundary-surface"}


def test_punctuation_and_repeats_slug_the_way_a_markdown_host_does():
    """`+` and `(` drop out and leave their spaces behind, which is what doubles the hyphens."""
    assert heading_anchor("Hybrid Retrieval (Dense + BM25 + RRF)") == (
        "hybrid-retrieval-dense--bm25--rrf"
    )
    assert (
        heading_anchor("Measurement Floor (`--noise-floor`)") == "measurement-floor---noise-floor"
    )


def test_a_repeated_heading_takes_the_numbered_anchor(tmp_path):
    """Two sections named alike are legal here (MD024 is off), so the second must not shadow."""
    doc = tmp_path / "repeat.md"
    doc.write_text("## Evidence\n\n## Evidence\n", encoding="utf-8")

    assert anchors(doc) == {"evidence", "evidence-1"}


def test_a_link_to_a_missing_file_and_a_missing_anchor_are_both_named(tmp_path):
    """Both failure modes, named with the file and line so a fix does not need a search."""
    root = _repo(
        tmp_path,
        {
            "index.md": (
                "# Index\n\n[gone](area/missing.md)\n[stale](area/topic.md#renamed)\n"
                "[fine](area/topic.md#what-it-answers)\n[external](https://example.invalid#x)\n"
            ),
            "area/topic.md": "# Topic\n\n## What it answers\n",
        },
    )

    findings = broken_links(root)

    assert findings == [
        "index.md:3: missing file -> area/missing.md",
        "index.md:4: missing anchor -> area/topic.md#renamed",
    ]


def test_a_link_inside_a_fenced_block_is_not_a_link(tmp_path):
    """Command samples routinely contain bracket-paren text that no reader clicks."""
    root = _repo(
        tmp_path,
        {"index.md": "# Index\n\n```bash\nmake thing  # [see](nowhere.md)\n```\n"},
    )

    assert broken_links(root) == []
