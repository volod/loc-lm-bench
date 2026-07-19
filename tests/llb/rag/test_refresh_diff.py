"""Manifest diff + per-doc fingerprints (the dynamic-corpus-refresh diff contract)."""

import json

from llb.prep.corpus_governance import (
    CORPUS_MANIFEST,
    corpus_doc_fingerprints,
    corpus_fingerprint,
)
from llb.rag.refresh.diff import diff_fingerprints


def test_diff_classifies_added_modified_deleted_unchanged():
    indexed = {"a.md": "1", "b.md": "2", "c.md": "3"}
    current = {"a.md": "1", "b.md": "CHANGED", "d.md": "4"}
    diff = diff_fingerprints(indexed, current)
    assert diff.added == ["d.md"]
    assert diff.modified == ["b.md"]
    assert diff.deleted == ["c.md"]
    assert diff.unchanged == ["a.md"]
    assert diff.changed == {"b.md", "d.md"}
    assert diff.has_changes
    assert diff.counts() == {"added": 1, "modified": 1, "deleted": 1, "unchanged": 1}
    assert "1 added" in diff.summary()


def test_diff_no_changes():
    same = {"a.md": "1"}
    diff = diff_fingerprints(same, dict(same))
    assert not diff.has_changes
    assert diff.unchanged == ["a.md"]


def test_file_fingerprints_track_content(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")
    first = corpus_doc_fingerprints(tmp_path)
    assert sorted(first) == ["a.md", "b.txt"]  # only corpus doc suffixes, keyed by doc_id
    (tmp_path / "a.md").write_text("alpha CHANGED", encoding="utf-8")
    second = corpus_doc_fingerprints(tmp_path)
    assert second["a.md"] != first["a.md"]
    assert second["b.txt"] == first["b.txt"]


def _manifest_item(doc_id: str, sha: str, **governance):
    return {
        "status": "ok",
        "doc_id": doc_id,
        "source": f"src/{doc_id}",
        "kind": "text",
        "n_chars": 100,
        "source_sha256": sha,
        "language": "uk",
        "version": None,
        "effective_date": None,
        "source_system": "local",
        "acl_label": None,
        **governance,
    }


def test_manifest_fingerprints_cover_content_and_governance(tmp_path):
    items = [_manifest_item("a.md", "sha-a"), _manifest_item("b.md", "sha-b")]
    (tmp_path / CORPUS_MANIFEST).write_text(json.dumps({"items": items}), encoding="utf-8")
    first = corpus_doc_fingerprints(tmp_path)
    assert sorted(first) == ["a.md", "b.md"]

    # content change and governance change both move the per-doc fingerprint
    items[0]["source_sha256"] = "sha-a2"
    items[1]["acl_label"] = "hr-only"
    (tmp_path / CORPUS_MANIFEST).write_text(json.dumps({"items": items}), encoding="utf-8")
    second = corpus_doc_fingerprints(tmp_path)
    assert second["a.md"] != first["a.md"]
    assert second["b.md"] != first["b.md"]

    # non-ok items are excluded, matching corpus_fingerprint's contract
    items.append(_manifest_item("broken.md", "x") | {"status": "error"})
    (tmp_path / CORPUS_MANIFEST).write_text(json.dumps({"items": items}), encoding="utf-8")
    assert "broken.md" not in corpus_doc_fingerprints(tmp_path)


def test_unchanged_docs_keep_stable_fingerprints_alongside_corpus_fingerprint(tmp_path):
    items = [_manifest_item("a.md", "sha-a"), _manifest_item("b.md", "sha-b")]
    (tmp_path / CORPUS_MANIFEST).write_text(json.dumps({"items": items}), encoding="utf-8")
    before_docs = corpus_doc_fingerprints(tmp_path)
    before_corpus = corpus_fingerprint(tmp_path)
    items[0]["source_sha256"] = "sha-a2"
    (tmp_path / CORPUS_MANIFEST).write_text(json.dumps({"items": items}), encoding="utf-8")
    assert corpus_fingerprint(tmp_path) != before_corpus  # aggregate moves
    assert corpus_doc_fingerprints(tmp_path)["b.md"] == before_docs["b.md"]  # per-doc stays
