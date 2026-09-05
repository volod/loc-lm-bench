"""Ingestion reports the governance coverage that bounds later conflict audits."""

import json

from typer.testing import CliRunner

from llb.cli.app import app
from llb.conflicts.governance.coverage import governance_coverage
from llb.prep.corpus.governance import manifest_governance_by_doc
from llb.prep.corpus.ingest import CORPUS_MANIFEST, ingest_corpus

BODY = "A sufficiently long document body for corpus ingestion. " * 4


def _write_document(root, name, metadata=None):
    (root / name).write_text(BODY, encoding="utf-8")
    if metadata is not None:
        (root / f"{name}.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _manifest(out):
    return json.loads((out / CORPUS_MANIFEST).read_text(encoding="utf-8"))


def test_undated_corpus_reports_cost_without_refusing_ingestion(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    _write_document(root, "first.md")
    _write_document(root, "second.md")
    out = tmp_path / "out"

    result = CliRunner().invoke(
        app,
        [
            "ingest-corpus",
            "--root",
            str(root),
            "--out-dir",
            str(out),
            "--min-chars",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert (
        "0 of 2 documents with `effective_date` or `version` "
        "(0 `effective_date`, 0 `version`)" in result.output
    )
    assert "0 of 1 document pair orderable" in result.output
    assert "No supersession can ever be derived on this corpus" in result.output
    coverage = _manifest(out)["governance_coverage"]
    assert coverage["dated_documents"] == 0
    assert coverage["documents_by_field"] == {"effective_date": 0, "version": 0}
    assert coverage["orderable_document_pairs"] == 0


def test_dated_corpus_reports_each_field_and_matches_audit_coverage(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    _write_document(root, "date.md", {"effective_date": "2026-01-01"})
    _write_document(root, "v1.md", {"version": "v1"})
    _write_document(root, "v2.md", {"version": "v2"})
    out = tmp_path / "out"

    ingest = ingest_corpus(root, out, min_chars=1)
    manifest_coverage = _manifest(out)["governance_coverage"]
    governance = list(manifest_governance_by_doc(out).values())
    audit_coverage = governance_coverage(governance, [])

    assert ingest.governance_coverage == manifest_coverage
    assert manifest_coverage["dated_documents"] == 3
    assert manifest_coverage["documents_by_field"] == {"effective_date": 1, "version": 2}
    assert manifest_coverage["orderable_document_pairs"] == 1
    for field in (
        "documents",
        "dated_documents",
        "documents_by_field",
        "document_pairs",
        "orderable_document_pairs",
    ):
        assert manifest_coverage[field] == audit_coverage[field]


def test_one_shared_edition_still_reports_that_no_supersession_is_derivable(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    for name in ("first.md", "second.md"):
        _write_document(root, name, {"effective_date": "2026-01-01"})

    result = ingest_corpus(root, tmp_path / "out", min_chars=1)

    assert result.governance_coverage["dated_documents"] == 2
    assert result.governance_coverage["orderable_document_pairs"] == 0
    assert result.governance_coverage["consequence"].startswith(
        "No supersession can ever be derived on this corpus"
    )
