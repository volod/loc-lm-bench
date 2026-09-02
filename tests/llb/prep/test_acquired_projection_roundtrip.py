"""Round-trip conformance for the committed acquired-corpus projection fixture."""

import json
import shutil
from pathlib import Path

import pytest

from llb.prep.corpus.governance_fields import (
    ACQUIRED_GOVERNANCE_FIELDS,
    LOCAL_GOVERNANCE_FIELDS,
    OPERATOR_GOVERNANCE_FIELDS,
)
from llb.prep.corpus.ingest import CORPUS_MANIFEST, ingest_corpus
from llb.prep.corpus.fingerprints import corpus_doc_fingerprints

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = PROJECT_ROOT / "samples" / "corpora" / "acquired_projection_v1"
PROJECTED_SIDECAR_FIELDS = (
    tuple(field for field in OPERATOR_GOVERNANCE_FIELDS if field not in LOCAL_GOVERNANCE_FIELDS)
    + ACQUIRED_GOVERNANCE_FIELDS
)
EXPECTED_DOCUMENT_COUNT = 20
NON_REDISTRIBUTABLE_LICENCE = "local-only"


def _load_sidecar(document: Path) -> dict[str, object]:
    sidecar = document.with_name(document.name + ".metadata.json")
    assert sidecar.is_file(), f"missing projection sidecar for '{document.name}'"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"projection sidecar must be an object: '{sidecar.name}'"
    for field in PROJECTED_SIDECAR_FIELDS:
        assert field in payload, f"{sidecar.name}: missing projected field '{field}'"
    return payload


def _assert_projection_roundtrip(root: Path, out_dir: Path) -> None:
    documents = sorted(root.glob("*.md"))
    assert len(documents) == EXPECTED_DOCUMENT_COUNT
    sidecar_documents = {
        path.name.removesuffix(".metadata.json") for path in root.glob("*.md.metadata.json")
    }
    assert sidecar_documents == {document.name for document in documents}
    expected = {document.name: _load_sidecar(document) for document in documents}

    result = ingest_corpus(root, out_dir, min_chars=1)
    assert result.n_docs == EXPECTED_DOCUMENT_COUNT
    assert result.n_skipped == 0
    assert {item.status for item in result.items} == {"ok"}

    manifest = json.loads((out_dir / CORPUS_MANIFEST).read_text(encoding="utf-8"))
    actual = {item["source"]: item for item in manifest["items"]}
    assert set(actual) == set(expected)
    for source, projected in expected.items():
        assert (out_dir / source).read_bytes() == (root / source).read_bytes()
        for field in PROJECTED_SIDECAR_FIELDS:
            assert actual[source][field] == projected[field], f"{source}: {field}"


def test_committed_acquired_projection_roundtrips(tmp_path: Path) -> None:
    _assert_projection_roundtrip(FIXTURE_ROOT, tmp_path / "out")


def test_fixture_covers_revision_and_redistribution_boundaries() -> None:
    documents = sorted(FIXTURE_ROOT.glob("*.md"))
    sidecars = {document.name: _load_sidecar(document) for document in documents}

    revisions = {
        source: metadata["revision_of"]
        for source, metadata in sidecars.items()
        if metadata["revision_of"] is not None
    }
    assert revisions == {"fixture-doc-02.md": "fixture-doc-01.md"}
    assert all(parent in sidecars for parent in revisions.values())
    assert any(metadata["licence"] == NON_REDISTRIBUTABLE_LICENCE for metadata in sidecars.values())
    for identity_field in ("source_uri", "capture_id", "payload_digest"):
        values = [metadata[identity_field] for metadata in sidecars.values()]
        assert len(set(values)) == EXPECTED_DOCUMENT_COUNT, identity_field


def test_fixture_revision_retains_superseded_document_and_span(tmp_path: Path) -> None:
    root = tmp_path / "acquired_projection_v1"
    shutil.copytree(FIXTURE_ROOT, root)
    old_name = "fixture-doc-01.md"
    revision_name = "fixture-doc-02.md"
    revision_text = (root / revision_name).read_text(encoding="utf-8")
    revision_sidecar = _load_sidecar(root / revision_name)
    (root / revision_name).unlink()
    (root / f"{revision_name}.metadata.json").unlink()
    out = tmp_path / "out"

    ingest_corpus(root, out, min_chars=1)
    old_text = (out / old_name).read_text(encoding="utf-8")
    span_start = old_text.index("three years")
    span_end = span_start + len("three years")

    (root / old_name).unlink()
    (root / f"{old_name}.metadata.json").unlink()
    (root / revision_name).write_text(revision_text, encoding="utf-8")
    (root / f"{revision_name}.metadata.json").write_text(
        json.dumps(revision_sidecar, indent=2) + "\n", encoding="utf-8"
    )
    result = ingest_corpus(root, out, min_chars=1)

    assert result.n_docs == EXPECTED_DOCUMENT_COUNT
    assert result.removed_sources == []
    assert (out / old_name).read_text(encoding="utf-8")[span_start:span_end] == "three years"
    assert (out / revision_name).read_text(encoding="utf-8") == revision_text
    assert set(corpus_doc_fingerprints(out)) >= {old_name, revision_name}
    manifest = json.loads((out / CORPUS_MANIFEST).read_text(encoding="utf-8"))
    assert {item["doc_id"] for item in manifest["items"]} >= {old_name, revision_name}


def test_acquired_document_rejects_in_place_text_change(tmp_path: Path) -> None:
    root = tmp_path / "acquired_projection_v1"
    root.mkdir()
    name = "fixture-doc-01.md"
    shutil.copy2(FIXTURE_ROOT / name, root / name)
    shutil.copy2(FIXTURE_ROOT / f"{name}.metadata.json", root / f"{name}.metadata.json")
    out = tmp_path / "out"
    ingest_corpus(root, out, min_chars=1)
    staged_text = (out / name).read_text(encoding="utf-8")

    (root / name).write_text(staged_text + "\nChanged in place.\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"acquired document '{name}' changed in place"):
        ingest_corpus(root, out, min_chars=1)
    assert (out / name).read_text(encoding="utf-8") == staged_text


@pytest.mark.parametrize("field", PROJECTED_SIDECAR_FIELDS)
def test_renamed_projected_field_fails_with_field_named(tmp_path: Path, field: str) -> None:
    drifted_root = tmp_path / "acquired_projection_v1"
    shutil.copytree(FIXTURE_ROOT, drifted_root)
    sidecar = drifted_root / "fixture-doc-01.md.metadata.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload[f"{field}_renamed"] = payload.pop(field)
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(AssertionError, match=rf"missing projected field '{field}'"):
        _assert_projection_roundtrip(drifted_root, tmp_path / "out")
