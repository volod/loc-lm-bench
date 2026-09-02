"""Acquisition provenance read from the projection sidecar and carried to every consumer.

The contract is `docs/design/acquired-corpus-projection.md`: an upstream service renders seven
fields into `<source>.metadata.json`, this side reads them, and a corpus carrying none of them
ingests exactly as an operator's own directory does.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from llb.prep.corpus.fingerprints import corpus_fingerprint
from llb.prep.corpus.governance import manifest_governance_by_doc
from llb.prep.corpus.governance_fields import (
    ACQUIRED_GOVERNANCE_FIELDS,
    FINGERPRINTED_GOVERNANCE_FIELDS,
    GOVERNANCE_FIELDS,
)
from llb.prep.corpus.ingest import CORPUS_MANIFEST, ingest_corpus
from llb.prep.corpus.ingest_text import CorpusItem
from llb.prep.ontology.models import DocRecord
from llb.prep.ontology.pipeline.bundle_provenance import document_rows
from llb.rag.chunking.corpus import chunk_corpus

DOC = "# Розділ\n\n" + ("Це достатньо довгий український документ. " * 20)
OTHER_DOC = "Це текстовий документ про кругообіг води у природі. " * 20

PROJECTED = {
    "language": "uk",
    "version": "2",
    "effective_date": "2026-01-01",
    "source_system": "acq-service",
    "acl_label": "public",
    "source_uri": "https://example.org/reg/17",
    "capture_time": "2026-01-02T03:04:05Z",
    "capture_id": "cap-0001",
    "payload_digest": "sha256:" + "a" * 64,
    "licence": "redistributable",
    "acquisition_run_id": "run-2026-01-02",
    "revision_of": "doc-0000",
}


def _manifest(out_dir):
    return json.loads((out_dir / CORPUS_MANIFEST).read_text(encoding="utf-8"))


def _projected_corpus(root, sidecar=PROJECTED):
    root.mkdir(parents=True, exist_ok=True)
    (root / "acquired.md").write_text(DOC, encoding="utf-8")
    (root / "acquired.md.metadata.json").write_text(
        json.dumps(sidecar, ensure_ascii=False), encoding="utf-8"
    )
    return root


def test_corpus_item_carries_every_governance_field():
    """The manifest row is built by splatting a governance row, so the two lists must agree."""
    assert set(GOVERNANCE_FIELDS) <= {field.name for field in dataclasses.fields(CorpusItem)}
    assert set(FINGERPRINTED_GOVERNANCE_FIELDS) < set(GOVERNANCE_FIELDS)


def test_acquired_fields_are_read_into_manifest_and_chunk_metadata(tmp_path):
    root = _projected_corpus(tmp_path / "src")
    out = tmp_path / "out"

    ingest_corpus(root, out, min_chars=50)

    item = next(item for item in _manifest(out)["items"] if item["source"] == "acquired.md")
    for field, expected in PROJECTED.items():
        assert item[field] == expected, field

    chunks = chunk_corpus(out, "sentence", 200, 0)
    metadata = next(chunk for chunk in chunks if chunk["doc_id"] == "acquired.md")["metadata"]
    for field in ACQUIRED_GOVERNANCE_FIELDS:
        assert metadata[field] == PROJECTED[field], field


def test_corpus_without_provenance_ingests_identically_with_fields_recorded_absent(tmp_path):
    """The invariant labels depend on: same text, same doc ids, same offsets, fields present-absent."""
    root = tmp_path / "src"
    (root / "nested").mkdir(parents=True)
    (root / "a.md").write_text(DOC, encoding="utf-8")
    (root / "nested" / "b.txt").write_text(OTHER_DOC, encoding="utf-8")
    out = tmp_path / "out"

    result = ingest_corpus(root, out, min_chars=50)

    assert (out / "a.md").read_text(encoding="utf-8") == DOC
    assert (out / "nested" / "b.txt").read_text(encoding="utf-8") == OTHER_DOC
    assert sorted(item.doc_id for item in result.items) == ["a.md", "nested/b.txt"]
    for item in _manifest(out)["items"]:
        for field in ACQUIRED_GOVERNANCE_FIELDS:
            assert field in item and item[field] is None, field

    chunks = chunk_corpus(out, "sentence", 200, 0)
    offsets = {(c["doc_id"], c["char_start"], c["char_end"]) for c in chunks}
    texts = {c["doc_id"]: (out / c["doc_id"]).read_text(encoding="utf-8") for c in chunks}
    for doc_id, start, end in offsets:
        assert texts[doc_id][start:end]
    for chunk in chunks:
        for field in ACQUIRED_GOVERNANCE_FIELDS:
            assert chunk["metadata"][field] is None, field


def test_front_matter_does_not_supply_acquired_fields(tmp_path):
    """Front matter is the operator-authored lane; a projected corpus writes a sidecar instead."""
    root = tmp_path / "src"
    root.mkdir()
    front_matter = "---\nversion: 3\nsource_uri: https://example.org/not-read\n---\n"
    (root / "a.md").write_text(front_matter + DOC, encoding="utf-8")
    out = tmp_path / "out"

    ingest_corpus(root, out, min_chars=50)

    item = _manifest(out)["items"][0]
    assert item["version"] == "3"  # operator field: still read from front matter
    assert item["source_uri"] is None


@pytest.mark.parametrize("field", ACQUIRED_GOVERNANCE_FIELDS)
def test_each_acquired_field_enters_the_corpus_fingerprint(tmp_path, field):
    baseline = _projected_corpus(tmp_path / "base")
    baseline_out = tmp_path / "base_out"
    ingest_corpus(baseline, baseline_out, min_chars=50)

    changed = _projected_corpus(tmp_path / "changed", {**PROJECTED, field: "changed-value"})
    changed_out = tmp_path / "changed_out"
    ingest_corpus(changed, changed_out, min_chars=50)

    assert corpus_fingerprint(baseline_out) != corpus_fingerprint(changed_out)


def test_ingestion_time_stays_out_of_the_fingerprint(tmp_path):
    assert "ingestion_time" in GOVERNANCE_FIELDS
    assert "ingestion_time" not in FINGERPRINTED_GOVERNANCE_FIELDS

    root = _projected_corpus(tmp_path / "src")
    out = tmp_path / "out"
    ingest_corpus(root, out, min_chars=50)
    first = corpus_fingerprint(out)

    ingest_corpus(root, out, min_chars=50, refresh=True)
    assert corpus_fingerprint(out) == first


def test_changed_sidecar_updates_provenance_on_an_unchanged_document(tmp_path):
    """A re-capture rewrites the sidecar, not the text; the reused row still carries the new run."""
    root = _projected_corpus(tmp_path / "src")
    out = tmp_path / "out"
    ingest_corpus(root, out, min_chars=50)

    (root / "acquired.md.metadata.json").write_text(
        json.dumps({**PROJECTED, "acquisition_run_id": "run-2026-02-02"}, ensure_ascii=False),
        encoding="utf-8",
    )
    rerun = ingest_corpus(root, out, min_chars=50)

    item = next(item for item in rerun.items if item.source == "acquired.md")
    assert item.reused is True
    assert item.acquisition_run_id == "run-2026-02-02"
    assert manifest_governance_by_doc(out)["acquired.md"]["acquisition_run_id"] == "run-2026-02-02"


def test_goldset_provenance_documents_carry_the_acquired_fields(tmp_path):
    """The bundle record resolves a drafted document to the capture it was derived from."""
    root = _projected_corpus(tmp_path / "src")
    out = tmp_path / "out"
    ingest_corpus(root, out, min_chars=50)
    text = (out / "acquired.md").read_text(encoding="utf-8")
    result = SimpleNamespace(
        corpus_root=out,
        docs=[DocRecord(doc_id="acquired.md", text=text, sha256="0" * 64, n_chars=len(text))],
    )

    row = document_rows(result)[0]

    assert row["doc_id"] == "acquired.md" and row["n_chars"] == len(text)
    for field in ACQUIRED_GOVERNANCE_FIELDS:
        assert row[field] == PROJECTED[field], field


def test_goldset_provenance_records_absent_provenance_explicitly(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "a.md").write_text(DOC, encoding="utf-8")
    out = tmp_path / "out"
    ingest_corpus(root, out, min_chars=50)
    result = SimpleNamespace(
        corpus_root=out,
        docs=[DocRecord(doc_id="a.md", text=DOC, sha256="0" * 64, n_chars=len(DOC))],
    )

    row = document_rows(result)[0]

    for field in ACQUIRED_GOVERNANCE_FIELDS:
        assert field in row and row[field] is None, field
