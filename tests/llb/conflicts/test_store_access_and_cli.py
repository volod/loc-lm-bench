"""The real store-loading path and the Typer command, over a FAISS-backed fixture store.

Marked `heavy_env`: these build a real vector index, so they need the `[rag]` extra and are
deselected in the base `[dev]`-only GitHub CI environment. Everything else in this package runs
without FAISS or an encoder.
"""

import json
import zlib

import numpy as np
import pytest
from typer.testing import CliRunner

from llb.cli.app import app
from llb.conflicts.constants import FINDINGS_FILE, REPORT_FILE, SUMMARY_FILE
from llb.conflicts.store_access import load_store_view
from llb.rag.store import RagStore

from conflict_helpers import DOC_2021, DOC_2021_COPY, FIXTURE_CORPUS

pytestmark = pytest.mark.heavy_env

DIM = 64


class FakeEmbedder:
    """Deterministic hashed bag-of-words encoder (the project's standard test fake)."""

    model_name = "fake-hashed-bow"

    def _matrix(self, texts):
        if not texts:
            return np.zeros((0, DIM), dtype="float32")
        rows = []
        for text in texts:
            vector = np.zeros(DIM, dtype="float32")
            for token in text.casefold().split():
                vector[zlib.crc32(token.encode("utf-8")) % DIM] += 1.0
            norm = float(np.linalg.norm(vector))
            rows.append(vector / norm if norm else vector)
        return np.stack(rows)

    def encode_passages(self, texts):
        return self._matrix(list(texts))

    def encode_queries(self, texts):
        return self._matrix(list(texts))


@pytest.fixture
def built_store(tmp_path):
    store = RagStore.build(
        FIXTURE_CORPUS, strategy="heading", size=600, overlap=0, embedder=FakeEmbedder()
    )
    index_dir = tmp_path / "rag"
    store.save(index_dir)
    return index_dir


def test_store_view_reads_chunks_and_vectors_without_an_encoder(built_store):
    """Conflict detection compares stored vectors, so it must not need the encoder to load."""
    view = load_store_view(built_store)
    assert len(view.chunks) == len(view.vectors)
    assert view.embedding_model == "fake-hashed-bow"
    assert view.dim == DIM
    assert view.doc_fingerprints


def test_store_view_reports_a_missing_store_clearly(tmp_path):
    with pytest.raises(SystemExit, match="no store at"):
        load_store_view(tmp_path / "absent")


def test_cli_hash_effort_writes_the_report_artifacts(tmp_path):
    out = tmp_path / "report"
    result = CliRunner().invoke(
        app,
        ["audit-corpus-conflicts", "--corpus", str(FIXTURE_CORPUS), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    for name in (FINDINGS_FILE, REPORT_FILE, SUMMARY_FILE):
        assert (out / name).is_file()
    summary = json.loads((out / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["effort"] == "hash"
    assert summary["relations"]["duplicate"] >= 1
    assert "duplicate" in result.output


def test_cli_semantic_effort_uses_the_built_store(built_store, tmp_path):
    out = tmp_path / "report"
    result = CliRunner().invoke(
        app,
        [
            "audit-corpus-conflicts",
            "--corpus",
            str(FIXTURE_CORPUS),
            "--effort",
            "semantic",
            "--store",
            str(built_store),
            "--cos-threshold",
            "0.85",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((out / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["tree"]["n_vectors"] > 0
    assert summary["tree"]["embedding_model"] == "fake-hashed-bow"
    rows = [
        json.loads(line)
        for line in (out / FINDINGS_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any({DOC_2021, DOC_2021_COPY} & {row["a"]["doc_id"], row["b"]["doc_id"]} for row in rows)


def test_cli_rejects_an_unknown_effort(tmp_path):
    result = CliRunner().invoke(
        app, ["audit-corpus-conflicts", "--corpus", str(FIXTURE_CORPUS), "--effort", "free"]
    )
    assert result.exit_code != 0


def test_cli_claim_effort_requires_a_model(built_store, tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "audit-corpus-conflicts",
            "--corpus",
            str(FIXTURE_CORPUS),
            "--effort",
            "claim",
            "--store",
            str(built_store),
        ],
    )
    assert result.exit_code != 0
    assert "conflict-model" in result.output


def test_audit_never_modifies_the_corpus(built_store, tmp_path):
    """Detection only: the audit must leave every corpus byte untouched."""
    before = {
        path.name: path.read_bytes() for path in sorted(FIXTURE_CORPUS.iterdir()) if path.is_file()
    }
    CliRunner().invoke(
        app,
        [
            "audit-corpus-conflicts",
            "--corpus",
            str(FIXTURE_CORPUS),
            "--effort",
            "semantic",
            "--store",
            str(built_store),
            "--out",
            str(tmp_path / "report"),
        ],
    )
    after = {
        path.name: path.read_bytes() for path in sorted(FIXTURE_CORPUS.iterdir()) if path.is_file()
    }
    assert before == after
