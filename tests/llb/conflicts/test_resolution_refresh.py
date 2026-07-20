"""FAISS-backed overlay refresh and exact rollback ranking."""

import shutil
import zlib

import numpy as np
import pytest

from llb.conflicts.overlay import overlay_from_plan
from llb.conflicts.resolution_io import install_overlay, rollback_overlay
from llb.rag.refresh.store_refresh import refresh_vector_store
from llb.rag.store import RagStore

from conflict_helpers import FIXTURE_CORPUS

pytestmark = pytest.mark.heavy_env


class FakeEmbedder:
    model_name = "fake-resolution-bow"

    def _matrix(self, texts):
        rows = []
        for text in texts:
            vector = np.zeros(64, dtype="float32")
            for token in text.casefold().split():
                vector[zlib.crc32(token.encode("utf-8")) % len(vector)] += 1.0
            norm = float(np.linalg.norm(vector))
            rows.append(vector / norm if norm else vector)
        return np.stack(rows) if rows else np.zeros((0, 64), dtype="float32")

    def encode_passages(self, texts):
        return self._matrix(list(texts))

    def encode_queries(self, texts):
        return self._matrix(list(texts))


def test_keep_both_refresh_and_overlay_deletion_restore_exact_ranking(tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURE_CORPUS, corpus)
    store_dir = tmp_path / "rag"
    baseline_store = RagStore.build(
        corpus, strategy="heading", size=600, overlap=0, embedder=FakeEmbedder()
    )
    baseline_store.save(store_dir)
    question = "deadline for reviewing a written appeal"
    baseline = [hit["chunk_id"] for hit in baseline_store.retrieve(question, 10)]
    refs = []
    for doc_id in ("archive-policy.md", "deadline-note.md"):
        text = (corpus / doc_id).read_text(encoding="utf-8")
        refs.append(
            {
                "doc_id": doc_id,
                "char_start": 0,
                "char_end": len(text),
                "text": text,
                "offsets_exact": True,
            }
        )
    plan = {
        "schema_version": 1,
        "policy": "conservative",
        "items": [
            {
                "finding_id": "keep-both",
                "relation": "complementary",
                "tier": "claim",
                "action": "keep_both",
                "status": "accepted",
                "target_side": None,
                "a": refs[0],
                "b": refs[1],
            }
        ],
    }
    install_overlay(corpus, overlay_from_plan(plan), plan)
    applied = refresh_vector_store(
        store_dir, corpus, embedder=FakeEmbedder(), timestamp="20990101T000000Z"
    )
    assert applied.refreshed and applied.n_embedded == 0
    assert [hit["chunk_id"] for hit in applied.new_store.retrieve(question, 10)] == baseline
    rollback_overlay(corpus)
    restored = refresh_vector_store(
        store_dir, corpus, embedder=FakeEmbedder(), timestamp="20990102T000000Z"
    )
    assert restored.refreshed and restored.n_embedded == 0
    assert [hit["chunk_id"] for hit in restored.new_store.retrieve(question, 10)] == baseline
