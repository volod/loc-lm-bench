"""Parent-child store logic (pure, no FAISS/embedder needed)."""

import pytest

from llb.core.config import DEFAULT_EMBEDDING_MODEL
from llb.rag.store import (
    RagStore,
    _build_children,
    _children_to_parents,
    store_embedder_mismatch,
)


def test_build_rejects_unknown_retrieval_mode(tmp_path):
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        RagStore.build(tmp_path, mode="typo")


def test_store_embedder_mismatch_flags_a_different_encoder():
    meta = {"embedding_model": "BAAI/bge-m3"}
    assert store_embedder_mismatch(meta, "BAAI/bge-m3") is None  # store == config: fine
    assert store_embedder_mismatch(meta, "intfloat/multilingual-e5-base") == "BAAI/bge-m3"


def test_store_embedder_mismatch_defaults_to_the_pinned_model():
    assert store_embedder_mismatch({}, DEFAULT_EMBEDDING_MODEL) is None


def test_children_to_parents_dedups_and_preserves_rank():
    parents = {
        "p1": {"chunk_id": "p1", "doc_id": "d.txt", "char_start": 0, "char_end": 100, "text": "a"},
        "p2": {
            "chunk_id": "p2",
            "doc_id": "d.txt",
            "char_start": 100,
            "char_end": 200,
            "text": "b",
        },
    }
    child_hits = [
        {"chunk_id": "p1::c0", "parent_id": "p1", "retrieval_score": 0.9},
        {"chunk_id": "p1::c1", "parent_id": "p1", "retrieval_score": 0.8},  # same parent
        {"chunk_id": "p2::c0", "parent_id": "p2", "retrieval_score": 0.7},
    ]
    out = _children_to_parents(child_hits, parents)
    assert [p["chunk_id"] for p in out] == ["p1", "p2"]  # one row per unique parent
    assert [p["rank"] for p in out] == [1, 2]
    assert out[0]["matched_child_id"] == "p1::c0"  # first (best) child wins
    assert out[0]["retrieval_score"] == 0.9


def test_children_to_parents_skips_unknown_parent():
    out = _children_to_parents([{"chunk_id": "x::c0", "parent_id": "missing"}], {})
    assert out == []


def test_build_children_shifts_offsets_and_links_parent():
    parent_text = "Перше речення тут. Друге речення також."
    parents = [
        {
            "chunk_id": "d.txt#sentence#0000",
            "doc_id": "d.txt",
            "char_start": 50,
            "char_end": 50 + len(parent_text),
            "text": parent_text,
            "metadata": {},
        }
    ]
    children = _build_children(parents, "sentence", child_size=20, overlap=0, embedder=None)
    assert children
    for c in children:
        assert c["parent_id"] == "d.txt#sentence#0000"
        assert c["char_start"] >= 50  # offset shifted by the parent start
        # child text is the exact source slice relative to the parent
        rel = c["char_start"] - 50
        assert parent_text[rel : rel + (c["char_end"] - c["char_start"])] == c["text"]


def test_parent_child_retrieval_expands_until_k_unique_parents():
    class FakeEmbedder:
        def encode_queries(self, texts):
            return [[1.0]]

    class FakeIndex:
        def __init__(self):
            self.search_sizes = []

        def search(self, query, k):
            self.search_sizes.append(k)
            return [[1.0 - i / 100 for i in range(k)]], [list(range(k))]

    parents = [
        {"chunk_id": "p1", "doc_id": "d", "char_start": 0, "char_end": 9, "text": "a"},
        {"chunk_id": "p2", "doc_id": "d", "char_start": 9, "char_end": 10, "text": "b"},
    ]
    children = [{"chunk_id": f"c{i}", "parent_id": "p1" if i < 9 else "p2"} for i in range(10)]
    index = FakeIndex()
    store = RagStore(children, index, FakeEmbedder(), {"mode": "parent_child"}, parents)

    hits = store.retrieve("question", 2)

    assert [hit["chunk_id"] for hit in hits] == ["p1", "p2"]
    assert index.search_sizes == [8, 10]
