"""Hybrid RagStore fusion, metadata filter seam, refusal paths, and the exact-term
lexical-win regression over the committed `samples/goldsets/exact_terms_uk` fixture.

Pure: a fake dense index + fake embedder stand in for FAISS / sentence-transformers, the
BM25 side is the real `LexicalIndex`, so the whole hybrid path runs in the lightweight CI
install (no GPU, no [rag] extra).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llb.core.config import RunConfig
from llb.goldset.schema import load_goldset
from llb.rag.chunking import chunk_corpus
from llb.rag.compare import OracleDocFilter
from llb.rag.filters import metadata_filter
from llb.rag.lexical import LexicalIndex
from llb.rag.retrieval import evaluate_retrieval
from llb.rag.store import LEXICAL_FILE, META_FILE, MODE_HYBRID, RagStore

FIXTURE = Path("samples/goldsets/exact_terms_uk")


class FakeEmbedder:
    def encode_queries(self, texts):
        return [[1.0]]


class FakeIndex:
    """Deterministic dense side: returns ids in a FIXED order with descending scores."""

    def __init__(self, order):
        self.order = list(order)

    def search(self, query, k):
        ids = self.order[:k]
        return [[1.0 - i / 1000 for i in range(len(ids))]], [ids]


def _chunks(n):
    return [
        {
            "doc_id": f"d{i % 2}.md",
            "chunk_id": f"c{i}",
            "char_start": i * 10,
            "char_end": i * 10 + 10,
            "text": f"текст {i}",
            "metadata": {},
        }
        for i in range(n)
    ]


def _hybrid_store(chunks, dense_order, lexical, weight=0.5, candidates=50):
    meta = {"mode": MODE_HYBRID, "embedding_model": "fake", "lexical": {"lemmatize": False}}
    store = RagStore(chunks, FakeIndex(dense_order), FakeEmbedder(), meta, lexical=lexical)
    store.fusion_weight = weight
    store.fusion_candidates = candidates
    return store


def test_hybrid_retrieve_fuses_dense_and_lexical_rankings():
    chunks = _chunks(4)
    chunks[3]["text"] = "наказ 4821 про фінансування"
    lexical = LexicalIndex.build([c["text"] for c in chunks])
    # dense puts the lexically-right chunk LAST; fusion must pull it up
    store = _hybrid_store(chunks, [0, 1, 2, 3], lexical, weight=0.3)
    hits = store.retrieve("наказ 4821", k=2)
    assert hits[0]["chunk_id"] == "c3"
    assert hits[0]["rank"] == 1 and hits[1]["rank"] == 2
    assert hits[0]["retrieval_score"] > hits[1]["retrieval_score"]


def test_hybrid_weight_one_reproduces_the_dense_order():
    chunks = _chunks(3)
    lexical = LexicalIndex.build([c["text"] for c in chunks])
    store = _hybrid_store(chunks, [2, 0, 1], lexical, weight=1.0)
    assert [h["chunk_id"] for h in store.retrieve("текст", k=3)] == ["c2", "c0", "c1"]


def test_chunk_filter_applies_before_fusion_on_both_sides():
    chunks = _chunks(4)  # doc ids alternate d0/d1
    lexical = LexicalIndex.build([c["text"] for c in chunks])
    store = _hybrid_store(chunks, [0, 1, 2, 3], lexical)
    hits = store.retrieve("текст", k=4, chunk_filter=metadata_filter(doc_ids={"d1.md"}))
    assert hits and all(h["doc_id"] == "d1.md" for h in hits)


def test_flat_store_applies_chunk_filter_and_renumbers_ranks():
    chunks = _chunks(4)
    store = RagStore(chunks, FakeIndex([0, 1, 2, 3]), FakeEmbedder(), {"mode": "flat"})
    hits = store.retrieve("текст", k=2, chunk_filter=metadata_filter(doc_ids={"d1.md"}))
    assert [h["chunk_id"] for h in hits] == ["c1", "c3"]
    assert [h["rank"] for h in hits] == [1, 2]


def test_load_refuses_hybrid_store_without_lexical_file(tmp_path):
    (tmp_path / "chunks.jsonl").write_text("", encoding="utf-8")
    (tmp_path / META_FILE).write_text(json.dumps({"mode": MODE_HYBRID}), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing its lexical index"):
        RagStore.load(tmp_path)


def test_run_eval_refuses_hybrid_config_over_dense_store(monkeypatch, tmp_path):
    from llb.executor import runner
    from llb.rag import store as store_mod

    cfg = RunConfig(retrieval_mode="hybrid", data_dir=tmp_path)
    fake = SimpleNamespace(meta={"embedding_model": cfg.embedding_model}, lexical=None)
    monkeypatch.setattr(store_mod.RagStore, "load", classmethod(lambda cls, d: fake))
    with pytest.raises(SystemExit, match="retrieval-mode hybrid"):
        runner._load_store(cfg)


def test_run_eval_applies_fusion_knobs_from_the_config(monkeypatch, tmp_path):
    from llb.executor import runner
    from llb.rag import store as store_mod

    cfg = RunConfig(
        retrieval_mode="hybrid", fusion_weight=0.7, fusion_candidates=30, data_dir=tmp_path
    )
    fake = SimpleNamespace(
        meta={"embedding_model": cfg.embedding_model, "mode": MODE_HYBRID},
        lexical=object(),
        fusion_weight=0.5,
        fusion_candidates=50,
    )
    monkeypatch.setattr(store_mod.RagStore, "load", classmethod(lambda cls, d: fake))
    store = runner._load_store(cfg)
    assert store.fusion_weight == 0.7 and store.fusion_candidates == 30


def test_hybrid_save_load_round_trip_keeps_lexical_search(tmp_path):
    chunks = _chunks(3)
    chunks[1]["text"] = "унікальний наказ 4899"
    lexical = LexicalIndex.build([c["text"] for c in chunks])
    lexical.save(tmp_path / LEXICAL_FILE)
    loaded = LexicalIndex.load(tmp_path / LEXICAL_FILE)
    assert loaded.search("4899", k=1)[0][0] == 1


def test_oracle_doc_filter_scopes_each_question_to_its_gold_doc():
    chunks = _chunks(4)
    store = RagStore(chunks, FakeIndex([0, 1, 2, 3]), FakeEmbedder(), {"mode": "flat"})
    items = [("питання", [{"doc_id": "d1.md", "char_start": 10, "char_end": 20, "text": "x"}])]
    oracle = OracleDocFilter(store, items)
    hits = oracle.retrieve("питання", k=4)
    assert hits and all(h["doc_id"] == "d1.md" for h in hits)
    # unknown question -> unfiltered passthrough
    assert len(oracle.retrieve("інше", k=4)) == 4


@pytest.mark.slow
def test_exact_term_fixture_hybrid_strictly_beats_dense():
    """The lexical-win regression: on the committed exact-term registry, hybrid recall@10
    must strictly beat a dense side that confuses the near-identical entries (modeled as a
    fixed arbitrary dense order)."""
    chunks = chunk_corpus(FIXTURE / "corpus", "recursive", 800, 120, None)
    assert len(chunks) > 10  # recall@10 must be non-trivial
    items = load_goldset(FIXTURE / "goldset.jsonl")
    lexical = LexicalIndex.build([c["text"] for c in chunks])
    dense_order = list(range(len(chunks)))  # near-duplicates: dense adds no signal
    dense = RagStore(chunks, FakeIndex(dense_order), FakeEmbedder(), {"mode": "flat"})
    hybrid = _hybrid_store(chunks, dense_order, lexical, weight=0.5)

    def spans(item):
        return [s.model_dump() for s in item.source_spans]

    k = 10
    dense_metrics = evaluate_retrieval(
        [(dense.retrieve(i.question, k), spans(i)) for i in items], k
    )
    hybrid_metrics = evaluate_retrieval(
        [(hybrid.retrieve(i.question, k), spans(i)) for i in items], k
    )
    assert hybrid_metrics["recall_at_k"] > dense_metrics["recall_at_k"]
    assert hybrid_metrics["mrr"] > dense_metrics["mrr"]


@pytest.mark.slow
def test_stored_chunk_text_is_byte_identical_with_lemmatization_on():
    chunks = chunk_corpus(FIXTURE / "corpus", "recursive", 800, 120, None)
    source = (FIXTURE / "corpus" / "orders_registry_uk.md").read_text(encoding="utf-8")
    LexicalIndex.build(
        [c["text"] for c in chunks], lemmatize=True, lemmatizer=lambda t: t.rstrip("ааіи")
    )
    for chunk in chunks:
        assert chunk["text"] == source[chunk["char_start"] : chunk["char_end"]]
