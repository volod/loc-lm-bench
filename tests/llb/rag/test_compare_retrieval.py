"""GraphRAG backend residual 3 -- graph-vs-FAISS retrieval comparison core (`llb.rag.compare`).

Pure: driven by fake stores exposing the `.retrieve` seam, so it runs in the lightweight CI install
(no FAISS, no DuckDB, no GPU). The CLI wiring (`compare-retrieval`) layers real stores on top.
"""

from llb.cli.rag.compare_stores import _compare_vector_corpus_root
from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset
from llb.rag.compare import ROW_ORACLE_DOC, add_rerank_rows, compare_retrieval, format_comparison
from llb.rag.question_types import (
    aligned_question_types,
    load_question_types,
    load_question_types_by_question,
)


class _FakeStore:
    """A store that always returns the same fixed hits (truncated to k)."""

    def __init__(self, hits: list[ChunkRecord]) -> None:
        self._hits = hits

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        return self._hits[:k]


def _chunk(doc: str, start: int, end: int) -> ChunkRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "x"}


def _span(doc: str, start: int, end: int) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "g"}


def _items() -> list[tuple[str, list[SourceSpanRecord]]]:
    return [("питання", [_span("d1", 0, 10)])]


def test_compare_scores_each_backend_and_picks_recall_winner():
    stores = {
        "faiss": _FakeStore([_chunk("d1", 0, 10)]),  # overlaps the gold span -> hit
        "graph/local_khop": _FakeStore([_chunk("d1", 50, 60)]),  # no overlap -> miss
    }
    report = compare_retrieval(stores, _items(), k=5)
    assert report["k"] == 5 and report["n"] == 1
    assert report["backends"]["faiss"]["recall_at_k"] == 1.0
    assert report["backends"]["graph/local_khop"]["recall_at_k"] == 0.0
    assert report["best_recall"] == "faiss"


def test_compare_breaks_recall_ties_by_mrr_then_label():
    # both recall 1.0, but local_khop hits at rank 1 (higher MRR) vs faiss at rank 2
    stores = {
        "faiss": _FakeStore([_chunk("d1", 50, 60), _chunk("d1", 0, 10)]),
        "graph/local_khop": _FakeStore([_chunk("d1", 0, 10)]),
    }
    report = compare_retrieval(stores, _items(), k=5)
    assert report["backends"]["faiss"]["recall_at_k"] == 1.0
    assert report["backends"]["graph/local_khop"]["recall_at_k"] == 1.0
    assert report["best_recall"] == "graph/local_khop"  # higher MRR wins the tie


def test_compare_empty_backends_has_no_winner():
    report = compare_retrieval({}, _items(), k=3)
    assert report["best_recall"] is None
    assert report["backends"] == {}


def test_format_comparison_is_ascii_and_lists_backends():
    report = compare_retrieval({"faiss": _FakeStore([_chunk("d1", 0, 10)])}, _items(), k=5)
    text = format_comparison(report)
    assert text.isascii()  # AGENTS.md: ASCII-only output
    assert "faiss" in text and "recall@k" in text and "best (recall@k): faiss" in text


def test_format_comparison_handles_no_backends():
    text = format_comparison(compare_retrieval({}, _items(), k=5))
    assert "no backends loaded" in text


def test_compare_reports_question_type_slices_without_retrieving_twice():
    store = _FakeStore([_chunk("d1", 0, 10)])
    report = compare_retrieval(
        {"fused/local_khop": store},
        [*_items(), *_items()],
        k=5,
        slice_labels=["comparative", "multi-hop"],
    )
    assert report["slices"]["comparative"]["n"] == 1
    assert report["slices"]["multi-hop"]["backends"]["fused/local_khop"]["mrr"] == 1.0
    rendered = format_comparison(report)
    assert "slice comparative (n=1)" in rendered
    assert "slice multi-hop (n=1)" in rendered


def test_compare_rejects_misaligned_slice_labels():
    import pytest

    with pytest.raises(ValueError, match="align"):
        compare_retrieval({}, _items(), k=5, slice_labels=[])


def test_add_rerank_rows_pairs_each_backend_and_skips_the_oracle():
    # rerank-context-order: the reranked twin scores the SAME store's candidates after the
    # cross-encoder cut, so the report shows the pre/post-rerank delta per backend. A scorer
    # that ranks the gold-hitting chunk first lifts MRR from 1/2 to 1 on the reranked row.
    def gold_first_scorer(question: str, texts: list[str]) -> list[float]:
        return [1.0 if text == "gold" else 0.0 for text in texts]

    hits = [_chunk("d1", 50, 60), {**_chunk("d1", 0, 10), "text": "gold"}]
    stores = {"faiss": _FakeStore(hits), ROW_ORACLE_DOC: _FakeStore(hits)}
    rows = add_rerank_rows(stores, gold_first_scorer, candidates=5)
    assert set(rows) == {"faiss", "faiss+rerank", ROW_ORACLE_DOC}  # oracle gets no twin
    report = compare_retrieval(rows, _items(), k=2)
    assert report["backends"]["faiss"]["mrr"] == 0.5  # gold at rank 2 pre-rerank
    assert report["backends"]["faiss+rerank"]["mrr"] == 1.0  # reranked to rank 1
    assert report["best_recall"] == "faiss+rerank"


def test_compare_vector_stores_infers_sibling_corpus(tmp_path):
    root = tmp_path / "bundle"
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    goldset = root / "goldset.jsonl"
    goldset.write_text("", encoding="utf-8")

    assert _compare_vector_corpus_root(goldset, None) == corpus
    explicit = tmp_path / "other-corpus"
    assert _compare_vector_corpus_root(goldset, explicit) == explicit
    assert _compare_vector_corpus_root(tmp_path / "missing" / "goldset.jsonl", None) is None


def test_question_type_labels_find_parent_sidecar_for_accepted_ledger(tmp_path):
    import json

    accepted = tmp_path / "accepted"
    accepted.mkdir()
    goldset = accepted / "goldset.jsonl"
    goldset.write_text("", encoding="utf-8")
    rows = [
        {"id": "a", "question_type": "comparative"},
        {"id": "b", "question_type": "multi-hop"},
    ]
    (tmp_path / "needle_items.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    assert aligned_question_types(goldset, ["b", "missing", "a"]) == [
        "multi-hop",
        None,
        "comparative",
    ]
    assert load_question_types(goldset) == {"a": "comparative", "b": "multi-hop"}


def test_question_type_labels_are_absent_without_a_needle_sidecar(tmp_path):
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text("", encoding="utf-8")
    assert aligned_question_types(goldset, ["a"]) is None
    assert load_question_types(goldset) == {}


def test_question_type_map_omits_duplicate_question_text_with_conflicting_labels(tmp_path):
    goldset = tmp_path / "goldset.jsonl"
    items = [
        GoldItem(
            id=item_id,
            question=question,
            reference_answer="x",
            source_doc_id="d",
            source_spans=[SourceSpan(doc_id="d", char_start=0, char_end=1, text="x")],
            provenance="ontology-drafted",
            split="final",
        )
        for item_id, question in (("a", "same"), ("b", "same"), ("c", "unique"))
    ]
    dump_goldset(items, goldset)
    (tmp_path / "needle_items.jsonl").write_text(
        "\n".join(
            [
                '{"id":"a","question_type":"factoid"}',
                '{"id":"b","question_type":"multi-hop"}',
                '{"id":"c","question_type":"comparative"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_question_types_by_question(goldset) == {"unique": "comparative"}
