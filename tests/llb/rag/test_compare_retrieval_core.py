"""GraphRAG backend residual 3 -- graph-vs-FAISS retrieval comparison core (`llb.rag.compare`).

Pure: driven by fake stores exposing the `.retrieve` seam, so it runs in the lightweight CI install
(no FAISS, no DuckDB, no GPU). The CLI wiring (`compare-retrieval`) layers real stores on top.
"""

from llb.rag.compare import (
    ROW_ORACLE_DOC,
    add_rerank_rows,
    compare_retrieval,
    duplicate_census,
    format_comparison,
)


from _compare_retrieval_helpers import (
    _FakeStore,
    _chunk,
    _items,
    _MetaStore,
)


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


def test_duplicate_census_reads_only_stores_with_build_meta():
    stats = {
        "n": 4,
        "unique": 3,
        "collapsed": 1,
        "duplicate_chunks": 2,
        "duplicate_share": 0.5,
        "groups": 1,
        "largest_group": 2,
        "intra_document_groups": 1,
        "cross_document_groups": 0,
    }
    stores = {
        "faiss": _MetaStore([_chunk("d1", 0, 10)], stats),
        "graph/local_khop": _FakeStore([_chunk("d1", 0, 10)]),  # no meta -> no census row
    }
    census = duplicate_census(stores)
    assert set(census) == {"faiss"}
    report = compare_retrieval(stores, _items(), k=5)
    report["duplicates"] = census
    rendered = format_comparison(report)
    assert "1 intra-document, 0 cross-document" in rendered
    assert rendered.isascii()


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
