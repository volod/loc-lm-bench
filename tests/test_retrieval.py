from llb.rag import retrieval


def chunk(doc, start, end):
    return {"doc_id": doc, "char_start": start, "char_end": end}


def span(doc, start, end):
    return {"doc_id": doc, "char_start": start, "char_end": end}


def test_overlap_requires_shared_chars():
    assert retrieval.spans_overlap(0, 10, 5, 15)
    assert not retrieval.spans_overlap(0, 10, 10, 20)  # touching, not overlapping


def test_hit_needs_same_doc():
    c = chunk("a.txt", 0, 100)
    assert retrieval.chunk_hits_span(c, span("a.txt", 50, 60))
    assert not retrieval.chunk_hits_span(c, span("b.txt", 50, 60))


def test_first_hit_rank_and_recall():
    retrieved = [chunk("a.txt", 200, 300), chunk("a.txt", 0, 100)]
    spans = [span("a.txt", 50, 60)]
    assert retrieval.first_hit_rank(retrieved, spans) == 2
    assert retrieval.recall_at_k(retrieved, spans, 1) == 0.0  # miss in top-1
    assert retrieval.recall_at_k(retrieved, spans, 2) == 1.0  # hit by top-2


def test_reciprocal_rank_and_miss():
    retrieved = [chunk("a.txt", 0, 100)]
    assert retrieval.reciprocal_rank(retrieved, [span("a.txt", 10, 20)]) == 1.0
    assert retrieval.reciprocal_rank(retrieved, [span("z.txt", 10, 20)]) == 0.0


def test_evaluate_retrieval_aggregates():
    hit = ([chunk("a.txt", 0, 100)], [span("a.txt", 10, 20)])
    miss = ([chunk("a.txt", 0, 100)], [span("b.txt", 10, 20)])
    report = retrieval.evaluate_retrieval([hit, miss], k=5)
    assert report["n"] == 2
    assert report["recall_at_k"] == 0.5
    assert report["mrr"] == 0.5


def test_evaluate_retrieval_empty():
    assert retrieval.evaluate_retrieval([], k=5) == {"n": 0, "k": 5, "recall_at_k": 0.0, "mrr": 0.0}
