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
    assert retrieval.evaluate_retrieval([], k=5) == {
        "n": 0,
        "k": 5,
        "recall_at_k": 0.0,
        "mrr": 0.0,
        "span_char_coverage_at_k": 0.0,
        "span_intact_at_k": 0.0,
        "served_chars_at_k": 0.0,
    }


def test_served_chars_counts_the_top_k_text_an_overlap_included():
    # What the model is served, not the character union: an overlapped copy is served twice.
    retrieved = [
        {"doc_id": "a.txt", "char_start": 0, "char_end": 5, "text": "01234"},
        {"doc_id": "a.txt", "char_start": 3, "char_end": 8, "text": "34567"},
        {"doc_id": "a.txt", "char_start": 8, "char_end": 9, "text": "8"},
    ]
    assert retrieval.served_chars_at_k(retrieved, 10) == 11
    assert retrieval.served_chars_at_k(retrieved, 2) == 10  # the cut drops what it does not serve
    assert retrieval.evaluate_retrieval([(retrieved, [])], k=10)["served_chars_at_k"] == 11.0


def test_span_carried_whole_by_one_chunk_is_fully_covered_and_intact():
    retrieved = [chunk("a.txt", 0, 100)]
    spans = [span("a.txt", 40, 60)]
    assert retrieval.span_char_coverage_at_k(retrieved, spans, 10) == 1.0
    assert retrieval.span_intact_at_k(retrieved, spans, 10) == 1.0


def test_span_split_across_two_chunks_is_covered_but_not_intact():
    # The gold span straddles the boundary: both halves are retrieved, no chunk carries it whole.
    retrieved = [chunk("a.txt", 0, 50), chunk("a.txt", 50, 100)]
    spans = [span("a.txt", 40, 60)]
    assert retrieval.recall_at_k(retrieved, spans, 10) == 1.0
    assert retrieval.span_char_coverage_at_k(retrieved, spans, 10) == 1.0
    assert retrieval.span_intact_at_k(retrieved, spans, 10) == 0.0


def test_partly_retrieved_span_scores_its_character_share():
    # Only the first half of the span was retrieved: recall still fires, coverage says how much.
    retrieved = [chunk("a.txt", 0, 50)]
    spans = [span("a.txt", 40, 60)]
    assert retrieval.recall_at_k(retrieved, spans, 10) == 1.0
    assert retrieval.span_char_coverage_at_k(retrieved, spans, 10) == 0.5
    assert retrieval.span_intact_at_k(retrieved, spans, 10) == 0.0


def test_missed_span_scores_zero_on_both_intactness_metrics():
    retrieved = [chunk("a.txt", 0, 10)]
    spans = [span("a.txt", 40, 60)]
    assert retrieval.span_char_coverage_at_k(retrieved, spans, 10) == 0.0
    assert retrieval.span_intact_at_k(retrieved, spans, 10) == 0.0


def test_overlapping_chunks_count_each_character_once():
    retrieved = [chunk("a.txt", 0, 50), chunk("a.txt", 30, 55)]
    spans = [span("a.txt", 40, 60)]
    assert retrieval.span_char_coverage_at_k(retrieved, spans, 10) == 0.75  # 40..55 of 40..60


def test_intactness_is_averaged_over_an_items_spans():
    retrieved = [chunk("a.txt", 0, 100), chunk("b.txt", 0, 50)]
    spans = [span("a.txt", 10, 20), span("b.txt", 40, 60)]
    assert retrieval.span_intact_at_k(retrieved, spans, 10) == 0.5
    assert retrieval.span_char_coverage_at_k(retrieved, spans, 10) == 0.75  # (1.0 + 0.5) / 2


def test_intactness_respects_the_k_cutoff():
    retrieved = [chunk("a.txt", 900, 1000), chunk("a.txt", 0, 100)]
    spans = [span("a.txt", 10, 20)]
    assert retrieval.span_intact_at_k(retrieved, spans, 1) == 0.0
    assert retrieval.span_intact_at_k(retrieved, spans, 2) == 1.0


def test_intactness_reads_a_collapsed_chunks_other_occurrences():
    collapsed = {
        **chunk("a.txt", 0, 100),
        "metadata": {
            "duplicate_occurrences": [{"doc_id": "b.txt", "char_start": 200, "char_end": 300}]
        },
    }
    spans = [span("b.txt", 240, 260)]
    assert retrieval.span_char_coverage_at_k([collapsed], spans, 10) == 1.0
    assert retrieval.span_intact_at_k([collapsed], spans, 10) == 1.0


def test_an_item_labeling_no_span_is_vacuously_intact():
    assert retrieval.span_char_coverage_at_k([chunk("a.txt", 0, 10)], [], 10) == 1.0
    assert retrieval.span_intact_at_k([chunk("a.txt", 0, 10)], [], 10) == 1.0


def test_evaluate_retrieval_reports_the_intactness_pair_beside_recall():
    whole = ([chunk("a.txt", 0, 100)], [span("a.txt", 40, 60)])
    split = ([chunk("a.txt", 0, 50), chunk("a.txt", 50, 100)], [span("a.txt", 40, 60)])
    report = retrieval.evaluate_retrieval([whole, split], k=5)
    assert report["recall_at_k"] == 1.0  # both items hit, so recall cannot see the difference
    assert report["span_char_coverage_at_k"] == 1.0
    assert report["span_intact_at_k"] == 0.5
