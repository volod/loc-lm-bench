"""Focused tests split from ``test_fusion_evidence.py``."""

import pytest
from _fusion_evidence_helpers import (
    _ByQuestion,
    _chunk,
    _multi_hop_item,
    _span,
)

from llb.rag.fusion_evidence import (
    EvidenceItem,
    build_sweep_rows,
    evaluate_fusion_evidence,
    format_report,
)
from llb.rag.fusion_evidence.paired import (
    paired_comparison,
    sign_test_p,
)
from llb.rag.fusion_evidence.rows import (
    VECTOR_ROW,
    LaneCache,
)
from llb.rag.fusion_evidence.stats import bootstrap_index_sets


def test_all_spans_requires_every_labeled_span_not_just_one():
    from llb.rag.retrieval import all_spans_at_k, recall_at_k, span_coverage_at_k

    spans = [_span("d1", 0, 10), _span("d2", 0, 10)]
    one_hop = [_chunk("d1", 0, 10)]
    assert recall_at_k(one_hop, spans, 10) == 1.0  # the flat metric is already satisfied
    assert span_coverage_at_k(one_hop, spans, 10) == 0.5
    assert all_spans_at_k(one_hop, spans, 10) == 0.0
    both = [_chunk("d1", 0, 10), _chunk("d2", 0, 10)]
    assert all_spans_at_k(both, spans, 10) == 1.0


def test_span_coverage_of_an_unlabeled_item_is_complete():
    from llb.rag.retrieval import span_coverage_at_k

    assert span_coverage_at_k([], [], 10) == 1.0


def test_paired_bootstrap_is_deterministic_and_brackets_the_delta():
    index_sets = bootstrap_index_sets(4, resamples=200, seed=13)
    candidate, baseline = [1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]
    first = paired_comparison(candidate, baseline, index_sets)
    second = paired_comparison(candidate, baseline, bootstrap_index_sets(4, 200, 13))
    assert first == second
    assert first["delta"]["mean"] == pytest.approx(0.5)
    assert first["delta"]["lo"] <= first["delta"]["mean"] <= first["delta"]["hi"]
    assert (first["wins"], first["losses"], first["ties"]) == (2, 0, 2)


def test_bootstrap_ratio_handles_route_precision_and_a_zero_denominator():
    from llb.rag.fusion_evidence.stats import bootstrap_ratio

    index_sets = bootstrap_index_sets(3, resamples=20, seed=13)
    measured = bootstrap_ratio([True, False, False], [True, True, False], index_sets)
    assert measured["mean"] == pytest.approx(0.5)
    assert measured["lo"] <= measured["mean"] <= measured["hi"]
    empty_route = bootstrap_ratio([False], [False], [[0]])
    assert {key: empty_route[key] for key in ("mean", "lo", "hi")} == {
        "mean": 0.0,
        "lo": 0.0,
        "hi": 0.0,
    }
    assert empty_route["stability"]["reading"] == "flat"
    assert empty_route["stability"]["borderline"] is False


def test_sign_test_is_two_sided_and_symmetric():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 0) == pytest.approx(2 * 0.5**5)
    assert sign_test_p(5, 0) == sign_test_p(0, 5)


def test_paired_comparison_rejects_misaligned_vectors():
    with pytest.raises(ValueError, match="one baseline value"):
        paired_comparison([1.0], [], [])


def test_sweep_rows_retrieve_each_lane_once_per_question():
    vector = _ByQuestion({"q": [_chunk("d1", 0, 10)]})
    graph = _ByQuestion({"q": [_chunk("d2", 0, 10)]})
    rows = build_sweep_rows(
        vector, {"local_khop": graph}, ["q", "q"], k=10, weights=(0.0, 0.3, 1.0)
    )
    assert vector.calls == 1 and graph.calls == 1  # deduplicated questions, one pass per lane
    assert set(rows) == {
        VECTOR_ROW,
        "graph/local_khop",
        "fused/local_khop@0.00/d10",
        "fused/local_khop@0.30/d10",
        "fused/local_khop@1.00/d10",
    }
    # endpoint weights stay exact lane passthroughs
    assert rows["fused/local_khop@0.00/d10"].retrieve("q", 10) == vector.retrieve("q", 10)
    assert rows["fused/local_khop@1.00/d10"].retrieve("q", 10) == graph.retrieve("q", 10)
    fused = rows["fused/local_khop@0.30/d10"].retrieve("q", 10)
    assert {chunk["doc_id"] for chunk in fused} == {"d1", "d2"}


def test_sweep_adds_one_routed_row_and_reports_its_decision_counts():
    vector = _ByQuestion({"single": [_chunk("d1", 0, 10)], "multi": [_chunk("d1", 0, 10)]})
    graph = _ByQuestion({"single": [_chunk("d2", 0, 10)], "multi": [_chunk("d2", 0, 10)]})
    rows = build_sweep_rows(
        vector,
        {"local_khop": graph},
        ["single", "multi"],
        k=2,
        weights=(0.3,),
        routed_graph_weight=0.3,
        question_types={"single": "factoid", "multi": "multi-hop"},
    )
    routed = rows["routed/local_khop@0.30/d2"]
    assert routed.retrieve("single", 2) == rows[VECTOR_ROW].retrieve("single", 2)
    report = evaluate_fusion_evidence(
        rows,
        [
            EvidenceItem("s", "single", [_span("d1", 0, 10)], "factoid"),
            _multi_hop_item("m", "multi"),
        ],
        2,
        baseline=VECTOR_ROW,
        resamples=20,
    )
    assert report["rows"]["routed/local_khop@0.30/d2"]["routing"] == {
        "graph_questions": 1,
        "vector_questions": 1,
        "sidecar_questions": 2,
        "heuristic_questions": 0,
        "slices": {
            "factoid": {"graph_questions": 0, "vector_questions": 1},
            "multi-hop": {"graph_questions": 1, "vector_questions": 0},
        },
    }
    rendered = format_report(report)
    assert "### Question routing" in rendered


def test_replayed_fusion_matches_the_production_fused_retriever():
    from llb.rag.fusion import FusedRetriever

    hits = {"q": [_chunk("d1", 0, 10), _chunk("d1", 20, 30)]}
    graph_hits = {"q": [_chunk("d2", 0, 10), _chunk("d1", 20, 30)]}
    vector, graph = _ByQuestion(hits), _ByQuestion(graph_hits)
    rows = build_sweep_rows(vector, {"local_khop": graph}, ["q"], k=3, weights=(0.3,))
    live = FusedRetriever(_ByQuestion(hits), _ByQuestion(graph_hits), 0.3).retrieve("q", 3)
    assert rows["fused/local_khop@0.30/d3"].retrieve("q", 3) == live


def test_lane_cache_never_returns_more_than_the_swept_depth():
    cache = LaneCache(
        _ByQuestion({"q": [_chunk("d1", i, i + 1) for i in range(5)]}), ["q"], depth=2
    )
    assert len(cache.retrieve("q", 10)) == 2
    assert cache.retrieve("missing", 10) == []
