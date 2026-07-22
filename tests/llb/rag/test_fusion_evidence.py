"""graph-vector-fusion-multihop-evidence -- the multi-hop fusion evidence lane.

Pure: driven by fake stores exposing the `.retrieve` seam, so it runs in the lightweight CI
install (no FAISS, no DuckDB, no GPU). The CLI wiring layers real stores on top.
"""

import pytest

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.rag.fusion_evidence import (
    EvidenceItem,
    build_sweep_rows,
    evaluate_fusion_evidence,
    format_report,
    parse_candidates,
    parse_span_identities,
    parse_weights,
)
from llb.rag.fusion_evidence.models import (
    METRIC_ALL_SPANS,
    METRIC_RECALL,
    VERDICT_ADOPT,
    VERDICT_INCONCLUSIVE,
    VERDICT_NO_EVIDENCE,
    VERDICT_REJECT,
)
from llb.rag.fusion_evidence.rows import VECTOR_ROW, LaneCache
from llb.rag.fusion_evidence.stats import bootstrap_index_sets, paired_comparison, sign_test_p


def _chunk(doc: str, start: int, end: int) -> ChunkRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "x"}


def _span(doc: str, start: int, end: int) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "g"}


class _ByQuestion:
    """A store returning a fixed ranking per question (truncated to k)."""

    def __init__(self, hits: dict[str, list[ChunkRecord]]) -> None:
        self.hits = hits
        self.calls = 0

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        self.calls += 1
        return self.hits.get(question, [])[:k]


def _multi_hop_item(item_id: str, question: str) -> EvidenceItem:
    return EvidenceItem(item_id, question, [_span("d1", 0, 10), _span("d2", 0, 10)], "multi-hop")


# --- multi-span metrics -------------------------------------------------------------------


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


# --- statistics ---------------------------------------------------------------------------


def test_paired_bootstrap_is_deterministic_and_brackets_the_delta():
    index_sets = bootstrap_index_sets(4, resamples=200, seed=13)
    candidate, baseline = [1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]
    first = paired_comparison(candidate, baseline, index_sets)
    second = paired_comparison(candidate, baseline, bootstrap_index_sets(4, 200, 13))
    assert first == second
    assert first["delta"]["mean"] == pytest.approx(0.5)
    assert first["delta"]["lo"] <= first["delta"]["mean"] <= first["delta"]["hi"]
    assert (first["wins"], first["losses"], first["ties"]) == (2, 0, 2)


def test_sign_test_is_two_sided_and_symmetric():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 0) == pytest.approx(2 * 0.5**5)
    assert sign_test_p(5, 0) == sign_test_p(0, 5)


def test_paired_comparison_rejects_misaligned_vectors():
    with pytest.raises(ValueError, match="one baseline value"):
        paired_comparison([1.0], [], [])


# --- sweep rows ---------------------------------------------------------------------------


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


def test_parse_weights_dedupes_and_rejects_out_of_range():
    assert parse_weights("0, 0.3 ,0.3, 1") == (0.0, 0.3, 1.0)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        parse_weights("1.5")
    with pytest.raises(ValueError, match="no graph weight"):
        parse_weights(" , ")


# --- candidate depth ----------------------------------------------------------------------


def test_parse_candidates_reads_k_as_the_scored_cutoff_and_rejects_a_zero_depth():
    assert parse_candidates("k, 50 ,50, 20") == (None, 50, 20)
    with pytest.raises(ValueError, match="at least 1"):
        parse_candidates("0")
    with pytest.raises(ValueError, match="an integer or 'k'"):
        parse_candidates("deep")
    with pytest.raises(ValueError, match="no candidate depth"):
        parse_candidates(" , ")


def test_a_deeper_pool_surfaces_the_span_both_lanes_agree_on():
    # `d2` is the span BOTH lanes rank -- vector rank 4 (below a k=3 cutoff) and graph rank 2.
    # At depth k its vector evidence is invisible, so the fused row spends its third seat on the
    # graph lane's own top hit; at depth 4 the two lanes' agreement outranks that graph-only hit.
    vector = _ByQuestion(
        {
            "q": [
                _chunk("d1", 0, 10),
                _chunk("d1", 20, 30),
                _chunk("d1", 40, 50),
                _chunk("d2", 0, 10),
            ]
        }
    )
    graph = _ByQuestion(
        {"q": [_chunk("d9", 0, 10), _chunk("d2", 0, 10), _chunk("d8", 0, 10)]},
    )
    rows = build_sweep_rows(
        vector, {"local_khop": graph}, ["q"], k=3, weights=(0.3,), candidates=(None, 4)
    )
    shallow = [chunk["doc_id"] for chunk in rows["fused/local_khop@0.30/d3"].retrieve("q", 3)]
    deep = [chunk["doc_id"] for chunk in rows["fused/local_khop@0.30/d4"].retrieve("q", 3)]
    assert shallow == ["d1", "d1", "d9"]
    assert deep == ["d1", "d1", "d2"]


def test_depths_resolve_against_k_and_deduplicate_into_one_row():
    vector = _ByQuestion({"q": [_chunk("d1", 0, 10)]})
    graph = _ByQuestion({"q": [_chunk("d2", 0, 10)]})
    rows = build_sweep_rows(
        vector,
        {"local_khop": graph},
        ["q"],
        k=10,
        weights=(0.0, 0.3, 1.0),
        candidates=(None, 4, 10, 50),
    )
    assert set(rows) == {
        VECTOR_ROW,
        "graph/local_khop",
        # endpoint weights are lane passthroughs, so they carry no depth variants
        "fused/local_khop@0.00/d10",
        "fused/local_khop@1.00/d10",
        "fused/local_khop@0.30/d10",
        "fused/local_khop@0.30/d50",
    }
    assert vector.calls == 1 and graph.calls == 1  # one pass per lane, at the deepest pool


def test_depth_equal_to_k_reproduces_the_default_fused_row_exactly():
    hits = {"q": [_chunk("d1", 0, 10), _chunk("d1", 20, 30)]}
    graph_hits = {"q": [_chunk("d2", 0, 10), _chunk("d1", 20, 30)]}
    default = build_sweep_rows(
        _ByQuestion(hits), {"local_khop": _ByQuestion(graph_hits)}, ["q"], k=2, weights=(0.3,)
    )
    explicit = build_sweep_rows(
        _ByQuestion(hits),
        {"local_khop": _ByQuestion(graph_hits)},
        ["q"],
        k=2,
        weights=(0.3,),
        candidates=(2,),
    )
    row = "fused/local_khop@0.30/d2"
    assert explicit[row].retrieve("q", 2) == default[row].retrieve("q", 2)


# --- evaluation + verdict -----------------------------------------------------------------


def _fusion_report(**kwargs):
    """One multi-hop item the vector lane half-covers and the graph lane completes."""
    items = [
        _multi_hop_item("mh-1", "q1"),
        EvidenceItem("f-1", "q2", [_span("d1", 0, 10)], "factoid"),
    ]
    vector = _ByQuestion({"q1": [_chunk("d1", 0, 10)], "q2": [_chunk("d1", 0, 10)]})
    graph = _ByQuestion({"q1": [_chunk("d2", 0, 10)], "q2": []})
    rows = build_sweep_rows(
        vector, {"local_khop": graph}, [i.question for i in items], k=10, weights=(0.0, 0.3)
    )
    return evaluate_fusion_evidence(rows, items, 10, baseline=VECTOR_ROW, resamples=100, **kwargs)


def test_fusion_that_completes_the_multi_hop_evidence_is_adopted():
    report = _fusion_report()
    focus = report["rows"]["fused/local_khop@0.30/d10"]["slices"]["multi-hop"]
    assert focus["n"] == 1
    assert focus["metrics"][METRIC_ALL_SPANS]["mean"] == 1.0
    assert (
        report["rows"][VECTOR_ROW]["slices"]["multi-hop"]["metrics"][METRIC_ALL_SPANS]["mean"]
        == 0.0
    )
    assert focus["paired_vs_baseline"][METRIC_ALL_SPANS]["wins"] == 1
    verdict = report["verdict"]
    assert verdict["decision"] == VERDICT_ADOPT
    assert verdict["best_row"] == "fused/local_khop@0.30/d10"
    assert verdict["focus_n"] == 1


def test_zero_weight_fused_row_ties_the_vector_baseline_exactly():
    report = _fusion_report()
    passthrough = report["rows"]["fused/local_khop@0.00/d10"]["overall"]
    for metric, comparison in passthrough["paired_vs_baseline"].items():
        assert comparison["wins"] == comparison["losses"] == 0, metric
        assert comparison["delta"]["mean"] == 0.0, metric


def test_no_multi_hop_item_yields_no_evidence_not_a_recommendation():
    items = [EvidenceItem("f-1", "q", [_span("d1", 0, 10)], "factoid")]
    vector = _ByQuestion({"q": [_chunk("d1", 0, 10)]})
    graph = _ByQuestion({"q": []})
    rows = build_sweep_rows(vector, {"local_khop": graph}, ["q"], k=10, weights=(0.3,))
    report = evaluate_fusion_evidence(rows, items, 10, baseline=VECTOR_ROW, resamples=50)
    assert report["verdict"]["decision"] == VERDICT_NO_EVIDENCE
    assert report["verdict"]["focus_n"] == 0
    assert report["focus_items"] == []


def test_a_multi_hop_gain_paid_for_in_overall_recall_is_rejected():
    # A heavy graph share completes the multi-hop item's second span but crowds the factoid's
    # only gold chunk out of the top-k: a real gain that the overall lane pays for.
    items = [
        _multi_hop_item("mh-1", "q1"),
        EvidenceItem("f-1", "q2", [_span("d1", 0, 10)], "factoid"),
    ]
    vector = _ByQuestion(
        {
            "q1": [_chunk("d1", 0, 10), _chunk("d3", 0, 10)],
            "q2": [_chunk("d8", 0, 10), _chunk("d1", 0, 10)],
        }
    )
    graph = _ByQuestion(
        {"q1": [_chunk("d2", 0, 10)], "q2": [_chunk("d7", 0, 10), _chunk("d6", 0, 10)]}
    )
    rows = build_sweep_rows(vector, {"local_khop": graph}, ["q1", "q2"], k=2, weights=(0.9,))
    report = evaluate_fusion_evidence(rows, items, 2, baseline=VECTOR_ROW, resamples=50)
    fused = report["rows"]["fused/local_khop@0.90/d2"]
    assert fused["slices"]["multi-hop"]["paired_vs_baseline"][METRIC_ALL_SPANS]["wins"] == 1
    assert fused["overall"]["paired_vs_baseline"][METRIC_RECALL]["delta"]["mean"] < 0
    assert report["verdict"]["decision"] == VERDICT_REJECT
    assert "overall recall@k" in report["verdict"]["reason"]


def test_evaluate_rejects_an_unknown_baseline_row():
    with pytest.raises(ValueError, match="baseline row"):
        evaluate_fusion_evidence({}, [], 10, baseline=VECTOR_ROW)


def test_report_is_ascii_and_carries_the_slice_uncertainty_and_item_ledger():
    text = format_report(_fusion_report())
    assert text.isascii()  # AGENTS.md: ASCII-only output
    assert "Focus slice: multi-hop" in text
    assert "all-spans@k" in text and "bootstrap CI" in text
    assert "Item-level outcomes (multi-hop)" in text and "mh-1" in text
    assert "Slice: factoid" in text


def test_report_says_so_when_the_multi_hop_slice_is_empty():
    items = [EvidenceItem("f-1", "q", [_span("d1", 0, 10)], "factoid")]
    rows = build_sweep_rows(
        _ByQuestion({"q": [_chunk("d1", 0, 10)]}),
        {"local_khop": _ByQuestion({})},
        ["q"],
        10,
        (0.3,),
    )
    text = format_report(
        evaluate_fusion_evidence(rows, items, 10, baseline=VECTOR_ROW, resamples=0)
    )
    assert "No multi-hop item was scored." in text
    # an all-zero metric table for an empty slice would read like a measured result
    assert "No item falls in this slice" in text
    assert text.count("| vector |") == 2  # overall + factoid only, not the empty focus slice


def test_a_gain_whose_interval_includes_zero_is_inconclusive_not_adopted():
    # Four multi-hop items, one of which fusion recovers: the mean delta is +0.25 but resamples
    # that omit the single winner make the interval touch zero, so the lane must NOT recommend.
    items = [_multi_hop_item(f"mh-{i}", f"q{i}") for i in range(4)]
    covered = [_chunk("d1", 0, 10), _chunk("d2", 0, 10)]
    vector = _ByQuestion({"q0": [], "q1": covered, "q2": covered, "q3": covered})
    graph = _ByQuestion({"q0": [_chunk("d1", 0, 10)], "q1": [], "q2": [], "q3": []})
    rows = build_sweep_rows(
        vector, {"local_khop": graph}, [i.question for i in items], k=10, weights=(0.3,)
    )
    report = evaluate_fusion_evidence(rows, items, 10, baseline=VECTOR_ROW, resamples=500)
    focus = report["rows"]["fused/local_khop@0.30/d10"]["slices"]["multi-hop"]
    delta = focus["paired_vs_baseline"][METRIC_RECALL]["delta"]
    assert delta["mean"] == pytest.approx(0.25) and delta["lo"] == 0.0
    assert report["verdict"]["decision"] == VERDICT_INCONCLUSIVE
    assert "includes no difference" in report["verdict"]["reason"]


# --- span identity (fusion-span-overlap-identity) ---------------------------------------------


def test_parse_span_identities_dedupes_and_rejects_an_unknown_policy():
    assert parse_span_identities("exact, overlap ,overlap") == ("exact", "overlap")
    with pytest.raises(ValueError, match="span identity must be one of"):
        parse_span_identities("contains")
    with pytest.raises(ValueError, match="no span identity"):
        parse_span_identities(" , ")


def test_an_identity_grid_adds_labeled_rows_and_leaves_the_exact_labels_unchanged():
    vector = _ByQuestion({"q": [_chunk("d1", 0, 800)]})
    graph = _ByQuestion({"q": [_chunk("d1", 120, 160)]})
    rows = build_sweep_rows(
        vector,
        {"local_khop": graph},
        ["q"],
        k=10,
        weights=(0.0, 0.3, 1.0),
        identities=("exact", "overlap"),
    )
    assert set(rows) == {
        VECTOR_ROW,
        "graph/local_khop",
        # endpoint weights fuse nothing, so they carry no identity variant
        "fused/local_khop@0.00/d10",
        "fused/local_khop@1.00/d10",
        # the default policy keeps the label it had before the policy existed
        "fused/local_khop@0.30/d10",
        "fused/local_khop@0.30/d10/ioverlap",
    }
    assert (
        vector.calls == 1 and graph.calls == 1
    )  # the policy re-maps cached lanes, never requeries


def test_the_overlap_row_folds_the_mention_into_its_chunk_and_the_exact_row_does_not():
    vector = _ByQuestion({"q": [_chunk("d1", 0, 800)]})
    graph = _ByQuestion({"q": [_chunk("d1", 120, 160)]})
    rows = build_sweep_rows(
        vector, {"local_khop": graph}, ["q"], k=10, weights=(0.3,), identities=("exact", "overlap")
    )
    exact = rows["fused/local_khop@0.30/d10"].retrieve("q", 10)
    overlap = rows["fused/local_khop@0.30/d10/ioverlap"].retrieve("q", 10)
    assert [(hit["char_start"], hit["char_end"]) for hit in exact] == [(0, 800), (120, 160)]
    assert [(hit["char_start"], hit["char_end"]) for hit in overlap] == [(0, 800)]


def test_the_report_states_the_cross_lane_agreement_rate_per_fused_row():
    items = [_multi_hop_item("mh-1", "q")]
    vector = _ByQuestion({"q": [_chunk("d1", 0, 800), _chunk("d2", 0, 800)]})
    graph = _ByQuestion({"q": [_chunk("d1", 120, 160)]})
    rows = build_sweep_rows(
        vector, {"local_khop": graph}, ["q"], k=10, weights=(0.3,), identities=("exact", "overlap")
    )
    report = evaluate_fusion_evidence(rows, items, 10, baseline=VECTOR_ROW, resamples=50)
    exact = report["rows"]["fused/local_khop@0.30/d10"]["agreement"]
    overlap = report["rows"]["fused/local_khop@0.30/d10/ioverlap"]["agreement"]
    assert (exact["questions_with_shared_candidate"], exact["mean_shared_candidates"]) == (0, 0.0)
    assert overlap["questions_with_shared_candidate"] == 1
    assert overlap["share_of_questions"] == 1.0
    # only fused rows can measure agreement; a single-lane row reports none
    assert "agreement" not in report["rows"][VECTOR_ROW]
    text = format_report(report)
    assert "Cross-lane agreement" in text
    assert text.isascii()


def test_a_sweep_row_label_round_trips_through_the_answer_quality_lane_parser():
    from llb.eval.answer_quality.lanes import parse_lane_label

    rows = build_sweep_rows(
        _ByQuestion({"q": []}),
        {"global_community": _ByQuestion({"q": []})},
        ["q"],
        k=10,
        weights=(0.1,),
        candidates=(50,),
        identities=("overlap",),
    )
    label = next(name for name in rows if name.startswith("fused/"))
    lane = parse_lane_label(label)
    assert (lane.retrieval_strategy, lane.graph_weight) == ("global_community", 0.1)
    assert (lane.graph_fusion_candidates, lane.graph_fusion_span_identity) == (50, "overlap")
