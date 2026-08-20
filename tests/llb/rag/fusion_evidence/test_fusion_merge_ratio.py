"""Focused tests split from ``test_fusion_evidence.py``."""

import pytest
from tests.llb.rag._fusion_evidence_helpers import (
    _ByQuestion,
    _chunk,
    _multi_hop_item,
)

from llb.rag.fusion_evidence import (
    build_sweep_rows,
    evaluate_fusion_evidence,
    format_report,
    parse_merge_ratios,
)
from llb.rag.fusion_evidence.rows import VECTOR_ROW


def test_parse_merge_ratios_dedupes_and_rejects_a_value_outside_the_unit_interval():
    assert parse_merge_ratios("0.25, 0.5 ,0.5, 1.0") == (0.25, 0.5, 1.0)
    with pytest.raises(ValueError, match=r"span merge ratio must be within \(0, 1\]"):
        parse_merge_ratios("0")
    with pytest.raises(ValueError, match="span merge ratio must be a number"):
        parse_merge_ratios("half")
    with pytest.raises(ValueError, match="no span merge ratio"):
        parse_merge_ratios(" , ")


def test_a_ratio_grid_labels_only_the_folding_policy_and_keeps_the_default_unmarked():
    vector = _ByQuestion({"q": [_chunk("d1", 0, 800)]})
    graph = _ByQuestion({"q": [_chunk("d1", 120, 160)]})
    rows = build_sweep_rows(
        vector,
        {"local_khop": graph},
        ["q"],
        k=10,
        weights=(0.0, 0.3, 1.0),
        identities=("exact", "overlap"),
        merge_ratios=(0.25, 0.5, 1.0),
    )
    assert set(rows) == {
        VECTOR_ROW,
        "graph/local_khop",
        "fused/local_khop@0.00/d10",
        "fused/local_khop@1.00/d10",
        # `exact` has no partial overlap to threshold, so the ratio grid adds no row there
        "fused/local_khop@0.30/d10",
        # the pinned ratio keeps the label the identity sweep already measured
        "fused/local_khop@0.30/d10/ioverlap",
        "fused/local_khop@0.30/d10/ioverlap/r0.25",
        "fused/local_khop@0.30/d10/ioverlap/r1.00",
    }
    assert vector.calls == 1 and graph.calls == 1  # the ratio re-maps cached lanes


def test_a_stricter_ratio_row_keeps_the_clipped_mention_a_separate_candidate():
    # 60 of the mention's 80 characters sit inside the chunk -- ratio 0.75
    vector = _ByQuestion({"q": [_chunk("d1", 0, 100)]})
    graph = _ByQuestion({"q": [_chunk("d1", 40, 120)]})
    rows = build_sweep_rows(
        vector,
        {"local_khop": graph},
        ["q"],
        k=10,
        weights=(0.3,),
        identities=("overlap",),
        merge_ratios=(0.5, 1.0),
    )
    merged = rows["fused/local_khop@0.30/d10/ioverlap"].retrieve("q", 10)
    containment = rows["fused/local_khop@0.30/d10/ioverlap/r1.00"].retrieve("q", 10)
    assert [(hit["char_start"], hit["char_end"]) for hit in merged] == [(0, 100)]
    assert [(hit["char_start"], hit["char_end"]) for hit in containment] == [(0, 100), (40, 120)]


def test_the_report_states_the_agreement_rate_per_merge_ratio():
    items = [_multi_hop_item("mh-1", "q")]
    vector = _ByQuestion({"q": [_chunk("d1", 0, 100), _chunk("d2", 0, 800)]})
    graph = _ByQuestion({"q": [_chunk("d1", 40, 120)]})
    rows = build_sweep_rows(
        vector,
        {"local_khop": graph},
        ["q"],
        k=10,
        weights=(0.3,),
        identities=("overlap",),
        merge_ratios=(0.5, 1.0),
    )
    report = evaluate_fusion_evidence(rows, items, 10, baseline=VECTOR_ROW, resamples=50)
    merged = report["rows"]["fused/local_khop@0.30/d10/ioverlap"]["agreement"]
    containment = report["rows"]["fused/local_khop@0.30/d10/ioverlap/r1.00"]["agreement"]
    assert merged["questions_with_shared_candidate"] == 1
    assert containment["questions_with_shared_candidate"] == 0
    assert format_report(report).isascii()


def test_a_ratio_row_label_round_trips_through_the_answer_quality_lane_parser():
    from llb.eval.answer_quality.lanes import parse_lane_label

    rows = build_sweep_rows(
        _ByQuestion({"q": []}),
        {"global_community": _ByQuestion({"q": []})},
        ["q"],
        k=10,
        weights=(0.1,),
        candidates=(50,),
        identities=("overlap",),
        merge_ratios=(0.25,),
    )
    label = next(name for name in rows if name.startswith("fused/"))
    lane = parse_lane_label(label)
    assert lane.graph_fusion_span_identity == "overlap"
    assert lane.graph_fusion_span_merge_ratio == 0.25


def test_a_routed_row_carries_the_ratio_grid_too():
    vector = _ByQuestion({"q": [_chunk("d1", 0, 800)]})
    graph = _ByQuestion({"q": [_chunk("d1", 120, 160)]})
    rows = build_sweep_rows(
        vector,
        {"local_khop": graph},
        ["q"],
        k=10,
        weights=(0.3,),
        identities=("overlap",),
        routed_graph_weight=0.3,
        merge_ratios=(0.5, 0.25),
    )
    assert "routed/local_khop@0.30/d10/ioverlap" in rows
    assert "routed/local_khop@0.30/d10/ioverlap/r0.25" in rows


def test_the_verdict_prefers_the_pinned_threshold_when_two_ratios_tie():
    # every fused row here returns the identical ranking, so only the tie-break can choose
    items = [_multi_hop_item("mh-1", "q")]
    vector = _ByQuestion({"q": [_chunk("d1", 0, 800), _chunk("d2", 0, 800)]})
    graph = _ByQuestion({"q": [_chunk("d1", 120, 160)]})
    rows = build_sweep_rows(
        vector,
        {"local_khop": graph},
        ["q"],
        k=10,
        weights=(0.3,),
        identities=("overlap",),
        merge_ratios=(0.25, 0.5, 0.75),
    )
    report = evaluate_fusion_evidence(rows, items, 10, baseline=VECTOR_ROW, resamples=50)
    assert report["verdict"]["best_row"] == "fused/local_khop@0.30/d10/ioverlap"
