"""Focused tests split from ``test_fusion_evidence.py``."""

import pytest
from _fusion_evidence_helpers import (
    _ByQuestion,
    _chunk,
    _multi_hop_item,
)

from llb.rag.fusion_evidence import (
    build_sweep_rows,
    evaluate_fusion_evidence,
    format_report,
    parse_span_identities,
)
from llb.rag.fusion_evidence.rows import VECTOR_ROW


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
