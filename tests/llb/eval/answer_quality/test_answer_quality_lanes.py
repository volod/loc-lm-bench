"""Focused tests split from ``test_answer_quality.py``."""

import json
from pathlib import Path

import pytest
from tests.llb.eval._answer_quality_helpers import (
    FUSED,
    VECTOR,
    _lanes,
    _row,
    _types,
)

from llb.core.config import RunConfig
from llb.eval.answer_quality import (
    compare_answer_quality,
    format_report,
    lane_config,
    lane_labels_from_comparison,
    parse_lane_label,
    parse_lanes,
)
from llb.eval.answer_quality.models import METRIC_OBJECTIVE
from llb.eval.paired_cases import shared_item_ids
from llb.rag.fusion_evidence.models import (
    fused_row_label,
    routed_row_label,
)


def test_lane_label_parses_every_sweep_row_shape():
    assert parse_lane_label("vector").retrieval_backend == "faiss"
    graph = parse_lane_label("graph/local_khop")
    assert (graph.retrieval_backend, graph.retrieval_strategy) == ("graph", "local_khop")
    fused = parse_lane_label("fused/global_community@0.10/d50")
    assert fused.retrieval_backend == "fused"
    assert fused.retrieval_strategy == "global_community"
    assert fused.graph_weight == pytest.approx(0.1)
    assert fused.graph_fusion_candidates == 50


def test_fused_label_round_trips_the_sweeps_own_template():
    """The parser must never drift from the one place the sweep FORMATS a fused row label."""
    for identity in ("exact", "overlap"):
        label = fused_row_label("global_community", 0.1, 10, identity)
        spec = parse_lane_label(label)
        assert spec.label == label
        assert (spec.retrieval_strategy, spec.graph_fusion_candidates) == ("global_community", 10)
        assert spec.graph_fusion_span_identity == identity
        assert spec.graph_weight == pytest.approx(0.1)


def test_routed_label_round_trips_and_enables_question_type_routing():
    label = routed_row_label("global_community", 0.3, 50, "overlap")
    spec = parse_lane_label(label)
    assert label == "routed/global_community@0.30/d50/ioverlap"
    assert spec.graph_fusion_router == "question_type"
    config = lane_config(RunConfig(), spec, run_name_prefix="aq")
    assert config.retrieval_backend == "fused"
    assert config.graph_fusion_router == "question_type"


def test_routed_report_states_focus_gain_and_exact_factoid_passthrough():
    routed = "routed/global_community@0.30/d50/ioverlap"
    rows = {
        VECTOR: [
            _row("multi", 0.2, 0.0),
            _row("fact", 0.8, 1.0),
        ],
        routed: [
            _row("multi", 0.2, 1.0),
            _row("fact", 0.8, 1.0),
        ],
    }
    report = compare_answer_quality(
        rows,
        {"multi": "multi-hop", "fact": "factoid"},
        baseline=VECTOR,
        resamples=20,
    )
    rendered = format_report(report)
    assert "### Routing outcome" in rendered
    assert "makes factoid answers an exact baseline passthrough" in rendered


def test_fused_label_without_depth_leaves_the_lane_pool_at_top_k():
    assert parse_lane_label("fused/local_khop@0.30").graph_fusion_candidates is None


@pytest.mark.parametrize(
    "label", ["", "faiss", "fused/global_community", "fused/@0.3", "fused/x@1.5", "fused/x@0.3/d0"]
)
def test_unparseable_lane_labels_are_rejected(label: str):
    with pytest.raises(ValueError):
        parse_lane_label(label)


def test_lane_selection_deduplicates_in_the_order_given():
    assert [spec.label for spec in parse_lanes("vector, fused/x@0.3 ,vector")] == [
        "vector",
        "fused/x@0.3",
    ]


def test_lanes_are_read_from_the_sweep_verdict_that_named_the_best_row(tmp_path: Path):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"verdict": {"baseline": VECTOR, "best_row": FUSED}}))
    assert lane_labels_from_comparison(comparison) == [VECTOR, FUSED]


def test_a_sweep_verdict_without_a_best_row_is_not_scorable(tmp_path: Path):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"verdict": {"baseline": VECTOR, "best_row": None}}))
    with pytest.raises(ValueError, match="no fused row"):
        lane_labels_from_comparison(comparison)


def test_lane_config_can_reset_the_candidate_depth_that_with_overrides_would_drop():
    base = RunConfig(retrieval_backend="fused", graph_fusion_candidates=50, graph_weight=0.3)
    vector = lane_config(base, parse_lane_label(VECTOR), run_name_prefix="answer-quality")
    assert vector.retrieval_backend == "faiss"
    assert vector.graph_fusion_candidates is None
    assert vector.run_name == "answer-quality-vector"
    fused = lane_config(base, parse_lane_label("fused/local_khop@0.00"), run_name_prefix="aq")
    assert fused.graph_weight == pytest.approx(0.0)
    assert fused.retrieval_strategy == "local_khop"


def test_lanes_scoring_different_item_sets_is_not_a_comparison():
    with pytest.raises(ValueError, match="different item sets"):
        shared_item_ids(_lanes([_row("a", 1.0)], [_row("b", 1.0)]))


def test_a_lane_that_scored_an_item_twice_is_rejected():
    with pytest.raises(ValueError, match="more than once"):
        shared_item_ids(_lanes([_row("a", 1.0), _row("a", 0.0)], [_row("a", 1.0)]))


def test_shared_item_ids_are_sorted_so_every_lane_aligns_item_by_item():
    lanes = _lanes([_row("b", 1.0), _row("a", 0.0)], [_row("a", 0.0), _row("b", 1.0)])
    assert shared_item_ids(lanes) == ["a", "b"]


def test_comparison_aligns_rows_by_item_id_not_by_file_order():
    report = compare_answer_quality(
        _lanes([_row("b", 1.0), _row("a", 0.0)], [_row("a", 0.0), _row("b", 1.0)]),
        _types("a", "b"),
        baseline=VECTOR,
        resamples=0,
    )
    focus = report["lanes"][FUSED]["slices"]["multi-hop"]
    assert focus["paired_vs_baseline"][METRIC_OBJECTIVE]["delta"]["mean"] == pytest.approx(0.0)
    assert focus["paired_vs_baseline"][METRIC_OBJECTIVE]["ties"] == 2


def test_an_unknown_baseline_lane_is_rejected():
    with pytest.raises(ValueError, match="baseline lane"):
        compare_answer_quality(_lanes([_row("a", 1.0)], [_row("a", 1.0)]), {}, baseline="missing")
