"""Focused tests split from ``test_graph_vector_fusion.py``."""

import pytest
from _graph_vector_fusion_helpers import (
    FakeRetriever,
    _chunk,
    _mention,
)

from llb.core.config import RunConfig
from llb.rag.fusion import (
    FusedRetriever,
    fuse_lane_hits,
    lane_agreement,
)
from llb.rag.fusion_spans import (
    SPAN_IDENTITY_EXACT,
    SPAN_IDENTITY_OVERLAP,
    SPAN_MERGE_MIN_RATIO,
    lane_candidates,
)


def test_the_span_identity_policy_lands_in_the_fingerprint():
    assert RunConfig().fingerprint()["graph_fusion_span_identity"] == SPAN_IDENTITY_EXACT
    config = RunConfig(retrieval_backend="fused", graph_fusion_span_identity="overlap")
    assert config.fingerprint()["graph_fusion_span_identity"] == SPAN_IDENTITY_OVERLAP


def test_a_clipped_mention_merges_only_above_the_configured_ratio():
    chunk = _chunk("chunk", 0, 100, lane="vector")
    # 60 of the graph span's 80 characters sit inside the chunk -- ratio 0.75
    clipped = [_mention("clipped", 40, 120)]
    for ratio, expected in ((0.25, 1), (0.5, 1), (0.75, 1), (1.0, 2)):
        candidates = lane_candidates([chunk], clipped, SPAN_IDENTITY_OVERLAP, ratio)
        assert len(candidates.records) == expected, ratio


def test_a_grazing_mention_merges_only_under_a_permissive_ratio():
    chunk = _chunk("chunk", 0, 100, lane="vector")
    # 10 of the graph span's 80 characters sit inside the chunk -- ratio 0.125
    grazing = [_mention("grazing", 90, 170)]
    for ratio, expected in ((0.1, 1), (0.25, 2), (0.5, 2), (1.0, 2)):
        candidates = lane_candidates([chunk], grazing, SPAN_IDENTITY_OVERLAP, ratio)
        assert len(candidates.records) == expected, ratio


def test_containment_only_still_folds_a_fully_contained_mention():
    vector = [_chunk("chunk", 0, 800, lane="vector")]
    graph = [_mention("mention", 120, 160)]
    candidates = lane_candidates(vector, graph, SPAN_IDENTITY_OVERLAP, 1.0)
    assert candidates.shared() == [("doc.md", 0, 800)]


def test_the_merge_ratio_never_merges_two_vector_chunks_however_permissive():
    # the invariant is structural, not a consequence of the threshold: only the vector lane
    # creates anchors, so even a ratio that would admit any touch cannot chain a document
    vector = [_chunk("first", 0, 800, lane="vector"), _chunk("second", 680, 1480, lane="vector")]
    candidates = lane_candidates(vector, [], SPAN_IDENTITY_OVERLAP, 0.01)
    assert list(candidates.records) == [("doc.md", 0, 800), ("doc.md", 680, 1480)]


def test_the_merge_ratio_is_dead_under_the_exact_identity():
    vector = [_chunk("chunk", 0, 800, lane="vector")]
    graph = [_mention("mention", 120, 160)]
    baseline = fuse_lane_hits(vector, graph, 0.3, 5, span_identity=SPAN_IDENTITY_EXACT)
    for ratio in (0.25, 0.5, 1.0):
        assert (
            fuse_lane_hits(
                vector, graph, 0.3, 5, span_identity=SPAN_IDENTITY_EXACT, merge_ratio=ratio
            )
            == baseline
        ), ratio


def test_the_default_ratio_reproduces_the_unswept_ranking_and_is_recorded():
    vector = [_chunk("c1", 0, 800, lane="vector"), _chunk("c2", 800, 1600, lane="vector")]
    graph = [_mention("mention", 120, 160)]
    fused = fuse_lane_hits(vector, graph, 0.3, 3, span_identity=SPAN_IDENTITY_OVERLAP)
    assert fused == fuse_lane_hits(
        vector, graph, 0.3, 3, span_identity=SPAN_IDENTITY_OVERLAP, merge_ratio=0.5
    )
    assert fused[0]["metadata"]["fusion_span_merge_ratio"] == 0.5
    # `exact` has no partial overlap to threshold, so it records no threshold either
    exact = fuse_lane_hits(vector, graph, 0.3, 3, span_identity=SPAN_IDENTITY_EXACT)
    assert "fusion_span_merge_ratio" not in exact[0]["metadata"]


def test_a_stricter_ratio_lowers_cross_lane_agreement():
    vector = [_chunk("chunk", 0, 100, lane="vector")]
    graph = [_mention("clipped", 40, 120)]
    assert lane_agreement(vector, graph, SPAN_IDENTITY_OVERLAP, 0.5) == 1
    assert lane_agreement(vector, graph, SPAN_IDENTITY_OVERLAP, 1.0) == 0


def test_a_merge_ratio_outside_the_unit_interval_is_rejected():
    for ratio in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match=r"span merge ratio must be within \(0, 1\]"):
            FusedRetriever(FakeRetriever([]), FakeRetriever([]), 0.3, span_merge_ratio=ratio)


def test_the_merge_ratio_lands_in_the_fingerprint():
    assert RunConfig().fingerprint()["graph_fusion_span_merge_ratio"] == SPAN_MERGE_MIN_RATIO
    config = RunConfig(retrieval_backend="fused", graph_fusion_span_merge_ratio=0.25)
    assert config.fingerprint()["graph_fusion_span_merge_ratio"] == 0.25
