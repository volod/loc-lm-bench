"""Focused tests split from ``test_graph_vector_fusion.py``."""

import pytest
from tests.llb.rag._graph_vector_fusion_helpers import (
    FakeRetriever,
    _chunk,
    _mention,
)

from llb.rag.fusion.fuse import (
    FusedRetriever,
    fuse_lane_hits,
    lane_agreement,
)
from llb.rag.fusion.spans import (
    SPAN_IDENTITY_EXACT,
    SPAN_IDENTITY_OVERLAP,
    lane_candidates,
    overlap_ratio,
)


def test_exact_identity_leaves_a_contained_graph_span_a_separate_candidate():
    vector = [_chunk("chunk", 0, 800, lane="vector")]
    graph = [_mention("mention", 120, 160)]
    candidates = lane_candidates(vector, graph, SPAN_IDENTITY_EXACT)
    assert len(candidates.records) == 2
    assert candidates.shared() == []


def test_overlap_identity_folds_a_contained_graph_span_into_its_chunk():
    vector = [_chunk("chunk", 0, 800, lane="vector")]
    graph = [_mention("mention", 120, 160)]
    candidates = lane_candidates(vector, graph, SPAN_IDENTITY_OVERLAP)
    assert list(candidates.records) == [("doc.md", 0, 800)]
    assert candidates.shared() == [("doc.md", 0, 800)]
    assert candidates.rankings == [[("doc.md", 0, 800)], [("doc.md", 0, 800)]]
    assert candidates.merged[("doc.md", 0, 800)] == [
        {"lane": "graph", "doc_id": "doc.md", "char_start": 120, "char_end": 160}
    ]


def test_the_surviving_record_keeps_the_chunk_text_and_offsets_verbatim():
    vector = [_chunk("chunk-text", 0, 800, lane="vector")]
    graph = [_mention("mention-text", 120, 160)]
    fused = fuse_lane_hits(vector, graph, 0.3, 5, span_identity=SPAN_IDENTITY_OVERLAP)
    assert len(fused) == 1
    survivor = fused[0]
    # a merge never synthesizes a union span: text and offsets stay an exact corpus slice
    assert (survivor["text"], survivor["char_start"], survivor["char_end"]) == (
        "chunk-text",
        0,
        800,
    )
    assert survivor["metadata"]["fusion_lanes"] == ["vector", "graph"]
    assert survivor["metadata"]["fusion_span_identity"] == SPAN_IDENTITY_OVERLAP


def test_overlap_identity_merges_a_partially_overlapping_span_but_not_a_marginal_touch():
    chunk = _chunk("chunk", 0, 100, lane="vector")
    # 60 of the graph span's 80 characters sit inside the chunk -- a clipped mention, one candidate
    clipped = lane_candidates([chunk], [_mention("clipped", 40, 120)], SPAN_IDENTITY_OVERLAP)
    assert len(clipped.records) == 1
    # 10 of 80 characters -- two different pieces of evidence, kept separate
    grazing = lane_candidates([chunk], [_mention("grazing", 90, 170)], SPAN_IDENTITY_OVERLAP)
    assert len(grazing.records) == 2
    assert grazing.shared() == []


def test_a_disjoint_graph_span_stays_its_own_candidate_under_both_policies():
    vector = [_chunk("chunk", 0, 100, lane="vector")]
    graph = [_mention("elsewhere", 400, 440)]
    for identity in (SPAN_IDENTITY_EXACT, SPAN_IDENTITY_OVERLAP):
        candidates = lane_candidates(vector, graph, identity)
        assert len(candidates.records) == 2, identity
        assert candidates.shared() == [], identity


def test_a_span_in_another_document_never_merges():
    vector = [_chunk("chunk", 0, 800, lane="vector")]
    graph = [{**_mention("mention", 120, 160), "doc_id": "other.md"}]
    candidates = lane_candidates(vector, graph, SPAN_IDENTITY_OVERLAP)
    assert len(candidates.records) == 2
    assert overlap_ratio(("a.md", 0, 100), ("b.md", 0, 100)) == 0.0


def test_overlapping_vector_chunks_are_never_merged_with_each_other():
    # consecutive recursive chunks share their `chunk_overlap` tail; chaining them would collapse
    # a whole document into one candidate and destroy the vector ranking
    vector = [_chunk("first", 0, 800, lane="vector"), _chunk("second", 680, 1480, lane="vector")]
    candidates = lane_candidates(vector, [], SPAN_IDENTITY_OVERLAP)
    assert list(candidates.records) == [("doc.md", 0, 800), ("doc.md", 680, 1480)]


def test_a_mention_in_the_shared_tail_joins_the_better_ranked_chunk():
    vector = [_chunk("first", 0, 800, lane="vector"), _chunk("second", 680, 1480, lane="vector")]
    graph = [_mention("mention", 700, 740)]
    candidates = lane_candidates(vector, graph, SPAN_IDENTITY_OVERLAP)
    assert candidates.shared() == [("doc.md", 0, 800)]


def test_two_graph_spans_for_the_same_chunk_vote_once_and_are_both_recorded():
    vector = [_chunk("chunk", 0, 800, lane="vector")]
    graph = [_mention("first", 120, 160), _mention("second", 300, 340)]
    candidates = lane_candidates(vector, graph, SPAN_IDENTITY_OVERLAP)
    key = ("doc.md", 0, 800)
    assert candidates.rankings[1] == [key]  # one graph vote, not two
    assert len(candidates.merged[key]) == 2


def test_graph_only_spans_that_overlap_each_other_collapse_into_one_candidate():
    graph = [_mention("edge-evidence", 100, 200), _mention("mention", 120, 160)]
    candidates = lane_candidates([_chunk("far", 5000, 5800, lane="vector")], graph, "overlap")
    assert len(candidates.records) == 2  # the far chunk plus ONE graph candidate
    assert candidates.records[("doc.md", 100, 200)]["text"] == "edge-evidence"


def test_exact_identity_is_the_default_and_reproduces_the_unswitched_ranking():
    vector = [_chunk("a", 0, 800, lane="vector"), _chunk("b", 800, 1600, lane="vector")]
    graph = [_mention("g", 120, 160), _chunk("b", 800, 1600, lane="graph")]
    assert fuse_lane_hits(vector, graph, 0.3, 3) == fuse_lane_hits(
        vector, graph, 0.3, 3, span_identity=SPAN_IDENTITY_EXACT
    )
    assert FusedRetriever(FakeRetriever(vector), FakeRetriever(graph), 0.3).span_identity == (
        SPAN_IDENTITY_EXACT
    )


def test_overlap_identity_promotes_the_chunk_both_lanes_vouch_for():
    # the graph lane's mention sits inside the vector lane's THIRD chunk; under `exact` that
    # agreement is invisible and the graph mention competes for a seat of its own
    vector = [
        _chunk("c1", 0, 800, lane="vector"),
        _chunk("c2", 800, 1600, lane="vector"),
        _chunk("c3", 1600, 2400, lane="vector"),
    ]
    graph = [_mention("mention", 1700, 1740)]
    exact = fuse_lane_hits(vector, graph, 0.3, 3)
    overlap = fuse_lane_hits(vector, graph, 0.3, 3, span_identity=SPAN_IDENTITY_OVERLAP)
    # exact: the 40-character mention spends the third seat as a candidate of its own
    assert [(hit["char_start"], hit["char_end"]) for hit in exact] == [
        (0, 800),
        (800, 1600),
        (1700, 1740),
    ]
    # overlap: the same evidence instead lifts the chunk that contains it above its neighbour
    assert [(hit["char_start"], hit["char_end"]) for hit in overlap] == [
        (0, 800),
        (1600, 2400),
        (800, 1600),
    ]
    assert lane_agreement(vector, graph, SPAN_IDENTITY_EXACT) == 0
    assert lane_agreement(vector, graph, SPAN_IDENTITY_OVERLAP) == 1


def test_an_unknown_span_identity_is_rejected():
    with pytest.raises(ValueError, match="span identity must be one of"):
        FusedRetriever(FakeRetriever([]), FakeRetriever([]), 0.3, span_identity="contains")
