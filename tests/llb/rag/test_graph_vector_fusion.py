"""Graph-vector fusion ordering, endpoints, depth, span identity, and runner wiring tests."""

import pytest

from llb.core.config import RunConfig
from llb.core.contracts.rag import ChunkRecord
from llb.rag.fusion import FusedRetriever, fuse_lane_hits, lane_agreement
from llb.rag.fusion_spans import (
    SPAN_IDENTITY_EXACT,
    SPAN_IDENTITY_OVERLAP,
    lane_candidates,
    overlap_ratio,
)
from llb.rag.lexical import weighted_rrf_fuse


class FakeRetriever:
    def __init__(self, hits: list[ChunkRecord]) -> None:
        self.hits = hits
        self.calls = 0
        self.depths: list[int] = []

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        self.calls += 1
        self.depths.append(k)
        return self.hits[:k]


def _chunk(name: str, start: int, end: int, *, lane: str) -> ChunkRecord:
    return {
        "doc_id": "doc.md",
        "char_start": start,
        "char_end": end,
        "text": name,
        "chunk_id": f"{lane}-{name}",
        "metadata": {"lane": lane},
    }


def test_weighted_rrf_generalizes_to_n_lists_and_ignores_zero_weight_membership():
    fused = weighted_rrf_fuse([["a", "b"], ["c"], ["d"]], [0.5, 0.5, 0.0], k_const=0)
    assert [item for item, _score in fused] == ["a", "c", "b"]
    assert "d" not in {item for item, _score in fused}


def test_graph_weight_zero_is_exact_vector_passthrough_without_graph_query():
    vector_hits = [_chunk("a", 0, 10, lane="vector"), _chunk("b", 10, 20, lane="vector")]
    vector = FakeRetriever(vector_hits)
    graph = FakeRetriever([_chunk("g", 30, 40, lane="graph")])
    hits = FusedRetriever(vector, graph, graph_weight=0.0).retrieve("q", 2)
    assert hits == vector_hits
    assert graph.calls == 0


def test_graph_weight_one_is_exact_graph_passthrough_without_vector_query():
    vector = FakeRetriever([_chunk("a", 0, 10, lane="vector")])
    graph_hits = [_chunk("g", 30, 40, lane="graph")]
    graph = FakeRetriever(graph_hits)
    hits = FusedRetriever(vector, graph, graph_weight=1.0).retrieve("q", 1)
    assert hits == graph_hits
    assert vector.calls == 0


def test_fusion_surfaces_graph_hits_and_deduplicates_shared_exact_span():
    vector = FakeRetriever(
        [
            _chunk("a", 0, 10, lane="vector"),
            _chunk("shared-vector", 10, 20, lane="vector"),
            _chunk("c", 20, 30, lane="vector"),
        ]
    )
    graph = FakeRetriever(
        [
            _chunk("g", 30, 40, lane="graph"),
            _chunk("shared-graph", 10, 20, lane="graph"),
            _chunk("h", 40, 50, lane="graph"),
        ]
    )
    hits = FusedRetriever(vector, graph, graph_weight=0.3).retrieve("q", 3)
    assert [(hit["char_start"], hit["char_end"]) for hit in hits] == [
        (0, 10),
        (10, 20),
        (30, 40),
    ]
    shared = hits[1]
    assert shared["text"] == "shared-vector"
    assert shared["metadata"]["fusion_lanes"] == ["vector", "graph"]
    assert [hit["rank"] for hit in hits] == [1, 2, 3]


def test_fused_config_fields_land_in_the_fingerprint():
    config = RunConfig(retrieval_backend="fused", graph_weight=0.4, graph_fusion_candidates=50)
    assert config.fingerprint()["retrieval_backend"] == "fused"
    assert config.fingerprint()["graph_weight"] == 0.4
    assert config.fingerprint()["graph_fusion_candidates"] == 50
    # the default is the historical behavior: each lane is asked for exactly top_k
    assert RunConfig().fingerprint()["graph_fusion_candidates"] is None


def test_runner_loads_vector_and_graph_before_fusing(monkeypatch, tmp_path):
    from llb.executor import runner_retrieval

    vector = FakeRetriever([_chunk("v", 0, 10, lane="vector")])
    graph = FakeRetriever([_chunk("g", 10, 20, lane="graph")])
    monkeypatch.setattr(runner_retrieval, "_load_vector_store", lambda _config: vector)
    monkeypatch.setattr(runner_retrieval, "_load_graph_store", lambda _config: graph)
    loaded = runner_retrieval._load_store(
        RunConfig(
            data_dir=tmp_path,
            retrieval_backend="fused",
            graph_weight=0.3,
            graph_fusion_candidates=40,
        )
    )
    assert isinstance(loaded, FusedRetriever)
    assert loaded.vector is vector and loaded.graph is graph
    assert loaded.candidates == 40


# --- candidate depth ------------------------------------------------------------------------


def _numbered(lane: str, count: int) -> list[ChunkRecord]:
    return [_chunk(f"{lane}{i}", i * 10, i * 10 + 10, lane=lane) for i in range(count)]


def test_default_depth_asks_each_lane_for_exactly_k():
    vector, graph = FakeRetriever(_numbered("vector", 8)), FakeRetriever(_numbered("graph", 8))
    FusedRetriever(vector, graph, graph_weight=0.3).retrieve("q", 3)
    assert vector.depths == [3] and graph.depths == [3]


def test_a_deeper_pool_queries_both_lanes_deeper_and_still_returns_k():
    vector, graph = FakeRetriever(_numbered("vector", 8)), FakeRetriever(_numbered("graph", 8))
    hits = FusedRetriever(vector, graph, graph_weight=0.3, candidates=6).retrieve("q", 3)
    assert vector.depths == [6] and graph.depths == [6]
    assert len(hits) == 3
    assert [hit["rank"] for hit in hits] == [1, 2, 3]


def test_depth_below_k_is_lifted_to_k_never_starving_the_result():
    vector, graph = FakeRetriever(_numbered("vector", 8)), FakeRetriever(_numbered("graph", 8))
    hits = FusedRetriever(vector, graph, graph_weight=0.3, candidates=2).retrieve("q", 5)
    assert vector.depths == [5] and graph.depths == [5]
    assert len(hits) == 5


def test_depth_equal_to_k_reproduces_the_default_fused_ranking_exactly():
    def fused(candidates: int | None) -> list[ChunkRecord]:
        vector, graph = FakeRetriever(_numbered("vector", 8)), FakeRetriever(_numbered("graph", 8))
        # one span both lanes carry, so the ranking is not a trivial vector passthrough
        graph.hits[1] = _chunk("shared", 10, 20, lane="graph")
        return FusedRetriever(vector, graph, 0.3, candidates).retrieve("q", 4)

    assert fused(4) == fused(None)


def test_a_deeper_pool_does_not_change_an_endpoint_weight_passthrough():
    for weight in (0.0, 1.0):
        vector, graph = FakeRetriever(_numbered("vector", 8)), FakeRetriever(_numbered("graph", 8))
        deep = FusedRetriever(vector, graph, weight, candidates=50).retrieve("q", 3)
        plain = FusedRetriever(
            FakeRetriever(_numbered("vector", 8)), FakeRetriever(_numbered("graph", 8)), weight
        ).retrieve("q", 3)
        assert deep == plain, weight
        assert (vector.depths or graph.depths) == [3]  # the queried lane is never over-fetched


def test_a_non_positive_candidate_depth_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        FusedRetriever(FakeRetriever([]), FakeRetriever([]), 0.3, candidates=0)


# --- span identity (fusion-span-overlap-identity) ---------------------------------------------


def _mention(name: str, start: int, end: int) -> ChunkRecord:
    """A graph evidence span: a few dozen characters cut around an entity, not a chunk."""
    return _chunk(name, start, end, lane="graph")


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


def test_the_span_identity_policy_lands_in_the_fingerprint():
    assert RunConfig().fingerprint()["graph_fusion_span_identity"] == SPAN_IDENTITY_EXACT
    config = RunConfig(retrieval_backend="fused", graph_fusion_span_identity="overlap")
    assert config.fingerprint()["graph_fusion_span_identity"] == SPAN_IDENTITY_OVERLAP
