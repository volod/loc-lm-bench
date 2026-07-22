"""Graph-vector fusion ordering, endpoint, depth, deduplication, and runner wiring tests."""

import pytest

from llb.core.config import RunConfig
from llb.core.contracts.rag import ChunkRecord
from llb.rag.fusion import FusedRetriever
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
