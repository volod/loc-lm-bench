"""Graph-vector retrieval fusion over the shared offset-bearing chunk seam."""

from copy import deepcopy
from typing import Protocol

from llb.core.contracts.rag import ChunkRecord
from llb.rag.lexical import weighted_rrf_fuse


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[ChunkRecord]: ...


SpanKey = tuple[str, int, int]
# Graph evidence spans and vector chunks rarely have identical boundaries, so the standard RRF
# damping constant would make a 0.3 graph lane unable to enter a top-10 vector ranking at all.
# Undamped reciprocal ranks make graph_weight behave as an effective candidate share.
GRAPH_VECTOR_RRF_K = 0


def span_key(chunk: ChunkRecord) -> SpanKey:
    """Stable identity shared by vector chunks and graph evidence records."""
    return (chunk["doc_id"], chunk["char_start"], chunk["char_end"])


def lane_depth(candidates: int | None, k: int) -> int:
    """How many candidates each lane is asked for before the fused result is cut to `k`.

    `None` (the default) asks for exactly `k`, which makes every graph candidate that enters the
    fused result displace a vector candidate one-for-one. A deeper pool lets `graph_weight` move
    the RANKING without spending a result seat per graph candidate; a shallower request is lifted
    to `k`, since a lane can never contribute fewer candidates than the result needs.
    """
    return k if candidates is None else max(candidates, k)


class FusedRetriever:
    """Fuse vector and graph candidate rankings, deduplicating their exact source spans."""

    def __init__(
        self,
        vector: Retriever,
        graph: Retriever,
        graph_weight: float,
        candidates: int | None = None,
    ) -> None:
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError(f"graph weight must be within [0, 1], got {graph_weight}")
        if candidates is not None and candidates < 1:
            raise ValueError(f"fusion candidate depth must be at least 1, got {candidates}")
        self.vector = vector
        self.graph = graph
        self.graph_weight = graph_weight
        self.candidates = candidates

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        """Return top-k fused chunks; endpoint weights are exact lane passthroughs."""
        return self._retrieve_lanes(question, question, k)

    def retrieve_queries(
        self, dense_query: str, lexical_query: str, k: int, **_kwargs: object
    ) -> list[ChunkRecord]:
        """Route HyDE text to vector dense search and the user query to graph/BM25."""
        return self._retrieve_lanes(dense_query, lexical_query, k)

    def _retrieve_lanes(self, dense_query: str, lexical_query: str, k: int) -> list[ChunkRecord]:
        if k < 1:
            return []
        # Endpoint weights stay exact single-lane passthroughs at exactly `k`: a deeper pool
        # cannot change a ranking that is never fused.
        if self.graph_weight == 1.0:
            return self.graph.retrieve(lexical_query, k)
        if self.graph_weight == 0.0:
            return self._vector_hits(dense_query, lexical_query, k)
        depth = lane_depth(self.candidates, k)
        vector_hits = self._vector_hits(dense_query, lexical_query, depth)
        graph_hits = self.graph.retrieve(lexical_query, depth)
        return fuse_lane_hits(vector_hits, graph_hits, self.graph_weight, k)

    def _vector_hits(self, dense_query: str, lexical_query: str, depth: int) -> list[ChunkRecord]:
        """Vector-lane candidates, routing HyDE text to dense search when the lane supports it."""
        vector_method = getattr(self.vector, "retrieve_queries", None)
        if callable(vector_method):
            hits: list[ChunkRecord] = vector_method(dense_query, lexical_query, depth)
            return hits
        return self.vector.retrieve(dense_query, depth)


def fuse_lane_hits(
    vector_hits: list[ChunkRecord],
    graph_hits: list[ChunkRecord],
    graph_weight: float,
    k: int,
) -> list[ChunkRecord]:
    """Fuse ONE question's already-retrieved lane rankings at `graph_weight`.

    Split out of `FusedRetriever` so a graph-weight sweep can query each lane once per question
    and re-fuse the SAME candidates at every weight: the lane rankings do not depend on the
    weight, only the fusion does, so this is identical to re-querying per weight but costs one
    retrieval pass. Endpoint weights stay exact lane passthroughs.
    """
    if k < 1:
        return []
    if graph_weight == 1.0:
        return graph_hits[:k]
    if graph_weight == 0.0:
        return vector_hits[:k]
    records, lanes = _span_candidates(vector_hits, graph_hits)
    fused = weighted_rrf_fuse(
        lanes,
        [1.0 - graph_weight, graph_weight],
        k_const=GRAPH_VECTOR_RRF_K,
    )
    out: list[ChunkRecord] = []
    for rank, (key, score) in enumerate(fused[:k], 1):
        chunk = deepcopy(records[key])
        metadata = dict(chunk.get("metadata") or {})
        metadata["fusion_lanes"] = _matching_lanes(key, vector_hits, graph_hits)
        metadata["graph_weight"] = graph_weight
        chunk["metadata"] = metadata
        chunk["retrieval_score"] = float(score)
        chunk["rank"] = rank
        out.append(chunk)
    return out


def _span_candidates(
    vector_hits: list[ChunkRecord], graph_hits: list[ChunkRecord]
) -> tuple[dict[SpanKey, ChunkRecord], list[list[SpanKey]]]:
    """Map both rankings to span ids; prefer the vector record for shared spans."""
    records: dict[SpanKey, ChunkRecord] = {}
    vector_ranking: list[SpanKey] = []
    graph_ranking: list[SpanKey] = []
    for hit in vector_hits:
        key = span_key(hit)
        records.setdefault(key, hit)
        vector_ranking.append(key)
    for hit in graph_hits:
        key = span_key(hit)
        records.setdefault(key, hit)
        graph_ranking.append(key)
    return records, [vector_ranking, graph_ranking]


def _matching_lanes(
    key: SpanKey, vector_hits: list[ChunkRecord], graph_hits: list[ChunkRecord]
) -> list[str]:
    lanes = []
    if any(span_key(hit) == key for hit in vector_hits):
        lanes.append("vector")
    if any(span_key(hit) == key for hit in graph_hits):
        lanes.append("graph")
    return lanes
