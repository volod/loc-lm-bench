"""Graph-vector retrieval fusion over the shared offset-bearing chunk seam."""

from copy import deepcopy
from typing import Protocol

from llb.core.contracts.rag import ChunkRecord
from llb.rag.fusion_spans import (
    DEFAULT_SPAN_IDENTITY,
    SPAN_MERGE_MIN_RATIO,
    LaneCandidates,
    SpanKey,
    lane_candidates,
    merges_spans,
    resolve_merge_ratio,
    resolve_span_identity,
)
from llb.rag.lexical import weighted_rrf_fuse


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[ChunkRecord]: ...


class GraphWeightRouter(Protocol):
    def graph_weight(self, question: str) -> float: ...


# Graph evidence spans and vector chunks rarely have identical boundaries, so the standard RRF
# damping constant would make a 0.3 graph lane unable to enter a top-10 vector ranking at all.
# Undamped reciprocal ranks make graph_weight behave as an effective candidate share.
GRAPH_VECTOR_RRF_K = 0


def lane_depth(candidates: int | None, k: int) -> int:
    """How many candidates each lane is asked for before the fused result is cut to `k`.

    `None` (the default) asks for exactly `k`, which makes every graph candidate that enters the
    fused result displace a vector candidate one-for-one. A deeper pool lets `graph_weight` move
    the RANKING without spending a result seat per graph candidate; a shallower request is lifted
    to `k`, since a lane can never contribute fewer candidates than the result needs.
    """
    return k if candidates is None else max(candidates, k)


class FusedRetriever:
    """Fuse vector and graph candidate rankings over one shared span-identity policy."""

    def __init__(
        self,
        vector: Retriever,
        graph: Retriever,
        graph_weight: float,
        candidates: int | None = None,
        span_identity: str = DEFAULT_SPAN_IDENTITY,
        router: GraphWeightRouter | None = None,
        span_merge_ratio: float = SPAN_MERGE_MIN_RATIO,
    ) -> None:
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError(f"graph weight must be within [0, 1], got {graph_weight}")
        if candidates is not None and candidates < 1:
            raise ValueError(f"fusion candidate depth must be at least 1, got {candidates}")
        self.vector = vector
        self.graph = graph
        self.graph_weight = graph_weight
        self.candidates = candidates
        self.span_identity = resolve_span_identity(span_identity)
        self.span_merge_ratio = resolve_merge_ratio(span_merge_ratio)
        self.router = router

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
        graph_weight = (
            self.router.graph_weight(lexical_query)
            if self.router is not None
            else self.graph_weight
        )
        # Endpoint weights stay exact single-lane passthroughs at exactly `k`: a deeper pool
        # cannot change a ranking that is never fused.
        if graph_weight == 1.0:
            return self.graph.retrieve(lexical_query, k)
        if graph_weight == 0.0:
            return self._vector_hits(dense_query, lexical_query, k)
        depth = lane_depth(self.candidates, k)
        vector_hits = self._vector_hits(dense_query, lexical_query, depth)
        graph_hits = self.graph.retrieve(lexical_query, depth)
        return fuse_lane_hits(
            vector_hits,
            graph_hits,
            graph_weight,
            k,
            span_identity=self.span_identity,
            merge_ratio=self.span_merge_ratio,
        )

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
    *,
    span_identity: str = DEFAULT_SPAN_IDENTITY,
    merge_ratio: float = SPAN_MERGE_MIN_RATIO,
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
    candidates = lane_candidates(vector_hits, graph_hits, span_identity, merge_ratio)
    fused = weighted_rrf_fuse(
        candidates.rankings,
        [1.0 - graph_weight, graph_weight],
        k_const=GRAPH_VECTOR_RRF_K,
    )
    return [
        _fused_chunk(candidates, key, score, rank, graph_weight, span_identity, merge_ratio)
        for rank, (key, score) in enumerate(fused[:k], 1)
    ]


def _fused_chunk(
    candidates: LaneCandidates,
    key: SpanKey,
    score: float,
    rank: int,
    graph_weight: float,
    span_identity: str,
    merge_ratio: float,
) -> ChunkRecord:
    """One fused result row: the surviving record verbatim, plus the fusion provenance."""
    chunk = deepcopy(candidates.records[key])
    metadata = dict(chunk.get("metadata") or {})
    metadata["fusion_lanes"] = list(candidates.lanes[key])
    metadata["graph_weight"] = graph_weight
    metadata["fusion_span_identity"] = span_identity
    if merges_spans(span_identity):
        # Only a folding policy has a threshold to record; under `exact` it governed nothing.
        metadata["fusion_span_merge_ratio"] = resolve_merge_ratio(merge_ratio)
    merged = candidates.merged.get(key)
    if merged:
        metadata["fusion_merged_spans"] = [dict(span) for span in merged]
    chunk["metadata"] = metadata
    chunk["retrieval_score"] = float(score)
    chunk["rank"] = rank
    return chunk


def lane_agreement(
    vector_hits: list[ChunkRecord],
    graph_hits: list[ChunkRecord],
    span_identity: str = DEFAULT_SPAN_IDENTITY,
    merge_ratio: float = SPAN_MERGE_MIN_RATIO,
) -> int:
    """How many candidates BOTH lanes vouch for under `span_identity` -- the RRF agreement rate.

    Depth is only a live knob when this is non-zero: under undamped RRF a candidate only one lane
    returned, below rank k, can never enter the fused top-k.
    """
    return len(lane_candidates(vector_hits, graph_hits, span_identity, merge_ratio).shared())
