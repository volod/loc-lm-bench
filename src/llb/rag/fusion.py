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


class FusedRetriever:
    """Fuse vector and graph candidate rankings, deduplicating their exact source spans."""

    def __init__(self, vector: Retriever, graph: Retriever, graph_weight: float) -> None:
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError(f"graph weight must be within [0, 1], got {graph_weight}")
        self.vector = vector
        self.graph = graph
        self.graph_weight = graph_weight

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        """Return top-k fused chunks; endpoint weights are exact lane passthroughs."""
        if k < 1:
            return []
        if self.graph_weight == 0.0:
            return self.vector.retrieve(question, k)
        if self.graph_weight == 1.0:
            return self.graph.retrieve(question, k)

        vector_hits = self.vector.retrieve(question, k)
        graph_hits = self.graph.retrieve(question, k)
        records, lanes = _span_candidates(vector_hits, graph_hits)
        fused = weighted_rrf_fuse(
            lanes,
            [1.0 - self.graph_weight, self.graph_weight],
            k_const=GRAPH_VECTOR_RRF_K,
        )
        out: list[ChunkRecord] = []
        for rank, (key, score) in enumerate(fused[:k], 1):
            chunk = deepcopy(records[key])
            metadata = dict(chunk.get("metadata") or {})
            metadata["fusion_lanes"] = _matching_lanes(key, vector_hits, graph_hits)
            metadata["graph_weight"] = self.graph_weight
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
