"""Execute prepared dense/lexical/subquery retrieval plans over the shared store seam."""

from typing import Any, cast

from llb.core.contracts.rag import ChunkRecord
from llb.rag.lexical import weighted_rrf_fuse
from llb.rag.query_prep.base import QueryPrepResult


SpanKey = tuple[str, int, int]
ORIGINAL_QUERY_RRF_WEIGHT = 2.0


def retrieve_prepared(
    store: Any,
    result: QueryPrepResult,
    k: int,
    *,
    chunk_filter: Any | None = None,
) -> list[ChunkRecord]:
    """Retrieve the prepared plan, fusing HyDE and per-subquery rankings when needed."""
    rankings: list[list[ChunkRecord]] = []
    weights: list[float] = []
    if result.subqueries:
        rankings.append(
            _retrieve_queries(store, result.processed, result.processed, k, chunk_filter)
        )
        weights.append(ORIGINAL_QUERY_RRF_WEIGHT)
        rankings.extend(
            _retrieve_queries(store, query, query, k, chunk_filter) for query in result.subqueries
        )
        weights.extend([1.0] * len(result.subqueries))
    if result.hypothetical_answer is not None:
        rankings.insert(
            0,
            _retrieve_queries(
                store,
                result.hypothetical_answer,
                result.processed,
                k,
                chunk_filter,
            ),
        )
        weights.insert(0, 1.0)
    if rankings:
        if len(rankings) == 1:
            return rankings[0][:k]
        return fuse_ranked_chunks(rankings, k, weights=weights)
    return _retrieve_queries(store, result.processed, result.processed, k, chunk_filter)


def _retrieve_queries(
    store: Any,
    dense_query: str,
    lexical_query: str,
    k: int,
    chunk_filter: Any | None,
) -> list[ChunkRecord]:
    method = getattr(store, "retrieve_queries", None)
    if callable(method):
        return cast(
            list[ChunkRecord],
            method(
                dense_query,
                lexical_query,
                k,
                chunk_filter=chunk_filter,
            ),
        )
    kwargs = {"chunk_filter": chunk_filter} if chunk_filter is not None else {}
    return cast(list[ChunkRecord], store.retrieve(dense_query, k, **kwargs))


def fuse_ranked_chunks(
    rankings: list[list[ChunkRecord]], k: int, *, weights: list[float] | None = None
) -> list[ChunkRecord]:
    """Weighted RRF over query hit lists, deduplicated by exact source span."""
    if not rankings or k < 1:
        return []
    records: dict[SpanKey, ChunkRecord] = {}
    span_rankings: list[list[SpanKey]] = []
    for ranking in rankings:
        spans: list[SpanKey] = []
        for chunk in ranking:
            key = (chunk["doc_id"], chunk["char_start"], chunk["char_end"])
            records.setdefault(key, chunk)
            spans.append(key)
        span_rankings.append(spans)
    fused = weighted_rrf_fuse(span_rankings, weights or [1.0] * len(span_rankings))
    out: list[ChunkRecord] = []
    for rank, (key, score) in enumerate(fused[:k], 1):
        chunk = cast(ChunkRecord, dict(records[key]))
        chunk["retrieval_score"] = float(score)
        chunk["rank"] = rank
        out.append(chunk)
    return out
