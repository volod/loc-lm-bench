"""Focused helpers for dense/lexical hybrid retrieval."""

from collections.abc import Callable
from typing import Any, cast

from llb.core.contracts.rag import ChunkRecord
from llb.rag.filters import ChunkFilter
from llb.rag.vector_index import VectorIndex


def allowed_chunk_ids(
    chunks: list[ChunkRecord], chunk_filter: ChunkFilter | None
) -> set[int] | None:
    if chunk_filter is None:
        return None
    return {index for index, chunk in enumerate(chunks) if chunk_filter(chunk)}


def dense_hybrid_ids(
    chunks: list[ChunkRecord],
    index: VectorIndex,
    ranked_candidates: Callable[[Any, Any], list[tuple[int, float]]],
    query_vec: Any,
    depth: int,
    chunk_filter: ChunkFilter | None,
    allowed: set[int] | None,
    fusion_weight: float,
) -> list[int]:
    """Dense candidates for hybrid fusion, skipping a disabled dense lane."""
    if fusion_weight <= 0.0:
        return []
    search_k = len(chunks) if chunk_filter else min(len(chunks), depth)
    scores, ids = index.search(query_vec, max(1, search_k))
    dense_ids = [chunk_id for chunk_id, _ in ranked_candidates(ids, scores)]
    if allowed is not None:
        dense_ids = [chunk_id for chunk_id in dense_ids if chunk_id in allowed]
    return dense_ids[:depth]


def hybrid_chunks(
    chunks: list[ChunkRecord], fused: list[tuple[int, float]], k: int
) -> list[ChunkRecord]:
    """Materialize fused ids as ranked chunk records."""
    hits: list[ChunkRecord] = []
    for rank, (chunk_id, score) in enumerate(fused[:k], 1):
        chunk = cast(ChunkRecord, dict(chunks[chunk_id]))
        chunk["retrieval_score"] = float(score)
        chunk["rank"] = rank
        hits.append(chunk)
    return hits
