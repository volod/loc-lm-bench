"""Retrieval quality metrics by SOURCE-SPAN overlap (pure Python).

A retrieved chunk is a HIT for a gold item when it covers the same document and its
character range overlaps any of the item's labeled source spans. Anchoring on char
offsets (not chunk ids) means the metric survives chunk_size / strategy changes during
tuning -- it measures the embedding + retrieval config, not the chunk policy.

These metrics validate the pinned embedding (Premise 4: recall@10 >= 0.8). They are
CONSTANT across generation models under pinned retrieval, so they are reported as
context, never as a model-ranking axis.

Inputs are plain dicts so this module has zero heavy deps and is fully unit-testable:
  chunk = {"doc_id": str, "char_start": int, "char_end": int, ...}
  span  = {"doc_id": str, "char_start": int, "char_end": int, ...}
"""

from llb.core.contracts.rag import ChunkRecord, RetrievalMetrics, RetrievalPair, SourceSpanRecord
from llb.rag.duplicates import occurrence_spans


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """True if [a_start, a_end) and [b_start, b_end) share at least one character."""
    return a_start < b_end and b_start < a_end


def _place_hits_span(place: SourceSpanRecord, span: SourceSpanRecord) -> bool:
    return place["doc_id"] == span["doc_id"] and spans_overlap(
        place["char_start"], place["char_end"], span["char_start"], span["char_end"]
    )


def chunk_hits_span(chunk: ChunkRecord, span: SourceSpanRecord) -> bool:
    """True if a retrieved chunk overlaps a labeled span in the same document.

    A chunk that collapsed byte-identical copies (`llb.rag.duplicates`) is matched at EVERY place
    its text appears, so indexing a repeated passage once neither loses nor invents a hit.
    """
    return any(_place_hits_span(place, span) for place in occurrence_spans(chunk))


def chunk_hits_any(chunk: ChunkRecord, spans: list[SourceSpanRecord]) -> bool:
    return any(chunk_hits_span(chunk, span) for span in spans)


def first_hit_rank(retrieved: list[ChunkRecord], spans: list[SourceSpanRecord]) -> int | None:
    """1-based rank of the first retrieved chunk that hits a labeled span, else None."""
    for rank, chunk in enumerate(retrieved, 1):
        if chunk_hits_any(chunk, spans):
            return rank
    return None


def recall_at_k(retrieved: list[ChunkRecord], spans: list[SourceSpanRecord], k: int) -> float:
    """1.0 if any of the top-k retrieved chunks hits a labeled span, else 0.0."""
    rank = first_hit_rank(retrieved[:k], spans)
    return 1.0 if rank is not None else 0.0


def reciprocal_rank(retrieved: list[ChunkRecord], spans: list[SourceSpanRecord]) -> float:
    """1 / rank of the first hit (0.0 if none retrieved)."""
    rank = first_hit_rank(retrieved, spans)
    return 1.0 / rank if rank is not None else 0.0


def covered_span_count(retrieved: list[ChunkRecord], spans: list[SourceSpanRecord], k: int) -> int:
    """How many of the item's labeled spans at least one top-k chunk covers."""
    top = retrieved[:k]
    return sum(any(chunk_hits_span(chunk, span) for chunk in top) for span in spans)


def span_coverage_at_k(
    retrieved: list[ChunkRecord], spans: list[SourceSpanRecord], k: int
) -> float:
    """Fraction of the item's labeled spans covered by the top-k (1.0 when it labels no span).

    `recall_at_k` credits an item as soon as ANY labeled span is retrieved, which a multi-hop
    item satisfies by returning only one of its hops. Coverage is the multi-span refinement: it
    is the share of the evidence an answer actually needs that the context carries.
    """
    if not spans:
        return 1.0
    return covered_span_count(retrieved, spans, k) / len(spans)


def all_spans_at_k(retrieved: list[ChunkRecord], spans: list[SourceSpanRecord], k: int) -> float:
    """1.0 when EVERY labeled span is covered by the top-k, else 0.0 (the multi-hop gate)."""
    return 1.0 if span_coverage_at_k(retrieved, spans, k) == 1.0 else 0.0


def evaluate_retrieval(per_item: list[RetrievalPair], k: int) -> RetrievalMetrics:
    """Aggregate recall@k and MRR over (retrieved, gold_spans) pairs.

    Returns {n, k, recall_at_k, mrr}. Empty input yields zeros.
    """
    n = len(per_item)
    if n == 0:
        return {"n": 0, "k": k, "recall_at_k": 0.0, "mrr": 0.0}
    recall = sum(recall_at_k(r, s, k) for r, s in per_item) / n
    mrr = sum(reciprocal_rank(r, s) for r, s in per_item) / n
    return {"n": n, "k": k, "recall_at_k": recall, "mrr": mrr}
