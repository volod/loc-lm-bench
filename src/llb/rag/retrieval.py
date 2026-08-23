"""Retrieval quality metrics by SOURCE-SPAN overlap (pure Python).

A retrieved chunk is a HIT for a gold item when it covers the same document and its
character range overlaps any of the item's labeled source spans. Overlap is deliberately
generous, so the module also reports the INTACTNESS pair -- `span_char_coverage_at_k` and
`span_intact_at_k` -- which say how much of the span arrived and whether one chunk carried
it whole. Anchoring on char
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
from llb.rag.duplicates.collapse import occurrence_spans


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """True if [a_start, a_end) and [b_start, b_end) share at least one character."""
    return a_start < b_end and b_start < a_end


def _place_hits_span(place: SourceSpanRecord, span: SourceSpanRecord) -> bool:
    return place["doc_id"] == span["doc_id"] and spans_overlap(
        place["char_start"], place["char_end"], span["char_start"], span["char_end"]
    )


def chunk_hits_span(chunk: ChunkRecord, span: SourceSpanRecord) -> bool:
    """True if a retrieved chunk overlaps a labeled span in the same document.

    A chunk that collapsed byte-identical copies (`llb.rag.duplicates.collapse`) is matched at EVERY place
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


def _place_span_overlap(place: SourceSpanRecord, span: SourceSpanRecord) -> tuple[int, int] | None:
    """The character range a retrieved place and a gold span share, or None when they share none."""
    if place["doc_id"] != span["doc_id"]:
        return None
    start = max(place["char_start"], span["char_start"])
    end = min(place["char_end"], span["char_end"])
    return (start, end) if start < end else None


def _union_length(intervals: list[tuple[int, int]]) -> int:
    """Total length of the UNION of `[start, end)` ranges, so overlapping chunks count once."""
    total = 0
    reach: int | None = None
    for start, end in sorted(intervals):
        lower = start if reach is None else max(start, reach)
        if end > lower:
            total += end - lower
        if reach is None or end > reach:
            reach = end
    return total


def span_char_coverage(retrieved: list[ChunkRecord], span: SourceSpanRecord) -> float:
    """Share of ONE gold span's characters the retrieved chunks carry between them (0.0-1.0).

    A degenerate span (no characters) has nothing to carry and scores 0.0, which is also what
    `chunk_hits_span` reports for it -- the pair never disagrees about the same span.
    """
    length = span["char_end"] - span["char_start"]
    if length <= 0:
        return 0.0
    pieces = [
        overlap
        for chunk in retrieved
        for place in occurrence_spans(chunk)
        if (overlap := _place_span_overlap(place, span)) is not None
    ]
    return _union_length(pieces) / length


def span_carried_whole(retrieved: list[ChunkRecord], span: SourceSpanRecord) -> bool:
    """True when a SINGLE retrieved chunk contains the whole gold span, boundaries included."""
    if span["char_end"] <= span["char_start"]:
        return False
    return any(
        place["doc_id"] == span["doc_id"]
        and place["char_start"] <= span["char_start"]
        and place["char_end"] >= span["char_end"]
        for chunk in retrieved
        for place in occurrence_spans(chunk)
    )


def span_char_coverage_at_k(
    retrieved: list[ChunkRecord], spans: list[SourceSpanRecord], k: int
) -> float:
    """Mean per-span character coverage of the top-k (1.0 when the item labels no span).

    `recall_at_k` fires on a ONE-character overlap, so it cannot tell a chunk carrying a whole
    table row from one carrying half of it. This is the graded reading of the same top-k: how
    much of the evidence actually arrived.
    """
    if not spans:
        return 1.0
    top = retrieved[:k]
    return sum(span_char_coverage(top, span) for span in spans) / len(spans)


def span_intact_at_k(retrieved: list[ChunkRecord], spans: list[SourceSpanRecord], k: int) -> float:
    """Share of the item's spans that some SINGLE top-k chunk carries whole (1.0 with no spans).

    The strict companion to `span_char_coverage_at_k`: a span reassembled from two adjacent
    chunks is fully COVERED but not INTACT, and a model reading a cut table row sees the
    difference even though every character is somewhere in the context.
    """
    if not spans:
        return 1.0
    top = retrieved[:k]
    return sum(1.0 for span in spans if span_carried_whole(top, span)) / len(spans)


def served_chars_at_k(retrieved: list[ChunkRecord], k: int) -> int:
    """Characters the top-k context serves the model, counting an overlap as served twice.

    The COST column beside the quality ones: a lever that delivers more of a span by serving more
    text is a different trade from one that delivers the same characters in fewer pieces, and this
    is the only number that tells the two apart. It reads the served text, not the character union,
    because what the model pays for is what is laid into its prompt.
    """
    return sum(len(str(chunk.get("text", ""))) for chunk in retrieved[:k])


def evaluate_retrieval(per_item: list[RetrievalPair], k: int) -> RetrievalMetrics:
    """Aggregate the four top-k readings over (retrieved, gold_spans) pairs, plus served cost.

    Returns {n, k, recall_at_k, mrr, span_char_coverage_at_k, span_intact_at_k,
    served_chars_at_k}: whether the evidence was found, how early, how much of it arrived,
    whether one chunk carried it whole, and what serving it cost. Empty input yields zeros.
    """
    n = len(per_item)
    if n == 0:
        return {
            "n": 0,
            "k": k,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "span_char_coverage_at_k": 0.0,
            "span_intact_at_k": 0.0,
            "served_chars_at_k": 0.0,
        }
    return {
        "n": n,
        "k": k,
        "recall_at_k": sum(recall_at_k(r, s, k) for r, s in per_item) / n,
        "mrr": sum(reciprocal_rank(r, s) for r, s in per_item) / n,
        "span_char_coverage_at_k": sum(span_char_coverage_at_k(r, s, k) for r, s in per_item) / n,
        "span_intact_at_k": sum(span_intact_at_k(r, s, k) for r, s in per_item) / n,
        "served_chars_at_k": sum(served_chars_at_k(r, k) for r, _ in per_item) / n,
    }
