"""Strategy registry and the unified span dispatcher over every chunking strategy."""

from typing import Any

from llb.core.contracts.common import JsonObject
from llb.rag.chunking.recursive import recursive_spans
from llb.rag.chunking.semantic import semantic_spans
from llb.rag.chunking.spans import (
    fixed_spans,
    sentence_chunk_spans,
    validate_chunking,
)
from llb.rag.chunking.structure import heading_spans, markdown_spans, page_aligned_spans

PURE_STRATEGIES = ("fixed", "sentence")
STRATEGIES = ("fixed", "sentence", "recursive", "markdown", "semantic", "page", "heading", "late")


def chunk_spans(
    text: str,
    strategy: str,
    size: int,
    overlap: int,
    embedder: Any = None,
    page_spans: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int, JsonObject]]:
    """Unified (start, end, metadata) spans for a strategy.

    `page_spans` feeds the `page` strategy (sidecar page char spans from `doc_page_spans`);
    without them -- a plain `.md`/`.txt` doc, or `parent_child` children re-chunking a parent
    slice whose page coordinates are unknown -- `page` falls back to `recursive`.
    """
    validate_chunking(size, overlap)
    if strategy == "markdown":
        return markdown_spans(text, size, overlap)
    if strategy == "heading":
        return heading_spans(text, size, overlap)
    plain = _plain_strategy_spans(text, strategy, size, overlap, embedder, page_spans)
    return [(s, e, {}) for s, e in plain]


def _plain_strategy_spans(
    text: str,
    strategy: str,
    size: int,
    overlap: int,
    embedder: Any,
    page_spans: list[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    """(start, end) spans for the metadata-less strategies."""
    if strategy == "fixed":
        return fixed_spans(text, size, overlap)
    if strategy in ("sentence", "late"):  # late = sentence spans + late-pooled vectors
        return sentence_chunk_spans(text, size)
    if strategy == "recursive":
        return recursive_spans(text, size, overlap)
    if strategy == "page":
        if page_spans:
            return page_aligned_spans(text, size, overlap, page_spans)
        return recursive_spans(text, size, overlap)
    if strategy == "semantic":
        if embedder is None:
            raise SystemExit('ERROR: the "semantic" strategy needs an embedder (the [rag] extra).')
        return semantic_spans(text, size, embedder)
    raise ValueError(f"unknown strategy: {strategy}")
