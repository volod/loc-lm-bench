"""Late chunking: embed the WHOLE document, then pool token vectors per chunk span.

Classic chunk embedding encodes each chunk text in isolation, so a chunk that says "the
service" loses the document context naming WHICH service. Late chunking flips the order:
the document is encoded once (token embeddings contextualized by everything around them,
within the encoder's window) and each chunk's vector is the mean of the token vectors that
fall inside its char span. Chunk SPANS stay identical to the `sentence` strategy
(`llb.rag.chunking`), so a retrieval delta between `sentence` and `late` isolates the
embedding effect.

The pinned local encoders have short windows (e5-base: 512 tokens), so a long document is
processed in consecutive token windows; a token is contextualized by its window, not the
whole document -- an honest, recorded approximation of the long-context original
(Guenther et al., "Late Chunking", 2024).

Pure helpers (`pool_span_vectors`, `window_char_spans`, `encode_records_late`) are
dependency-free and unit-tested with fakes; the real token-embedding path
(`encode_chunks_late`, `encode_store_vectors`) needs the `[rag]` extra via
`Embedder.passage_token_offsets` / `Embedder.encode_passage_tokens`.
"""

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llb.core.contracts import ChunkRecord

_LOG = logging.getLogger(__name__)

# Token budget reserved inside each encoder window for special tokens plus the embedding
# family's passage prefix (e5's "passage: " is ~4 tokens; CLS/SEP add 2).
LATE_WINDOW_RESERVE_TOKENS = 16

Vector = list[float]
# (text, chunk char spans) -> one vector per span, None where no token overlapped.
DocEncoder = Callable[[str, list[tuple[int, int]]], "list[Vector | None]"]


def _normalize(vec: Vector) -> Vector:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def pool_span_vectors(
    chunk_spans: list[tuple[int, int]],
    token_spans: list[tuple[int, int]],
    token_vectors: list[Vector],
) -> "list[Vector | None]":
    """Mean-pool token vectors overlapping each chunk span; L2-normalize; None when empty."""
    out: "list[Vector | None]" = []
    for chunk_start, chunk_end in chunk_spans:
        hits = [
            vec
            for (tok_start, tok_end), vec in zip(token_spans, token_vectors)
            if tok_start < chunk_end and tok_end > chunk_start
        ]
        if not hits:
            out.append(None)
            continue
        dim = len(hits[0])
        pooled = [sum(vec[d] for vec in hits) / len(hits) for d in range(dim)]
        out.append(_normalize(pooled))
    return out


def window_char_spans(
    token_offsets: list[tuple[int, int]], max_tokens: int
) -> list[tuple[int, int]]:
    """Consecutive encoder windows as char spans: `max_tokens` tokens per window."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    windows: list[tuple[int, int]] = []
    for i in range(0, len(token_offsets), max_tokens):
        group = token_offsets[i : i + max_tokens]
        windows.append((group[0][0], group[-1][1]))
    return windows


def encode_chunks_late(
    embedder: Any, text: str, spans: list[tuple[int, int]]
) -> "list[Vector | None]":
    """Real late-chunking path over one document (needs the `[rag]` extra).

    Tokenizes the document once, encodes it window by window through the embedder's
    passage-side token-embedding hook, and pools per chunk span.
    """
    token_offsets = embedder.passage_token_offsets(text)
    if not token_offsets:
        return [None] * len(spans)
    max_tokens = max(16, int(embedder.max_seq_tokens()) - LATE_WINDOW_RESERVE_TOKENS)
    all_token_spans: list[tuple[int, int]] = []
    all_token_vectors: list[Vector] = []
    for window_start, window_end in window_char_spans(token_offsets, max_tokens):
        token_spans, token_vectors = embedder.encode_passage_tokens(text[window_start:window_end])
        all_token_spans.extend((window_start + s, window_start + e) for s, e in token_spans)
        all_token_vectors.extend(token_vectors)
    return pool_span_vectors(spans, all_token_spans, all_token_vectors)


def encode_records_late(
    records: list[ChunkRecord],
    read_text: Callable[[str], str],
    encode_doc: DocEncoder,
    encode_fallback: Callable[[list[str]], Any],
) -> list[Vector]:
    """Late-encode chunk records grouped per document, preserving record order.

    A chunk no token overlapped (or an unreadable doc) falls back to plain per-chunk
    passage encoding, so the returned matrix is always complete.
    """
    out: "list[Vector | None]" = [None] * len(records)
    _encode_per_document(records, read_text, encode_doc, out)
    _fill_fallback_vectors(records, encode_fallback, out)
    complete = [vec for vec in out if vec is not None]
    if len(complete) != len(records):
        raise ValueError("late encoding produced fewer vectors than chunk records")
    return complete


def _encode_per_document(
    records: list[ChunkRecord],
    read_text: Callable[[str], str],
    encode_doc: DocEncoder,
    out: "list[Vector | None]",
) -> None:
    """Late-encode records grouped per document into `out` (unreadable docs stay None)."""
    by_doc: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        by_doc.setdefault(record["doc_id"], []).append(i)
    for doc_id, indices in by_doc.items():
        text = read_text(doc_id)
        if not text:
            continue  # unreadable doc -> per-chunk fallback
        spans = [(records[i]["char_start"], records[i]["char_end"]) for i in indices]
        for i, vec in zip(indices, encode_doc(text, spans)):
            out[i] = vec


def _fill_fallback_vectors(
    records: list[ChunkRecord],
    encode_fallback: Callable[[list[str]], Any],
    out: "list[Vector | None]",
) -> None:
    """Plain per-chunk-encode every still-missing slot of `out` so the matrix is complete."""
    missing = [i for i, vec in enumerate(out) if vec is None]
    if not missing:
        return
    _LOG.info(
        "[late-encoding] %d/%d chunks fell back to per-chunk encoding",
        len(missing),
        len(records),
    )
    for i, vec in zip(missing, encode_fallback([records[i]["text"] for i in missing])):
        out[i] = [float(x) for x in vec]


def encode_store_vectors(records: list[ChunkRecord], corpus_root: Path | str, embedder: Any) -> Any:
    """Late-encoded float32 matrix for a store build (needs the `[rag]` extra)."""
    import functools

    import numpy as np

    if not hasattr(embedder, "passage_token_offsets"):
        raise ValueError(
            "the 'late' strategy needs a token-level local embedder "
            f"({type(embedder).__name__} does not expose token embeddings)"
        )
    corpus_root = Path(corpus_root)

    def read_text(doc_id: str) -> str:
        try:
            return (corpus_root / doc_id).read_text(encoding="utf-8")
        except OSError:
            return ""

    vectors = encode_records_late(
        records,
        read_text,
        functools.partial(encode_chunks_late, embedder),
        embedder.encode_passages,
    )
    return np.asarray(vectors, dtype="float32")
