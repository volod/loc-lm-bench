"""Native semantic chunking: break where consecutive-sentence embedding distance spikes."""

from typing import Any

from llb.rag.chunking.spans import _pack, _percentile, sentence_spans


def semantic_spans(
    text: str, size: int, embedder: Any, threshold_pct: float = 90.0
) -> list[tuple[int, int]]:
    """Native semantic chunking: break where consecutive-sentence embedding distance spikes.

    Uses the PINNED embedder (injected) and our sentence offsets, so chunks are exact source
    substrings -- unlike langchain's SemanticChunker, whose joined text breaks the span metric.
    Groups longer than `size` are packed down so the KV/retrieval budget is respected.
    """
    sents = sentence_spans(text)
    if len(sents) <= 1:
        return sents
    vectors = embedder.encode_passages([text[s:e] for s, e in sents])
    dists = [
        1.0 - sum(float(a) * float(b) for a, b in zip(vectors[i], vectors[i + 1]))
        for i in range(len(vectors) - 1)
    ]
    threshold = _percentile(dists, threshold_pct)
    groups: list[tuple[int, int]] = []
    group_start = 0
    for i, dist in enumerate(dists):
        if dist > threshold:
            groups.append((group_start, i))
            group_start = i + 1
    groups.append((group_start, len(sents) - 1))

    spans: list[tuple[int, int]] = []
    for g0, g1 in groups:
        start, end = sents[g0][0], sents[g1][1]
        if end - start <= size:
            spans.append((start, end))
        else:
            spans.extend(_pack(sents[g0 : g1 + 1], size))
    return spans
