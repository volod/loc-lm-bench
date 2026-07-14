"""Focused text analysis similarity implementation."""

from typing import Any
from llb.scoring.text_analysis import Similarity


def embedder_similarity(embedder: Any = None) -> Similarity:
    """Production `similarity`: cosine over the PINNED embedder (the text-analysis sign-off matching basis).

    Vectors are L2-normalized by the `Embedder`, so cosine is their dot product. Heavy imports
    (the embedder, numpy) stay lazy; the returned callable caches encodings per surface string so
    a label's surfaces are embedded once across many predictions.
    """
    if embedder is None:
        from llb.rag.embedding import Embedder

        embedder = Embedder()
    cache: dict[str, Any] = {}

    def _vec(text: str) -> Any:
        if text not in cache:
            cache[text] = embedder.encode_queries([text])[0]
        return cache[text]

    def similarity(a: str, b: str) -> float:
        import numpy as np

        return float(np.dot(_vec(a), _vec(b)))

    return similarity
