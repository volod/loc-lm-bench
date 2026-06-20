"""Pinned text embedder (sentence-transformers, lazy-loaded).

The embedding model is validated separately and PINNED (Premise 4): a weak Ukrainian
embedder silently caps every generation model's RAG score. This wraps one
SentenceTransformer behind a tiny interface and applies the e5 family's required
"query:" / "passage:" prefixes when the pinned model is an e5 variant.

Heavy imports (`sentence_transformers`, `numpy`) are deferred to first use so the module
imports fine in the base install; the real embedding path needs the `[rag]` extra.
"""

from llb.config import DEFAULT_EMBEDDING_MODEL


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


class Embedder:
    """Lazy wrapper over a SentenceTransformer; normalizes vectors for cosine/IP search."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise SystemExit(
                    'ERROR: embeddings need the [rag] extra. Run: uv pip install -e ".[rag]"'
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _prefix(self, texts: list[str], kind: str) -> list[str]:
        if _is_e5(self.model_name):
            return [f"{kind}: {t}" for t in texts]
        return list(texts)

    def encode_passages(self, texts: list[str]):
        """Embed corpus chunks. Returns a float32 (n, dim) numpy array, L2-normalized."""
        import numpy as np

        model = self._load()
        vectors = model.encode(
            self._prefix(texts, "passage"), normalize_embeddings=True
        )
        return np.asarray(vectors, dtype="float32")

    def encode_queries(self, texts: list[str]):
        """Embed questions. Returns a float32 (n, dim) numpy array, L2-normalized."""
        import numpy as np

        model = self._load()
        vectors = model.encode(
            self._prefix(texts, "query"), normalize_embeddings=True
        )
        return np.asarray(vectors, dtype="float32")
