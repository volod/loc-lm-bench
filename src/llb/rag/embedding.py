"""Pinned text embedder (sentence-transformers, lazy-loaded) with per-family conventions.

The embedding model is validated separately and PINNED (Premise 4): a weak Ukrainian
embedder silently caps every generation model's RAG score. This wraps one
SentenceTransformer behind a tiny interface and applies each model FAMILY's required
query/passage convention, because a retrieval-tuned encoder scored with the WRONG
convention silently loses recall -- exactly the failure the embedding bake-off
(`llb compare-embeddings`, `src/llb/rag/embedding_bakeoff.py`) must never introduce:

  - e5      (`intfloat/multilingual-e5-*`): "query: " / "passage: " prefixes.
  - bge-m3  (`BAAI/bge-m3`): NO instruction on either side (FlagEmbedding retrieval default).
  - bge     (other BGE retrieval lines, e.g. `bge-large-en-v1.5`): a query-only instruction.
  - plain   (paraphrase / STS models like `lang-uk/ukr-paraphrase-multilingual-mpnet-base`):
            symmetric, no prefix.

Heavy imports (`sentence_transformers`, `numpy`) are deferred to first use so the module
imports fine in the base install; the real embedding path needs the `[rag]` extra.
"""

import os
from typing import Any

from llb import env
from llb.config import DEFAULT_EMBEDDING_MODEL

# Per-family query/passage conventions. Retrieval-tuned encoders expect an asymmetric
# instruction on the QUERY side (and, for e5, a "passage:" tag on the passage side); applying
# the wrong convention caps recall, so the bake-off scores every candidate under its own family.
FAMILY_E5 = "e5"
FAMILY_BGE_M3 = "bge-m3"
FAMILY_BGE = "bge"
FAMILY_PLAIN = "plain"

E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
# BGE v1.5 / bge-large retrieval instruction (English line). BGE-M3 needs NO instruction on
# either side, so it resolves to FAMILY_BGE_M3 and is deliberately excluded here.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def embedding_family(model_name: str) -> str:
    """Resolve the query/passage convention family for a model id (case-insensitive)."""
    name = model_name.lower()
    if FAMILY_E5 in name:
        return FAMILY_E5
    if "bge-m3" in name or "bge_m3" in name:
        return FAMILY_BGE_M3
    if "bge" in name:
        return FAMILY_BGE
    return FAMILY_PLAIN


def apply_query_convention(model_name: str, texts: list[str]) -> list[str]:
    """Prefix `texts` as QUERIES per the model family (no-op for symmetric families)."""
    family = embedding_family(model_name)
    if family == FAMILY_E5:
        return [E5_QUERY_PREFIX + t for t in texts]
    if family == FAMILY_BGE:
        return [BGE_QUERY_INSTRUCTION + t for t in texts]
    return list(texts)


def apply_passage_convention(model_name: str, texts: list[str]) -> list[str]:
    """Prefix `texts` as PASSAGES per the model family (only e5 tags the passage side)."""
    if embedding_family(model_name) == FAMILY_E5:
        return [E5_PASSAGE_PREFIX + t for t in texts]
    return list(texts)


class Embedder:
    """Lazy wrapper over a SentenceTransformer; normalizes vectors for cosine/IP search."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None):
        self.model_name = model_name
        self._device = device
        self._model = None

    @property
    def family(self) -> str:
        """The query/passage convention family this model belongs to."""
        return embedding_family(self.model_name)

    def _resolve_device(self) -> str | None:
        """Device for the SentenceTransformer: explicit constructor arg wins, else the
        `LLB_EMBED_DEVICE` env knob, else `None` (sentence-transformers auto-selects)."""
        return self._device or os.environ.get(env.LLB_EMBED_DEVICE) or None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                from transformers.utils.logging import disable_progress_bar
            except ImportError as exc:
                raise SystemExit(
                    'ERROR: embeddings need the [rag] extra. Run: uv pip install -e ".[rag]"'
                ) from exc
            # Persisted CLI logs must remain line-oriented ASCII, not contain tqdm control output.
            disable_progress_bar()
            self._model = SentenceTransformer(self.model_name, device=self._resolve_device())
        return self._model

    def encode_passages(self, texts: list[str]) -> Any:
        """Embed corpus chunks. Returns a float32 (n, dim) numpy array, L2-normalized."""
        import numpy as np

        model = self._load()
        vectors = model.encode(
            apply_passage_convention(self.model_name, texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def encode_queries(self, texts: list[str]) -> Any:
        """Embed questions. Returns a float32 (n, dim) numpy array, L2-normalized."""
        import numpy as np

        model = self._load()
        vectors = model.encode(
            apply_query_convention(self.model_name, texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")
