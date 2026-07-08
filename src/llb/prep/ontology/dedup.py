"""Near-duplicate suppression against earlier draft bundles (yield-max).

A coverage-target rerun (or a second corpus pass) can re-draft paraphrases of questions a reviewer
already saw. This drops a drafted item whose question is a near-duplicate of ANY question in one or
more prior bundles, measured by cosine similarity of the PINNED E5 embedding -- the same embedder
the RAG store uses, so "similar" means the same thing the retriever sees.

The embedder is injectable behind a tiny protocol, so the filter is unit-tested with a deterministic
fake embedder (no sentence-transformers, no GPU); the real path uses `llb.rag.embedding.Embedder`,
which needs the `[rag]` extra.
"""

import logging
import math
from pathlib import Path
from typing import Protocol

from llb.goldset.schema import GoldItem, load_goldset
from llb.prep.ontology.constants import GOLDSET_FILENAME, NEAR_DUP_COSINE_THRESHOLD

_LOG = logging.getLogger(__name__)

Vector = list[float]


class QuestionEmbedder(Protocol):
    """Minimal embedder seam: map questions to vectors (order-preserving)."""

    def embed(self, texts: list[str]) -> list[Vector]:
        """Return one vector per input text."""


class E5QuestionEmbedder:
    """Adapts the pinned RAG `Embedder` (multilingual-e5) to the `QuestionEmbedder` seam."""

    def __init__(self, model_name: str | None = None):
        from llb.core.config import DEFAULT_EMBEDDING_MODEL
        from llb.rag.embedding import Embedder

        self._embedder = Embedder(model_name or DEFAULT_EMBEDDING_MODEL)

    def embed(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        return [list(map(float, row)) for row in self._embedder.encode_queries(texts)]


def load_prior_questions(bundle_dirs: list[Path | str]) -> list[str]:
    """Collect the questions of every prior bundle's `goldset.jsonl` (missing bundles are skipped)."""
    questions: list[str] = []
    for bundle in bundle_dirs:
        path = Path(bundle) / GOLDSET_FILENAME
        if not path.is_file():
            _LOG.warning("[ontology] dedup: prior bundle has no %s: %s", GOLDSET_FILENAME, bundle)
            continue
        questions.extend(item.question for item in load_goldset(path))
    return questions


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class NearDuplicateFilter:
    """Drop drafted items whose question is a near-duplicate of any prior-bundle question."""

    def __init__(
        self,
        prior_questions: list[str],
        embedder: QuestionEmbedder,
        *,
        threshold: float = NEAR_DUP_COSINE_THRESHOLD,
    ):
        self._prior_questions = prior_questions
        self._embedder = embedder
        self._threshold = threshold
        self._prior_vectors: list[Vector] | None = None

    def _prior(self) -> list[Vector]:
        if self._prior_vectors is None:
            self._prior_vectors = self._embedder.embed(self._prior_questions)
        return self._prior_vectors

    def filter(self, items: list[GoldItem]) -> tuple[list[GoldItem], dict[str, object]]:
        """Return (kept items, report). No prior questions -> everything is kept."""
        prior = self._prior()
        if not prior or not items:
            report = {
                "enabled": True,
                "threshold": self._threshold,
                "prior_questions": len(self._prior_questions),
                "checked": len(items),
                "dropped": 0,
                "dropped_ids": [],
            }
            return items, report

        item_vectors = self._embedder.embed([item.question for item in items])
        kept: list[GoldItem] = []
        dropped: list[dict[str, object]] = []
        for item, vector in zip(items, item_vectors):
            best = max((_cosine(vector, pv) for pv in prior), default=0.0)
            if best >= self._threshold:
                dropped.append({"id": item.id, "max_similarity": round(best, 4)})
            else:
                kept.append(item)
        report = {
            "enabled": True,
            "threshold": self._threshold,
            "prior_questions": len(self._prior_questions),
            "checked": len(items),
            "dropped": len(dropped),
            "dropped_ids": [row["id"] for row in dropped],
            "dropped_detail": dropped,
        }
        _LOG.info(
            "[ontology] dedup: dropped %d/%d near-duplicates of %d prior questions (>= %.2f)",
            len(dropped),
            len(items),
            len(self._prior_questions),
            self._threshold,
        )
        return kept, report
