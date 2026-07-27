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
        from llb.core.config_validation import DEFAULT_EMBEDDING_MODEL
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


def load_prior_items(bundle_dirs: list[Path | str]) -> list[GoldItem]:
    """Collect prior gold rows in bundle order (missing bundles are skipped with the same warning)."""
    items: list[GoldItem] = []
    for bundle in bundle_dirs:
        path = Path(bundle) / GOLDSET_FILENAME
        if not path.is_file():
            _LOG.warning("[ontology] dedup: prior bundle has no %s: %s", GOLDSET_FILENAME, bundle)
            continue
        items.extend(load_goldset(path))
    return items


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
        prior_answers: list[str] | None = None,
        answer_threshold: float | None = None,
    ):
        if prior_answers is not None and len(prior_answers) != len(prior_questions):
            raise ValueError("prior_answers must align with prior_questions")
        self._prior_questions = prior_questions
        self._prior_answers = prior_answers
        self._answer_threshold = threshold if answer_threshold is None else answer_threshold
        self._embedder = embedder
        self._threshold = threshold
        self._prior_vectors: list[Vector] | None = None
        self._prior_answer_vectors: list[Vector] | None = None

    def _prior(self) -> list[Vector]:
        if self._prior_vectors is None:
            self._prior_vectors = self._embedder.embed(self._prior_questions)
        return self._prior_vectors

    def _prior_answer(self) -> list[Vector] | None:
        if self._prior_answers is None:
            return None
        if self._prior_answer_vectors is None:
            self._prior_answer_vectors = self._embedder.embed(self._prior_answers)
        return self._prior_answer_vectors

    def _empty_report(self, checked: int) -> dict[str, object]:
        return {
            "enabled": True,
            "threshold": self._threshold,
            "prior_questions": len(self._prior_questions),
            "checked": checked,
            "dropped": 0,
            "dropped_ids": [],
            "dropped_detail": [],
            "answer_aware": self._prior_answers is not None,
            "answer_threshold": (
                self._answer_threshold if self._prior_answers is not None else None
            ),
        }

    def filter(self, items: list[GoldItem]) -> tuple[list[GoldItem], dict[str, object]]:
        """Return (kept items, report). No prior questions -> everything is kept."""
        if not items:
            return items, self._empty_report(0)
        prior = self._prior()
        if not prior:
            return items, self._empty_report(len(items))

        item_vectors = self._embedder.embed([item.question for item in items])
        prior_answers = self._prior_answer()
        item_answer_vectors = (
            self._embedder.embed([item.reference_answer for item in items])
            if prior_answers is not None
            else None
        )
        kept: list[GoldItem] = []
        dropped: list[dict[str, object]] = []
        for item_index, (item, vector) in enumerate(zip(items, item_vectors)):
            similarities = [_cosine(vector, prior_vector) for prior_vector in prior]
            normalized = " ".join(item.question.casefold().split())
            exact_indices = {
                index
                for index, question in enumerate(self._prior_questions)
                if " ".join(question.casefold().split()) == normalized
            }
            eligible = [
                index
                for index, similarity in enumerate(similarities)
                if similarity >= self._threshold
            ]
            answer_similarities: list[float] | None = None
            if prior_answers is not None and item_answer_vectors is not None:
                answer_similarities = [
                    _cosine(item_answer_vectors[item_index], prior_vector)
                    for prior_vector in prior_answers
                ]
                eligible = [
                    index
                    for index in eligible
                    if index in exact_indices
                    or answer_similarities[index] >= self._answer_threshold
                ]
            if not eligible:
                kept.append(item)
                continue
            best_index = max(eligible, key=similarities.__getitem__)
            best = similarities[best_index]
            detail: dict[str, object] = {
                "id": item.id,
                "max_similarity": round(best, 4),
                "nearest_prior_question": self._prior_questions[best_index],
                "candidate_question": item.question,
            }
            if answer_similarities is not None and self._prior_answers is not None:
                detail.update(
                    {
                        "answer_similarity": round(answer_similarities[best_index], 4),
                        "nearest_prior_answer": self._prior_answers[best_index],
                        "candidate_answer": item.reference_answer,
                    }
                )
            dropped.append(detail)
        report = {
            "enabled": True,
            "threshold": self._threshold,
            "prior_questions": len(self._prior_questions),
            "checked": len(items),
            "dropped": len(dropped),
            "dropped_ids": [row["id"] for row in dropped],
            "dropped_detail": dropped,
            "answer_aware": self._prior_answers is not None,
            "answer_threshold": (
                self._answer_threshold if self._prior_answers is not None else None
            ),
        }
        _LOG.info(
            "[ontology] dedup: dropped %d/%d near-duplicates of %d prior questions (>= %.2f)",
            len(dropped),
            len(items),
            len(self._prior_questions),
            self._threshold,
        )
        return kept, report
