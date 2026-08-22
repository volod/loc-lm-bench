"""Near-duplicate suppression against earlier draft bundles (yield-max).

A coverage-target rerun (or a second corpus pass) can re-draft paraphrases of questions a reviewer
already saw. This drops a drafted item whose question is a near-duplicate of ANY question in one or
more prior bundles, measured by cosine similarity of the PINNED E5 embedding -- the same embedder
the RAG store uses, so "similar" means the same thing the retriever sees.

The embedder is injectable behind a tiny protocol, so the filter is unit-tested with a deterministic
fake embedder (no sentence-transformers, no GPU); the real path uses `llb.rag.encoders.embedder.Embedder`,
which needs the `[rag]` extra.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from llb.goldset.schema import GoldItem, load_goldset
from llb.prep.ontology.constants import GOLDSET_FILENAME, NEAR_DUP_COSINE_THRESHOLD

_LOG = logging.getLogger(__name__)

Vector = list[float]


@dataclass(frozen=True)
class _EmbeddedCandidates:
    """Aligned candidate and prior embeddings used during one filter pass."""

    item_questions: list[Vector]
    prior_questions: list[Vector]
    item_answers: list[Vector] | None
    prior_answers: list[Vector] | None


class QuestionEmbedder(Protocol):
    """Minimal embedder seam: map questions to vectors (order-preserving)."""

    def embed(self, texts: list[str]) -> list[Vector]:
        """Return one vector per input text."""


class E5QuestionEmbedder:
    """Adapts the pinned RAG `Embedder` (multilingual-e5) to the `QuestionEmbedder` seam."""

    def __init__(self, model_name: str | None = None):
        from llb.core.config_validation import DEFAULT_EMBEDDING_MODEL
        from llb.rag.encoders.embedder import Embedder

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


def _normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def _matching_indices(similarities: list[float], threshold: float) -> list[int]:
    return [index for index, similarity in enumerate(similarities) if similarity >= threshold]


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
        prior_ids: list[str] | None = None,
    ):
        if prior_answers is not None and len(prior_answers) != len(prior_questions):
            raise ValueError("prior_answers must align with prior_questions")
        if prior_ids is not None and len(prior_ids) != len(prior_questions):
            raise ValueError("prior_ids must align with prior_questions")
        self._prior_questions = prior_questions
        self._prior_answers = prior_answers
        # Carried so a drop names the prior ITEM it lost to, not only that prior's text: the
        # shadow linkage lane looks the decided pair up by record id.
        self._prior_ids = prior_ids
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

    def _report(self, checked: int, dropped: list[dict[str, object]]) -> dict[str, object]:
        return {
            "enabled": True,
            "threshold": self._threshold,
            "prior_questions": len(self._prior_questions),
            "checked": checked,
            "dropped": len(dropped),
            "dropped_ids": [row["id"] for row in dropped],
            "dropped_detail": dropped,
            "answer_aware": self._prior_answers is not None,
            "answer_threshold": (
                self._answer_threshold if self._prior_answers is not None else None
            ),
        }

    def _embed_candidates(
        self, items: list[GoldItem], prior_questions: list[Vector]
    ) -> _EmbeddedCandidates:
        prior_answers = self._prior_answer()
        item_answers = (
            self._embedder.embed([item.reference_answer for item in items])
            if prior_answers is not None
            else None
        )
        return _EmbeddedCandidates(
            item_questions=self._embedder.embed([item.question for item in items]),
            prior_questions=prior_questions,
            item_answers=item_answers,
            prior_answers=prior_answers,
        )

    def _answer_matches(
        self,
        item: GoldItem,
        item_index: int,
        candidates: _EmbeddedCandidates,
        eligible: list[int],
    ) -> tuple[list[int], list[float] | None]:
        if candidates.prior_answers is None or candidates.item_answers is None:
            return eligible, None
        similarities = [
            _cosine(candidates.item_answers[item_index], prior_vector)
            for prior_vector in candidates.prior_answers
        ]
        exact_indices = {
            index
            for index, question in enumerate(self._prior_questions)
            if _normalize_question(question) == _normalize_question(item.question)
        }
        answer_matches = [
            index
            for index in eligible
            if index in exact_indices or similarities[index] >= self._answer_threshold
        ]
        return answer_matches, similarities

    def _dropped_detail(
        self,
        item: GoldItem,
        item_index: int,
        candidates: _EmbeddedCandidates,
    ) -> dict[str, object] | None:
        similarities = [
            _cosine(candidates.item_questions[item_index], prior_vector)
            for prior_vector in candidates.prior_questions
        ]
        eligible = _matching_indices(similarities, self._threshold)
        eligible, answer_similarities = self._answer_matches(item, item_index, candidates, eligible)
        if not eligible:
            return None
        best_index = max(eligible, key=similarities.__getitem__)
        detail: dict[str, object] = {
            "id": item.id,
            "max_similarity": round(similarities[best_index], 4),
            "nearest_prior_question": self._prior_questions[best_index],
            "candidate_question": item.question,
        }
        if self._prior_ids is not None:
            detail["nearest_prior_id"] = self._prior_ids[best_index]
        if answer_similarities is not None and self._prior_answers is not None:
            detail.update(
                {
                    "answer_similarity": round(answer_similarities[best_index], 4),
                    "nearest_prior_answer": self._prior_answers[best_index],
                    "candidate_answer": item.reference_answer,
                }
            )
        return detail

    def filter(self, items: list[GoldItem]) -> tuple[list[GoldItem], dict[str, object]]:
        """Return (kept items, report). No prior questions -> everything is kept."""
        if not items:
            return items, self._report(0, [])
        prior = self._prior()
        if not prior:
            return items, self._report(len(items), [])

        candidates = self._embed_candidates(items, prior)
        kept: list[GoldItem] = []
        dropped: list[dict[str, object]] = []
        for item_index, item in enumerate(items):
            detail = self._dropped_detail(item, item_index, candidates)
            if detail is None:
                kept.append(item)
            else:
                dropped.append(detail)
        report = self._report(len(items), dropped)
        _LOG.info(
            "[ontology] dedup: dropped %d/%d near-duplicates of %d prior questions (>= %.2f)",
            len(dropped),
            len(items),
            len(self._prior_questions),
            self._threshold,
        )
        return kept, report
