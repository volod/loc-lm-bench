"""Cross-encoder claim ordering, its flat fallback, and its saved-call accounting."""

import pytest

from llb.conflicts.claim.precision import AdjudicatedRow
from llb.conflicts.claim.prefilter import prefilter_artifact, rank_claim_candidates
from llb.conflicts.constants import REL_COMPLEMENTARY, REL_DUPLICATE


def _chunks(count: int) -> list[dict]:
    return [
        {
            "doc_id": f"doc-{index}.md",
            "chunk_id": f"chunk-{index}",
            "char_start": 0,
            "char_end": 20,
            "text": f"passage {index}",
        }
        for index in range(count)
    ]


def test_injected_scorer_batches_by_left_and_records_same_conflicts_saving():
    chunks = _chunks(9)
    pairs = [(0, index, 1.0 - index / 100) for index in range(1, 9)]
    scores = {
        "passage 1": 0.1,
        "passage 2": 0.2,
        "passage 3": 0.3,
        "passage 4": 0.4,
        "passage 5": 0.8,
        "passage 6": 0.7,
        "passage 7": 0.6,
        "passage 8": 0.5,
    }
    calls: list[tuple[str, list[str]]] = []

    def scorer(left: str, right: list[str]) -> list[float]:
        calls.append((left, right))
        return [scores[text] for text in right]

    ranking = rank_claim_candidates(pairs, chunks, scorer)
    assert calls == [("passage 0", [f"passage {index}" for index in range(1, 9)])]
    assert [row.cosine_rank for row in ranking.candidates[:4]] == [5, 6, 7, 8]

    adjudicated = [
        AdjudicatedRow(
            rank=candidate.rerank_rank,
            left_key="left",
            right_key=f"right-{candidate.rerank_rank}",
            score=candidate.cosine,
            relation=REL_DUPLICATE if candidate.rerank_rank <= 4 else REL_COMPLEMENTARY,
            parsed=True,
        )
        for candidate in ranking.candidates
    ]
    artifact = prefilter_artifact(ranking, adjudicated, chunks, model="injected", device="test")

    assert artifact["calibration"]["monotone"]
    assert artifact["same_conflicts"] == {
        "evaluated": True,
        "conflict_rows_lost": 0,
        "actionable_rows": 4,
        "reranked_rows_needed": 4,
        "cosine_rows_needed": 8,
        "observed_rank_delta": 4,
        "adjudication_calls_saved": 4,
        "informative": True,
        "recommended_claim_budget": 4,
    }


def test_flat_scorer_preserves_cosine_order_exactly():
    chunks = _chunks(4)
    pairs = [(0, index, 1.0 - index / 10) for index in range(1, 4)]
    ranking = rank_claim_candidates(pairs, chunks, lambda left, right: [0.5] * len(right))

    assert ranking.flat_scores
    assert ranking.pairs() == pairs
    assert [row.cosine_rank for row in ranking.candidates] == [1, 2, 3]


def test_scorer_must_return_one_score_per_candidate():
    chunks = _chunks(3)
    with pytest.raises(ValueError, match="returned 1 scores for 2 passage pairs"):
        rank_claim_candidates([(0, 1, 0.9), (0, 2, 0.8)], chunks, lambda left, right: [1.0])
