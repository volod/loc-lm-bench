"""Cross-encoder ordering for the claim tier and its cost/recall reading.

The scorer changes only the order in which semantic candidates reach adjudication. The existing
claim budget remains the sole row cap, and callers retain every unadjudicated semantic finding.
A full-list run can therefore measure the smaller prefix that would recover the same actionable
rows without treating a cross-encoder score as a probability or confidence.
"""

import time
from dataclasses import dataclass

from llb.conflicts.claim.precision import AdjudicatedRow
from llb.conflicts.null_research.controls.cross_encoder import (
    ScoredRows,
    calibration_curve,
    score_pairs,
)
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord
from llb.rag.rerank import RerankScorer

# Scores inside this numerical tolerance carry no usable ordering. Preserving the cosine order is
# both deterministic and the promised no-op fallback for an actually flat scorer.
FLAT_SCORE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RankedClaimCandidate:
    """One semantic candidate in cross-encoder order, with its original rank retained."""

    original_index: int
    cosine_rank: int
    rerank_rank: int
    left: int
    right: int
    cosine: float
    cross_encoder_score: float

    def pair(self) -> tuple[int, int, float]:
        return self.left, self.right, self.cosine


@dataclass(frozen=True)
class ClaimPrefilterRanking:
    """The ordered shortlist plus the facts needed to explain how it was produced."""

    candidates: list[RankedClaimCandidate]
    seconds: float
    flat_scores: bool

    def pairs(self) -> list[tuple[int, int, float]]:
        return [candidate.pair() for candidate in self.candidates]


def rank_claim_candidates(
    pairs: list[tuple[int, int, float]],
    chunks: list[ChunkRecord],
    scorer: RerankScorer,
) -> ClaimPrefilterRanking:
    """Score all candidate passages and return a stable, descending cross-encoder order."""
    started = time.monotonic()
    text_pairs = [(chunks[left]["text"], chunks[right]["text"]) for left, right, _ in pairs]
    scores = score_pairs(scorer, text_pairs)
    spread = max(scores) - min(scores) if scores else 0.0
    flat = spread <= FLAT_SCORE_TOLERANCE
    order = list(range(len(pairs)))
    if not flat:
        order.sort(key=lambda index: -scores[index])
    ranked = [
        RankedClaimCandidate(
            original_index=index,
            cosine_rank=index + 1,
            rerank_rank=rank + 1,
            left=pairs[index][0],
            right=pairs[index][1],
            cosine=pairs[index][2],
            cross_encoder_score=scores[index],
        )
        for rank, index in enumerate(order)
    ]
    return ClaimPrefilterRanking(ranked, time.monotonic() - started, flat)


def _chunk_key(chunk: ChunkRecord, ordinal: int) -> str:
    return str(chunk.get("chunk_id") or f"{chunk['doc_id']}#{ordinal}")


def _same_conflicts_reading(
    ranking: ClaimPrefilterRanking,
    rows: list[AdjudicatedRow],
    calibration: JsonObject,
) -> JsonObject:
    complete = len(rows) == len(ranking.candidates)
    payload: JsonObject = {
        "evaluated": complete,
        "conflict_rows_lost": 0 if complete else None,
    }
    if not complete:
        payload["reason"] = (
            "the claim cap left candidate rows unadjudicated; their provisional semantic rows are "
            "recorded, but this bundle cannot label what the cap omitted"
        )
        return payload

    actionable = [candidate for candidate, row in zip(ranking.candidates, rows) if row.actionable]
    reranked_needed = max((candidate.rerank_rank for candidate in actionable), default=0)
    cosine_needed = max((candidate.cosine_rank for candidate in actionable), default=0)
    saved = cosine_needed - reranked_needed
    informative = bool(
        actionable and calibration.get("resolved") and calibration.get("monotone") and saved > 0
    )
    payload.update(
        {
            "actionable_rows": len(actionable),
            "reranked_rows_needed": reranked_needed,
            "cosine_rows_needed": cosine_needed,
            "observed_rank_delta": saved,
            "adjudication_calls_saved": saved if informative else 0,
            "informative": informative,
            # A non-monotone corpus does not license a smaller cross-encoder prefix even when one
            # happened to contain these labels. Its safe recommendation is today's full list.
            "recommended_claim_budget": reranked_needed if informative else len(ranking.candidates),
        }
    )
    if not informative:
        payload["fallback"] = "full_candidate_list"
    return payload


def _ranking_ledger(
    ranking: ClaimPrefilterRanking,
    rows: list[AdjudicatedRow],
    chunks: list[ChunkRecord],
) -> list[JsonObject]:
    """Every scored candidate, including rows a claim cap left provisional."""
    verdicts = {candidate.rerank_rank: row for candidate, row in zip(ranking.candidates, rows)}
    ledger: list[JsonObject] = []
    for candidate in ranking.candidates:
        row = verdicts.get(candidate.rerank_rank)
        ledger.append(
            {
                "rerank_rank": candidate.rerank_rank,
                "cosine_rank": candidate.cosine_rank,
                "left_chunk": _chunk_key(chunks[candidate.left], candidate.left),
                "right_chunk": _chunk_key(chunks[candidate.right], candidate.right),
                "cosine": round(candidate.cosine, 6),
                "cross_encoder_score": round(candidate.cross_encoder_score, 6),
                "adjudicated": row is not None,
                "relation": row.relation if row is not None else None,
                "actionable": row.actionable if row is not None else None,
                "parsed": row.parsed if row is not None else None,
            }
        )
    return ledger


def prefilter_artifact(
    ranking: ClaimPrefilterRanking,
    rows: list[AdjudicatedRow],
    chunks: list[ChunkRecord],
    *,
    model: str,
    device: str,
    adjudication_order: str = "cross_encoder",
) -> JsonObject:
    """Build the complete ranking ledger and the safe cost reading for one corpus."""
    labelled_scores = [
        candidate.cross_encoder_score for candidate in ranking.candidates[: len(rows)]
    ]
    labels: list[bool | None] = [row.actionable if row.parsed else None for row in rows]
    calibration = calibration_curve(
        ScoredRows(rows=[], cosines=[], scores=labelled_scores, actionable=labels)
    )
    scores = [candidate.cross_encoder_score for candidate in ranking.candidates]
    changed = sum(
        candidate.cosine_rank != candidate.rerank_rank for candidate in ranking.candidates
    )
    artifact: JsonObject = {
        "method": "cross_encoder_claim_prefilter",
        "model": model,
        "device": device,
        "candidate_rows": len(ranking.candidates),
        "adjudicated_rows": len(rows),
        "unadjudicated_rows": len(ranking.candidates) - len(rows),
        "model_calls_avoided_vs_full_list": len(ranking.candidates) - len(rows),
        "scoring_seconds": round(ranking.seconds, 3),
        "score_range": [round(min(scores), 6), round(max(scores), 6)] if scores else None,
        "flat_scores": ranking.flat_scores,
        "ordering": "cosine_fallback" if ranking.flat_scores else "cross_encoder",
        "adjudication_order": adjudication_order,
        "rows_moved": changed,
        "calibration": calibration,
        "rows": _ranking_ledger(ranking, rows, chunks),
    }
    artifact["same_conflicts"] = _same_conflicts_reading(ranking, rows, calibration)
    return artifact
