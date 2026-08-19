"""Pure scoring for the reranker bake-off: one candidate's pass over the SHARED candidate pools.

Every candidate reranks the IDENTICAL pool -- same encoder, same chunking, same retrieval depth,
same items -- so the only thing that differs between two rows is the cross-encoder. That is what
makes the paired reading a statement about rerankers rather than about retrieval, and it is why the
pool is retrieved once by the caller and passed in here rather than re-retrieved per candidate.

Nothing in this module loads a model: the scorer is the injectable `RerankScorer` seam, so ranking,
the metric rows, the latency arithmetic, and the floor pools are all unit-tested with a fake.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, cast

from llb.core.contracts.rag import ChunkRecord, RetrievalPair, SourceSpanRecord
from llb.rag.embedding_bakeoff_models import BakeoffItem
from llb.rag.rerank import RerankScorer, rerank_chunks
from llb.rag.rerank_bakeoff.families import resolve_convention
from llb.rag.card_parity import CardParityResult
from llb.rag.rerank_bakeoff.models import (
    KIND_RERANK,
    KIND_RETRIEVAL_ORDER,
    LoadedScorer,
    RerankCandidateResult,
)
from llb.rag.retrieval import evaluate_retrieval, first_hit_rank

_LOG = logging.getLogger(__name__)

MS_PER_S = 1000.0

# How often a long pass reports where it is. Ten items is a few seconds of work for a large
# cross-encoder and noise-free for a small one.
PROGRESS_EVERY_ITEMS = 10

# Per-item candidate pools in retrieval order, one list per scored item.
Pools = list[list[ChunkRecord]]
Clock = Callable[[], float]


@dataclass(slots=True)
class CandidatePass:
    """One reranker's finished pass: the fully re-sorted pools and what the pass cost."""

    ranked: Pools
    seconds: float


def rerank_pass(
    pools: Pools,
    items: Sequence[BakeoffItem],
    scorer: RerankScorer,
    clock: Clock = time.perf_counter,
    label: str = "",
) -> CandidatePass:
    """Re-sort every pool with `scorer`, timing only the reranking.

    The WHOLE pool is re-sorted rather than just the kept top-k: the prefix is the top-k the row is
    scored on (the sort is stable, so the two agree exactly), and the tail is what the measurement
    floor perturbs. One scorer call per item, which is the call an operator pays for at serve time.

    A pass over a real roster is minutes of silence otherwise -- a large cross-encoder scores a
    30-chunk pool in the better part of a second -- so progress is logged every
    `PROGRESS_EVERY_ITEMS` items with the rate measured so far.
    """
    ranked: Pools = []
    started = clock()
    for index, (pool, (question, _spans)) in enumerate(zip(pools, items), 1):
        ranked.append(rerank_chunks(question, pool, max(len(pool), 1), scorer) if pool else [])
        if index % PROGRESS_EVERY_ITEMS == 0:
            elapsed = clock() - started
            _LOG.info(
                "[compare-rerankers] %s: %d/%d items (%.0f ms/query)",
                label or "candidate",
                index,
                len(pools),
                elapsed * MS_PER_S / index,
            )
    return CandidatePass(ranked=ranked, seconds=clock() - started)


def retrieval_order_pass(pools: Pools) -> CandidatePass:
    """The reranker-OFF pass: the pool exactly as retrieval ordered it, at zero model cost."""
    return CandidatePass(ranked=[list(pool) for pool in pools], seconds=0.0)


def scored_pairs(ranked: Pools, items: Sequence[BakeoffItem], k: int) -> list[RetrievalPair]:
    """The (top-k ranking, gold spans) pairs the row and its per-item vectors are both read from."""
    return [(pool[:k], spans) for pool, (_question, spans) in zip(ranked, items)]


def first_hit_stats(pairs: list[RetrievalPair]) -> tuple[float | None, int]:
    """Mean rank of the first hit inside k, and how many items had one.

    Reported as a pair because the mean alone is not comparable across candidates: a reranker that
    finds evidence for FEWER items can post a better mean rank on the items it still finds.
    """
    ranks = [rank for hits, spans in pairs if (rank := first_hit_rank(hits, spans)) is not None]
    if not ranks:
        return None, 0
    return sum(ranks) / len(ranks), len(ranks)


def candidate_row(
    model: str,
    pairs: list[RetrievalPair],
    *,
    k: int,
    pool_depth: int,
    seconds: float,
    kind: str = KIND_RERANK,
    loaded: LoadedScorer | None = None,
    peak_vram_mb: float | None = None,
    fits_headroom: bool | None = None,
    card_parity: CardParityResult | None = None,
) -> RerankCandidateResult:
    """Shape one row from a finished pass: rank quality first, then what it cost to get it."""
    metrics = evaluate_retrieval(pairs, k)
    mean_rank, hit_items = first_hit_stats(pairs)
    n = max(metrics["n"], 1)
    convention = resolve_convention(model)
    row: RerankCandidateResult = {
        "model": model,
        "kind": kind,
        "family": convention.family if kind == KIND_RERANK else KIND_RETRIEVAL_ORDER,
        "recall_at_k": metrics["recall_at_k"],
        "mrr": metrics["mrr"],
        "first_hit_rank_mean": round(mean_rank, 3) if mean_rank is not None else None,
        "hit_items": hit_items,
        "n": metrics["n"],
        "k": metrics["k"],
        "pool_depth": pool_depth,
        "rerank_ms_per_query": round(seconds * MS_PER_S / n, 3),
        "pairs_per_second": round(n * pool_depth / seconds, 1) if seconds > 0 else 0.0,
    }
    if card_parity is not None:
        row["card_parity"] = card_parity
    if kind == KIND_RERANK:
        if convention.trust_remote_code:
            row["trust_remote_code"] = True
        if convention.default_prompt:
            row["default_prompt"] = convention.default_prompt
    if loaded is not None:
        row["load_seconds"] = round(loaded.load_seconds, 3)
        row["vram_mb"] = loaded.vram_mb
        if loaded.device is not None:
            row["device"] = loaded.device
    if peak_vram_mb is not None:
        row["vram_peak_mb"] = peak_vram_mb
    if fits_headroom is not None:
        row["fits_headroom"] = fits_headroom
    return row


def floor_pools(
    ranked: Pools, items: Sequence[BakeoffItem], score_key: str
) -> list[tuple[list[ChunkRecord], list[SourceSpanRecord]]]:
    """Pools re-keyed so the measurement floor perturbs the score the ROW was ranked on.

    The floor jitters `retrieval_score` (`llb.rag.noise_floor`), which for a reranked lane is the
    wrong number: the ranking came from the cross-encoder. Each chunk is copied with its rerank
    score written into that key, so the floor answers "how much of this lane's rank order is decided
    by noise in the RERANKER's scores" -- the question the row's recommendation rests on.
    """
    return [
        (
            [_rescored(chunk, score_key) for chunk in pool],
            spans,
        )
        for pool, (_question, spans) in zip(ranked, items)
    ]


def _rescored(chunk: ChunkRecord, score_key: str) -> ChunkRecord:
    """A copy of `chunk` whose `retrieval_score` is the value under `score_key` (when present)."""
    copy = dict(chunk)
    score = copy.get(score_key)
    if isinstance(score, (int, float)):
        copy["retrieval_score"] = float(score)
    return cast(ChunkRecord, copy)


def _score_of(chunk: ChunkRecord) -> float:
    """The chunk's ranking score, 0.0 when it carries none."""
    score = chunk.get("retrieval_score")
    return float(score) if isinstance(score, (int, float)) else 0.0


def score_scale(pools: list[tuple[list[ChunkRecord], list[SourceSpanRecord]]]) -> float:
    """Median within-pool score RANGE, the scale a lane's floor jitter has to be read against.

    Cross-encoders do not share a score scale: a sigmoid head lives in 0..1 while a logit head
    spans tens, so one absolute jitter would be a far tighter floor for the logit model purely
    because its numbers are bigger. Returns 1.0 for a lane with no measurable range, which leaves
    the shared default jitter unchanged.
    """
    ranges = [
        max(scores) - min(scores)
        for pool, _spans in pools
        if (scores := [_score_of(chunk) for chunk in pool])
    ]
    if not ranges:
        return 1.0
    ordered = sorted(ranges)
    median = ordered[len(ordered) // 2]
    return float(median) if median > 0 else 1.0
