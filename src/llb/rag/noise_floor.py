"""Measurement floor of a retrieval comparison: how far numeric noise alone moves recall@k.

`compare-retrieval` reports recall@k / MRR to three decimals, and the floor under those decimals
is not zero. Two processes that built BYTE-IDENTICAL chunks on the same host produced dense
vectors differing by up to 5.4e-7 per dimension -- the encoder's kernels depend on the batch
shapes it saw earlier in the process -- which moved the resulting cosine scores by up to 6.0e-7
and flipped one borderline item at k=10, worth 0.011 recall on a 95-item set. Repeat runs WITHIN
one process reproduce byte-identically, so a naive repeat check reports a spread of zero and the
drift stays invisible.

This module measures the floor instead of hoping it is small: retrieve a deeper candidate pool
once per lane, perturb every candidate score by noise of the measured amplitude, re-rank, and
report the spread of the metric over many seeded replicates. The replicates only re-sort a cached
pool, so the whole measurement costs one deeper retrieval pass per lane.

A lane's floor is large when its ranking near the cut is arbitrary -- scores separated by less
than the noise, or exact ties (a backend that ROUNDS its scores produces many). That is a real
property of the lane, not an artifact: it says the reported metric depends on tie order.

The measurement is store-agnostic: it needs a lane's candidates and their scores, nothing else.
Every comparison lane that publishes three-decimal recall@k / MRR rows therefore reads its own
floor through this module -- `compare-retrieval`, the embedder bake-off, the vector-store
comparison, and the graph-fusion sweep -- and each states its recommendation as clearing the
floor or not (`margin`), so a sub-item delta cannot be read as a ranking.
"""

import random
import statistics
import zlib

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.rag import ChunkRecord, RetrievalMetrics, SourceSpanRecord
from llb.rag.compare import CompareItem, Retriever
from llb.rag.retrieval import evaluate_retrieval

# Amplitude of the simulated per-score noise. Anchored on the measured between-process drift of
# the pinned e5-base encoder on this host: up to 6.0e-7 per cosine score (mean 1.3e-7) over 95
# questions x 4848 identical chunks. The default rounds that MAXIMUM up, so the floor is
# deliberately conservative -- a delta that clears it is not numeric noise.
DEFAULT_SCORE_JITTER = 1e-6

# Each replicate only re-sorts a cached candidate pool, so replicates are nearly free.
DEFAULT_REPLICATES = 64

# Candidate pool depth as a multiple of k. A candidate this far below the cut cannot cross it
# under noise this small, so a deeper pool would only cost time.
CANDIDATE_DEPTH_FACTOR = 3

DEFAULT_SEED = 13

_SCORE_KEY = "retrieval_score"


class MetricSpread(TypedDict):
    """Spread of one metric across the jitter replicates, plus its unjittered value."""

    base: float
    min: float
    max: float
    mean: float
    std: float
    half_width: float  # (max - min) / 2 -- the "+/-" to read beside the metric


class LaneFloor(TypedDict):
    """One lane's metric bands plus the fragility that explains how wide they are."""

    recall_at_k: MetricSpread
    mrr: MetricSpread
    n: int
    fragile_items: int  # items whose rank-k and rank-(k+1) scores sit within `jitter`


class FloorMargin(TypedDict):
    """The reading a recommendation rests on: the top two lanes and their gap versus the floor.

    Only the two best lanes matter, because a recommendation names ONE lane: if the leader's gap
    over the runner-up is inside the floor, the report has not distinguished them, whatever the
    third decimal says.
    """

    leader: str
    runner_up: str | None
    delta: float  # leader recall@k - runner-up recall@k (0.0 when there is no runner-up)
    floor: float  # the floor the delta is read against (`floor_recall_at_k`)
    clears_floor: bool


class NoiseFloorReport(TypedDict):
    """Per-lane metric spread under score noise, plus the worst-lane floor per metric."""

    replicates: int
    jitter: float
    candidates: int
    seed: int
    lanes: dict[str, LaneFloor]
    unscored: list[str]  # lanes whose candidates expose no score, so nothing can be perturbed
    floor_recall_at_k: float
    floor_mrr: float
    margin: NotRequired[FloorMargin]  # absent when no lane was measured


def measure_noise_floor(
    stores: dict[str, Retriever],
    items: list[CompareItem],
    k: int,
    replicates: int = DEFAULT_REPLICATES,
    jitter: float = DEFAULT_SCORE_JITTER,
    seed: int = DEFAULT_SEED,
) -> NoiseFloorReport:
    """Spread of recall@k / MRR per lane when every candidate score is perturbed by `jitter`.

    Each lane is retrieved twice: at `k` for `base` -- the same call the comparison row makes, so
    the two agree by construction -- and at `CANDIDATE_DEPTH_FACTOR * k` for the candidate pool
    every replicate re-ranks under fresh noise before keeping its own top k.
    """
    if replicates < 2:
        raise ValueError("noise-floor needs at least 2 replicates to have a spread")
    if jitter <= 0:
        raise ValueError("noise-floor jitter must be > 0")
    candidates = max(k, CANDIDATE_DEPTH_FACTOR * k)
    lanes: dict[str, LaneFloor] = {}
    unscored: list[str] = []
    for label, store in stores.items():
        pools = [
            (_candidate_pool(store, question, k, candidates), spans) for question, spans in items
        ]
        if not _pools_are_scored(pools):
            unscored.append(label)
            continue
        base = evaluate_retrieval(
            [(store.retrieve(question, k), spans) for question, spans in items], k
        )
        lanes[label] = _lane_floor(pools, base, k, replicates, jitter, _lane_seed(label, seed))
    report: NoiseFloorReport = {
        "replicates": replicates,
        "jitter": jitter,
        "candidates": candidates,
        "seed": seed,
        "lanes": lanes,
        "unscored": sorted(unscored),
        "floor_recall_at_k": _worst(lanes, "recall_at_k"),
        "floor_mrr": _worst(lanes, "mrr"),
    }
    margin = _margin(lanes, report["floor_recall_at_k"])
    if margin is not None:
        report["margin"] = margin
    return report


def _candidate_pool(store: Retriever, question: str, k: int, candidates: int) -> list[ChunkRecord]:
    """The lane's own top-k ranking, extended to `candidates` -- NOT a re-ranking at a deeper k.

    For most lanes the two are the same call: a dense store's top-k is the prefix of its top-3k.
    A FUSED lane is different -- its ranking depends on how deep each side was asked, so asking
    it for 3k results would fuse a deeper pool and the unjittered top-k of that pool need not be
    the row the comparison published. Such a lane exposes `retrieve_candidate_pool(question, k,
    candidates)`: fuse exactly as it does at `k`, then cut at `candidates` instead.
    """
    deeper = getattr(store, "retrieve_candidate_pool", None)
    if callable(deeper):
        pool: list[ChunkRecord] = deeper(question, k, candidates)
        return pool
    return store.retrieve(question, candidates)


def _margin(lanes: dict[str, LaneFloor], floor: float) -> FloorMargin | None:
    """Read the top two lanes' recall@k gap against the floor.

    Ranked recall first, then MRR, then label -- the order every comparison table in the repo
    ranks by (`_best_recall` in `llb.rag.compare`, `best_recall` in the bake-off), so the lane the
    margin calls the runner-up is the one the table prints second.
    """
    ranked = sorted(
        lanes,
        key=lambda label: (
            -lanes[label]["recall_at_k"]["base"],
            -lanes[label]["mrr"]["base"],
            label,
        ),
    )
    if not ranked:
        return None
    leader = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    delta = (
        lanes[leader]["recall_at_k"]["base"] - lanes[runner_up]["recall_at_k"]["base"]
        if runner_up is not None
        else 0.0
    )
    return {
        "leader": leader,
        "runner_up": runner_up,
        "delta": delta,
        "floor": floor,
        "clears_floor": runner_up is not None and delta > floor,
    }


def _pools_are_scored(pools: list[tuple[list[ChunkRecord], list[SourceSpanRecord]]]) -> bool:
    """True when at least one retrieved candidate carries a score to perturb."""
    return any(_SCORE_KEY in chunk for pool, _ in pools for chunk in pool)


def _lane_seed(label: str, seed: int) -> int:
    """Stable per-lane seed: `hash()` is salted per process, `crc32` is not."""
    return seed ^ zlib.crc32(label.encode("utf-8"))


def _lane_floor(
    pools: list[tuple[list[ChunkRecord], list[SourceSpanRecord]]],
    base: RetrievalMetrics,
    k: int,
    replicates: int,
    jitter: float,
    lane_seed: int,
) -> LaneFloor:
    sampled = [
        evaluate_retrieval(
            [(_jittered_top_k(pool, rng, jitter, k), spans) for pool, spans in pools], k
        )
        for rng in (random.Random(lane_seed + r) for r in range(replicates))
    ]
    return {
        "recall_at_k": _spread(base["recall_at_k"], [m["recall_at_k"] for m in sampled]),
        "mrr": _spread(base["mrr"], [m["mrr"] for m in sampled]),
        "n": len(pools),
        "fragile_items": sum(_is_fragile(pool, k, jitter) for pool, _ in pools),
    }


def _is_fragile(pool: list[ChunkRecord], k: int, jitter: float) -> bool:
    """True when the candidate at rank k+1 is within `jitter` of the one at rank k.

    Such an item's top-k membership is decided by noise (or, at an exact tie, by the backend's
    arbitrary candidate order) rather than by retrieval quality.
    """
    if len(pool) <= k:
        return False
    return abs(_score(pool[k - 1]) - _score(pool[k])) <= jitter


def _score(chunk: ChunkRecord) -> float:
    score = chunk.get(_SCORE_KEY)
    return float(score) if isinstance(score, (int, float)) else 0.0


def _jittered_top_k(
    pool: list[ChunkRecord], rng: random.Random, jitter: float, k: int
) -> list[ChunkRecord]:
    """Top-k of `pool` re-ranked after adding N(0, jitter) to every candidate score.

    Records are reordered, never mutated -- the caller's pool is reused by every replicate.
    """
    scored = [(-(_score(chunk) + rng.gauss(0.0, jitter)), i, chunk) for i, chunk in enumerate(pool)]
    scored.sort(key=lambda row: (row[0], row[1]))  # index tie-break keeps the sort deterministic
    return [chunk for _, _, chunk in scored[:k]]


def _spread(base: float, values: list[float]) -> MetricSpread:
    low, high = min(values), max(values)
    return {
        "base": base,
        "min": low,
        "max": high,
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "half_width": (high - low) / 2.0,
    }


def _worst(lanes: dict[str, LaneFloor], metric: str) -> float:
    """The floor to quote for the whole comparison: the widest band any lane showed."""
    return max((lane[metric]["half_width"] for lane in lanes.values()), default=0.0)  # type: ignore[literal-required]
