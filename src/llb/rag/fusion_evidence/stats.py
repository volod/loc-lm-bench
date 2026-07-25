"""Small-sample uncertainty for retrieval slices: paired bootstrap + exact sign test (pure).

A multi-hop slice is a handful of items, so a bare point estimate ("fused recall 0.83 vs vector
0.75") is not evidence -- the difference has to survive resampling. Every interval here is a
PAIRED percentile bootstrap over the same item index sets, so the candidate and the baseline are
always resampled together and their delta keeps the per-item pairing that makes a small slice
readable at all.

Every paired delta also carries its `stability`: how close the `lo > 0` reading sits to the cut
that produced it (`llb.rag.fusion_evidence.stability`). That annotation is FREE rather than a
second computation -- the resample means a percentile bound is read off are the same draw the
exceedance probability counts, so one pass yields both the binary reading and the continuous
quantity it thresholds. Every lane that reports a paired delta therefore gains the qualifier
without wiring it.

Pure Python and dependency-free (no numpy) so the fusion-evidence lane imports in the lightweight
CI install. Index sets are drawn once per report from a seeded `random.Random`, then shared by
every row and metric (common random numbers): deterministic, and it keeps the cost linear in the
number of replicates instead of multiplying by rows x metrics.
"""

import math
from random import Random

from typing_extensions import NotRequired, TypedDict

from llb.rag.fusion_evidence.stability import (
    LOOSER_CONFIDENCE,
    READING_FLAT,
    READING_SEPARATED,
    TIGHTER_CONFIDENCE,
    ReadingStability,
    brackets,
    exceedance,
    stability_from_readings,
)

DEFAULT_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 13


class Interval(TypedDict):
    """A point estimate with its percentile-bootstrap confidence bounds."""

    mean: float
    lo: float
    hi: float


class PairedComparison(TypedDict):
    """A candidate-minus-baseline delta plus the item-level win/loss/tie ledger behind it."""

    delta: Interval
    wins: int
    losses: int
    ties: int
    sign_test_p: float
    # How settled the `delta.lo > 0` reading is, measured from the SAME resample draw the interval
    # was taken from. Absent when no resample was drawn (an exceedance share has no meaning then)
    # or when the reporting confidence sits outside the two conventions the flag is defined
    # against -- in both cases the honest artifact carries no annotation rather than a misleading
    # zero.
    stability: NotRequired[ReadingStability]


def bootstrap_index_sets(n: int, resamples: int, seed: int) -> list[list[int]]:
    """`resamples` item index sets drawn with replacement from `range(n)` (deterministic)."""
    if n <= 0 or resamples <= 0:
        return []
    rng = Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(resamples)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ordered_percentiles(ordered: list[float], confidence: float) -> tuple[float, float]:
    """Lower/upper percentile bounds of an ALREADY SORTED bootstrap distribution (nearest-rank).

    Takes the sorted samples so one draw can be read at several confidence levels without paying
    for a sort per level -- which is what makes the borderline annotation free.
    """
    last = len(ordered) - 1
    tail = (1.0 - confidence) / 2.0
    lo = ordered[min(last, max(0, int(round(tail * last))))]
    hi = ordered[min(last, max(0, int(round((1.0 - tail) * last))))]
    return lo, hi


def _percentiles(samples: list[float], confidence: float) -> tuple[float, float]:
    """Lower/upper percentile bounds of a bootstrap distribution (nearest-rank, ASCII-safe)."""
    return _ordered_percentiles(sorted(samples), confidence)


def bootstrap_samples(values: list[float], index_sets: list[list[int]]) -> list[float]:
    """The resample MEANS behind one interval -- the draw both the bounds and `p_positive` read."""
    return [_mean([values[i] for i in indexes]) for indexes in index_sets]


def bootstrap_interval(
    values: list[float], index_sets: list[list[int]], confidence: float = DEFAULT_CONFIDENCE
) -> Interval:
    """Percentile-bootstrap interval for the mean of per-item `values`."""
    point = _mean(values)
    if not values or not index_sets:
        return {"mean": point, "lo": point, "hi": point}
    lo, hi = _percentiles(bootstrap_samples(values, index_sets), confidence)
    return {"mean": point, "lo": lo, "hi": hi}


def separation_stability(
    ordered_samples: list[float],
    confidence: float = DEFAULT_CONFIDENCE,
    *,
    looser_confidence: float = LOOSER_CONFIDENCE,
    tighter_confidence: float = TIGHTER_CONFIDENCE,
) -> ReadingStability:
    """How settled the `lo > 0` reading of one SORTED bootstrap draw is.

    The binary reading every paired lane cuts, taken at the reporting level and at both
    neighbouring conventions off the same sorted samples, plus the exceedance probability the cut
    thresholds. Cheap by construction: no extra resampling and no extra sort.
    """

    def read(level: float) -> str:
        lo, _ = _ordered_percentiles(ordered_samples, level)
        return READING_SEPARATED if lo > 0.0 else READING_FLAT

    return stability_from_readings(
        reading=read(confidence),
        looser_reading=read(looser_confidence),
        tighter_reading=read(tighter_confidence),
        p_positive=exceedance(ordered_samples),
    )


def bootstrap_ratio(
    numerators: list[bool],
    denominators: list[bool],
    index_sets: list[list[int]],
    confidence: float = DEFAULT_CONFIDENCE,
) -> Interval:
    """Bootstrap a ratio of counts, such as route precision or recall.

    A zero denominator yields 0.0: a router making no positive prediction has zero measured
    precision, not perfect precision or missing evidence.
    """
    if len(numerators) != len(denominators):
        raise ValueError("ratio needs one denominator flag per numerator flag")

    def ratio(indexes: list[int]) -> float:
        denominator = sum(denominators[i] for i in indexes)
        return sum(numerators[i] for i in indexes) / denominator if denominator else 0.0

    all_indexes = list(range(len(numerators)))
    point = ratio(all_indexes)
    if not numerators or not index_sets:
        return {"mean": point, "lo": point, "hi": point}
    lo, hi = _percentiles([ratio(indexes) for indexes in index_sets], confidence)
    return {"mean": point, "lo": lo, "hi": hi}


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign-test p-value over the non-tied pairs (1.0 when none differ)."""
    decided = wins + losses
    if decided == 0:
        return 1.0
    extreme = min(wins, losses)
    tail = sum(math.comb(decided, i) for i in range(extreme + 1)) / (2.0**decided)
    return min(1.0, 2.0 * tail)


def paired_comparison(
    candidate: list[float],
    baseline: list[float],
    index_sets: list[list[int]],
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedComparison:
    """Bootstrap the paired delta, count the item-level ledger, and say how settled the reading is.

    The stability annotation is derived from the very samples the interval's bounds come from, so
    every lane in the repo that reports a paired delta can qualify a knife-edge row without drawing
    anything extra. It is omitted -- rather than faked -- when there is no draw to count.
    """
    if len(candidate) != len(baseline):
        raise ValueError("paired comparison needs one baseline value per candidate value")
    deltas = [c - b for c, b in zip(candidate, baseline)]
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    point = _mean(deltas)
    comparison: PairedComparison = {
        "delta": {"mean": point, "lo": point, "hi": point},
        "wins": wins,
        "losses": losses,
        "ties": len(deltas) - wins - losses,
        "sign_test_p": sign_test_p(wins, losses),
    }
    if not deltas or not index_sets:
        return comparison
    ordered = sorted(bootstrap_samples(deltas, index_sets))
    lo, hi = _ordered_percentiles(ordered, confidence)
    comparison["delta"] = {"mean": point, "lo": lo, "hi": hi}
    if brackets(confidence):
        comparison["stability"] = separation_stability(ordered, confidence)
    return comparison


def format_interval(interval: Interval, places: int = 3) -> str:
    """`0.833 [0.667, 1.000]` -- the one rendering shared by every report table."""
    return (
        f"{interval['mean']:.{places}f} [{interval['lo']:.{places}f}, {interval['hi']:.{places}f}]"
    )
