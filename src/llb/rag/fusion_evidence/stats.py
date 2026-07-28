"""Small-sample uncertainty for retrieval slices: paired bootstrap + exact sign test (pure).

A multi-hop slice is a handful of items, so a bare point estimate ("fused recall 0.83 vs vector
0.75") is not evidence -- the difference has to survive resampling. Every interval here is a
PAIRED percentile bootstrap over the same item index sets, so the candidate and the baseline are
always resampled together and their delta keeps the per-item pairing that makes a small slice
readable at all.

Every paired delta also carries its `stability`: the calibrated sign-flip p that decides the
reading, the bootstrap exceedance diagnostic, and the neighbouring-confidence readings.

The same block also carries the minimum-evidence GATE: `separates` is the one separation test every
verdict cuts on, and it reads the discordant-item ledger beside the interval, because an interval
drawn from a handful of differing items can clear zero at a level its own exact sign test could
never reach (`llb.rag.fusion_evidence.stability`).

Pure Python and dependency-free (no numpy) so the fusion-evidence lane imports in the lightweight
CI install. Index sets are drawn once per report from a seeded `random.Random`, then shared by
every row and metric (common random numbers): deterministic, and it keeps the cost linear in the
number of replicates instead of multiplying by rows x metrics.
"""

from random import Random

from typing_extensions import NotRequired, TypedDict

from llb.rag.fusion_evidence.evidence_gate import (
    READING_FLAT,
    READING_INSUFFICIENT_EVIDENCE,
    READING_SEPARATED,
    apply_evidence_gate,
    reaches_reporting_level,
)
from llb.rag.fusion_evidence.stability import (
    LOOSER_CONFIDENCE,
    ReadingStability,
    TIGHTER_CONFIDENCE,
    brackets,
    exceedance,
    stability_from_readings,
)
from llb.rag.fusion_evidence.randomization import randomization_separates

DEFAULT_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 13


class Interval(TypedDict):
    """A point estimate with its percentile-bootstrap confidence bounds."""

    mean: float
    lo: float
    hi: float


class BootstrapRatio(Interval):
    """A count ratio whose lower-bound reading is qualified from the same bootstrap draw."""

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


def interval_from_ordered_samples(
    values: list[float],
    ordered_samples: list[float],
    confidence: float = DEFAULT_CONFIDENCE,
) -> Interval:
    """An interval from an already-sorted bootstrap draw, without resampling or sorting again."""
    point = _mean(values)
    if not values or not ordered_samples:
        return {"mean": point, "lo": point, "hi": point}
    lo, hi = _ordered_percentiles(ordered_samples, confidence)
    return {"mean": point, "lo": lo, "hi": hi}


def separation_stability(
    ordered_samples: list[float],
    confidence: float = DEFAULT_CONFIDENCE,
    *,
    discordant: int,
    pairs: int,
    randomization_p: float,
    randomization_method: str,
    randomization_samples: int,
    looser_confidence: float = LOOSER_CONFIDENCE,
    tighter_confidence: float = TIGHTER_CONFIDENCE,
) -> ReadingStability:
    """How settled the calibrated reading beside one sorted bootstrap draw is.

    The binary reading every paired lane cuts, taken at the reporting level and at both
    neighbouring conventions off the same sorted samples, plus the exceedance probability the cut
    thresholds. Cheap by construction: no extra resampling and no extra sort.

    `discordant` is the count of items the two lanes actually differ on, which gates each level at
    its own reachable minimum -- so a row separated on too few items reads `insufficient_evidence`
    rather than as a difference, at whichever of the three levels cannot support it. `pairs` is the
    count it came out of, recorded beside it so a gated row carries the rate that prices the item
    set it would take to resolve.
    """

    def read(level: float) -> str:
        if randomization_separates(randomization_p, level):
            reading = READING_SEPARATED
        else:
            lo, _ = _ordered_percentiles(ordered_samples, level)
            reading = (
                READING_INSUFFICIENT_EVIDENCE
                if lo > 0.0 and not reaches_reporting_level(discordant, level)
                else READING_FLAT
            )
        return apply_evidence_gate(reading, discordant=discordant, confidence=level)

    return stability_from_readings(
        reading=read(confidence),
        looser_reading=read(looser_confidence),
        tighter_reading=read(tighter_confidence),
        p_positive=exceedance(ordered_samples),
        randomization_p=randomization_p,
        randomization_method=randomization_method,
        randomization_samples=randomization_samples,
        discordant=discordant,
        pairs=pairs,
    )


def ratio_stability(
    ordered_samples: list[float],
    confidence: float = DEFAULT_CONFIDENCE,
    *,
    looser_confidence: float = LOOSER_CONFIDENCE,
    tighter_confidence: float = TIGHTER_CONFIDENCE,
) -> ReadingStability:
    """How settled a ratio's `lo > 0` reading is, without a paired sign-test gate.

    A route precision/recall ratio is a bootstrap estimate over one lane, not a paired delta.
    Its lower-bound cut therefore has the same confidence sensitivity and `p_positive` scale as a
    paired interval, but no discordant-pair ledger and no exact sign-test reachability rule.
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
) -> BootstrapRatio:
    """Bootstrap a ratio of counts, such as route precision or recall.

    A zero denominator yields 0.0: a router making no positive prediction has zero measured
    precision, not perfect precision or missing evidence. When a draw exists, the result carries
    an ungated lower-bound stability block derived from those same ratio samples.
    """
    if len(numerators) != len(denominators):
        raise ValueError("ratio needs one denominator flag per numerator flag")

    def ratio(indexes: list[int]) -> float:
        denominator = sum(denominators[i] for i in indexes)
        return sum(numerators[i] for i in indexes) / denominator if denominator else 0.0

    all_indexes = list(range(len(numerators)))
    point = ratio(all_indexes)
    estimate: BootstrapRatio = {"mean": point, "lo": point, "hi": point}
    if not numerators or not index_sets:
        return estimate
    ordered = sorted(ratio(indexes) for indexes in index_sets)
    lo, hi = _ordered_percentiles(ordered, confidence)
    estimate.update({"lo": lo, "hi": hi})
    if brackets(confidence):
        estimate["stability"] = ratio_stability(ordered, confidence)
    return estimate


def format_interval(interval: Interval, places: int = 3) -> str:
    """`0.833 [0.667, 1.000]` -- the one rendering shared by every report table."""
    return (
        f"{interval['mean']:.{places}f} [{interval['lo']:.{places}f}, {interval['hi']:.{places}f}]"
    )
