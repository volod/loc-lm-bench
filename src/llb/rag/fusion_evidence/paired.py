"""Paired evidence comparisons and their minimum-evidence reading."""

import math
from collections.abc import Iterable, Sequence

from typing_extensions import NotRequired, TypedDict

from llb.rag.fusion_evidence.evidence_gate import (
    READING_FLAT,
    READING_INSUFFICIENT_EVIDENCE,
    READING_SEPARATED,
    evidence_gate_note,
    reaches_reporting_level,
)
from llb.rag.fusion_evidence.stability import ReadingStability, brackets
from llb.rag.fusion_evidence.randomization import (
    paired_randomization,
    randomization_separates,
    seed_from_index_sets,
)
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    Interval,
    _mean,
    _ordered_percentiles,
    bootstrap_samples,
    separation_stability,
)


class PairedComparison(TypedDict):
    """A candidate-minus-baseline delta plus the item-level win/loss/tie ledger."""

    delta: Interval
    wins: int
    losses: int
    ties: int
    sign_test_p: float
    randomization_p: NotRequired[float]
    randomization_method: NotRequired[str]
    randomization_samples: NotRequired[int]
    stability: NotRequired[ReadingStability]


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign-test p-value over the non-tied pairs."""
    decided = wins + losses
    if decided == 0:
        return 1.0
    extreme = min(wins, losses)
    tail = sum(math.comb(decided, i) for i in range(extreme + 1)) / (2.0**decided)
    return min(1.0, 2.0 * tail)


def format_randomization_p(comparison: PairedComparison, places: int = 4) -> str:
    """Artifact cell for the calibrated p, or a dash on an archived uncalibrated block."""
    value = comparison.get("randomization_p")
    return "-" if value is None else f"{value:.{places}f}"


def paired_comparison(
    candidate: list[float],
    baseline: list[float],
    index_sets: list[list[int]],
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedComparison:
    """Bootstrap a paired delta and attach the ledger and stability reading."""
    if len(candidate) != len(baseline):
        raise ValueError("paired comparison needs one baseline value per candidate value")
    deltas = [
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(candidate, baseline)
    ]
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
    randomization = paired_randomization(
        deltas, resamples=len(index_sets), seed=seed_from_index_sets(index_sets)
    )
    comparison["randomization_p"] = randomization["p_value"]
    comparison["randomization_method"] = randomization["method"]
    comparison["randomization_samples"] = randomization["samples"]
    if brackets(confidence):
        comparison["stability"] = separation_stability(
            ordered,
            confidence,
            discordant=wins + losses,
            pairs=len(deltas),
            randomization_p=randomization["p_value"],
            randomization_method=randomization["method"],
            randomization_samples=randomization["samples"],
        )
    return comparison


def discordant_pairs(comparison: PairedComparison) -> int:
    """Items on which the two lanes differ."""
    return comparison["wins"] + comparison["losses"]


def compared_pairs(comparison: PairedComparison) -> int:
    """All paired items, including ties."""
    return comparison["wins"] + comparison["losses"] + comparison["ties"]


def discordant_deltas(deltas: list[float]) -> int:
    """Count differing items directly from a per-item delta vector."""
    return sum(delta != 0.0 for delta in deltas)


def separates(comparison: PairedComparison, confidence: float = DEFAULT_CONFIDENCE) -> bool:
    """Whether the calibrated sign-flip p clears alpha and the claim is reachable."""
    if "randomization_p" in comparison:
        clears = randomization_separates(comparison["randomization_p"], confidence)
    else:
        # Archived blocks have no calibrated p.  Preserve their historical reading until a
        # vector-backed audit can reconstitute it rather than inventing one from aggregates.
        clears = comparison["delta"]["lo"] > 0.0
    return clears and reaches_reporting_level(discordant_pairs(comparison), confidence)


def regresses(comparison: PairedComparison, confidence: float = DEFAULT_CONFIDENCE) -> bool:
    """Whether the BASELINE is ahead by an interval clear of zero, on enough differing items.

    `separates` reads the calibrated sign-flip p, which is one-sided by construction ("candidate
    ahead"), so it can never state a LOSS -- and a lane that BUYS one slice by paying for another
    has to be able to say so. The loss is therefore read off the paired interval, the same
    fallback an uncalibrated archived block gets, mirrored; it carries the same minimum-evidence
    gate, so a loss resting on three differing items is not reported as one either.
    """
    return comparison["delta"]["hi"] < 0.0 and reaches_reporting_level(
        discordant_pairs(comparison), confidence
    )


def reading_of(comparison: PairedComparison, confidence: float = DEFAULT_CONFIDENCE) -> str:
    """Return the separated, insufficient-evidence, or flat reading."""
    calibrated = "randomization_p" in comparison and randomization_separates(
        comparison["randomization_p"], confidence
    )
    if not calibrated and "randomization_p" in comparison:
        if comparison["delta"]["lo"] > 0.0 and not reaches_reporting_level(
            discordant_pairs(comparison), confidence
        ):
            return READING_INSUFFICIENT_EVIDENCE
        return READING_FLAT
    if "randomization_p" not in comparison and comparison["delta"]["lo"] <= 0.0:
        return READING_FLAT
    return READING_SEPARATED if separates(comparison, confidence) else READING_INSUFFICIENT_EVIDENCE


def gated_readings(
    comparisons: Iterable[PairedComparison], confidence: float = DEFAULT_CONFIDENCE
) -> tuple[int, int]:
    """Return ``(relabeled, total)`` over a report's paired blocks."""
    readings = [reading_of(comparison, confidence) for comparison in comparisons]
    return sum(reading == READING_INSUFFICIENT_EVIDENCE for reading in readings), len(readings)


def evidence_gate_clause(
    rows: Sequence[tuple[str, PairedComparison]], confidence: float = DEFAULT_CONFIDENCE
) -> str:
    """Format the insufficient-evidence clause over verdict-driving rows."""
    return evidence_gate_note(
        [
            (label, discordant_pairs(comparison), compared_pairs(comparison))
            for label, comparison in rows
            if reading_of(comparison, confidence) == READING_INSUFFICIENT_EVIDENCE
        ],
        confidence,
    )
