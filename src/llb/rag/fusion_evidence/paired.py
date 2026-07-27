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
    stability: NotRequired[ReadingStability]


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign-test p-value over the non-tied pairs."""
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
    if brackets(confidence):
        comparison["stability"] = separation_stability(
            ordered, confidence, discordant=wins + losses, pairs=len(deltas)
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
    """Whether the interval clears zero and the sign test can reach this level."""
    return comparison["delta"]["lo"] > 0.0 and reaches_reporting_level(
        discordant_pairs(comparison), confidence
    )


def reading_of(comparison: PairedComparison, confidence: float = DEFAULT_CONFIDENCE) -> str:
    """Return the separated, insufficient-evidence, or flat reading."""
    if comparison["delta"]["lo"] <= 0.0:
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
            if comparison["delta"]["lo"] > 0.0
        ],
        confidence,
    )
