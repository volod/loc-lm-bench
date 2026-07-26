"""Paired uncertainty for signed query-noise deltas and mitigation recovery.

The generic paired comparison asks only whether a candidate is ahead. Query robustness needs the
other direction too: noise can improve, degrade, or leave a lane indistinguishable from its clean
baseline. This module reads those three states at the reporting confidence and its two neighbouring
conventions from one shared bootstrap draw, then persists the ordinary `PairedComparison` shape.

The minimum-evidence rule still applies to either directional claim. A positive or negative
interval supported by too few changed item outcomes is an open question, not an improvement or a
degradation. The point estimate, interval, ledger, and exact sign-test p remain untouched.
"""

from typing import Any

from llb.rag.fusion_evidence.evidence_gate import (
    READING_INSUFFICIENT_EVIDENCE,
    reaches_reporting_level,
)
from llb.rag.fusion_evidence.stability import (
    LOOSER_CONFIDENCE,
    TIGHTER_CONFIDENCE,
    brackets,
    exceedance,
    stability_from_readings,
)
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    PairedComparison,
    bootstrap_samples,
    interval_from_ordered_samples,
    paired_comparison,
)

READING_IMPROVED = "improved"
READING_DEGRADED = "degraded"
READING_INDISTINGUISHABLE = "indistinguishable"

OBJECTIVE_DELTA = "objective_delta"
RECALL_DELTA = "recall_delta"
OBJECTIVE_RECOVERY = "objective_recovery"
RECALL_RECOVERY = "recall_recovery"


def _reading(
    deltas: list[float],
    ordered_samples: list[float],
    confidence: float,
    discordant: int,
) -> str:
    interval = interval_from_ordered_samples(deltas, ordered_samples, confidence)
    if interval["lo"] > 0.0:
        claim = READING_IMPROVED
    elif interval["hi"] < 0.0:
        claim = READING_DEGRADED
    else:
        return READING_INDISTINGUISHABLE
    return (
        claim if reaches_reporting_level(discordant, confidence) else READING_INSUFFICIENT_EVIDENCE
    )


def directional_comparison(
    candidate: list[float],
    baseline: list[float],
    index_sets: list[list[int]],
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedComparison:
    """A paired delta with a three-state, two-sided stability annotation.

    `p_positive` remains the share of resamples where candidate minus baseline is above zero.
    Consequently a decisive degradation sits near zero, a decisive improvement near one, and an
    indistinguishable row near the middle. The neighbouring readings make either knife edge
    explicit without changing the interval or the lane's existing aggregate delta.
    """
    comparison = paired_comparison(candidate, baseline, [])
    deltas = [value - reference for value, reference in zip(candidate, baseline)]
    if not deltas or not index_sets:
        return comparison
    ordered = sorted(bootstrap_samples(deltas, index_sets))
    comparison["delta"] = interval_from_ordered_samples(deltas, ordered, confidence)
    if not brackets(confidence):
        return comparison
    discordant = comparison["wins"] + comparison["losses"]
    comparison["stability"] = stability_from_readings(
        reading=_reading(deltas, ordered, confidence, discordant),
        looser_reading=_reading(deltas, ordered, LOOSER_CONFIDENCE, discordant),
        tighter_reading=_reading(deltas, ordered, TIGHTER_CONFIDENCE, discordant),
        p_positive=exceedance(ordered),
        discordant=discordant,
        pairs=len(deltas),
    )
    return comparison


def delta_comparisons(
    rows: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    index_sets: list[list[int]],
    confidence: float,
) -> dict[str, PairedComparison]:
    """Objective and retrieval comparisons for aligned candidate/baseline case rows."""
    if len(rows) != len(baselines):
        raise ValueError("query robustness uncertainty needs aligned candidate and baseline rows")
    if [str(row["item_id"]) for row in rows] != [str(row["item_id"]) for row in baselines]:
        raise ValueError("query robustness uncertainty needs the same item order in both lanes")
    return {
        OBJECTIVE_DELTA: directional_comparison(
            [float(row["objective_score"]) for row in rows],
            [float(row["objective_score"]) for row in baselines],
            index_sets,
            confidence,
        ),
        RECALL_DELTA: directional_comparison(
            [float(row["retrieval_hit"]) for row in rows],
            [float(row["retrieval_hit"]) for row in baselines],
            index_sets,
            confidence,
        ),
    }


def recovery_comparisons(
    rows: list[dict[str, Any]],
    unmitigated: list[dict[str, Any]],
    index_sets: list[list[int]],
    confidence: float,
) -> dict[str, PairedComparison]:
    """Objective and retrieval recovery against the same class's unmitigated row."""
    comparisons = delta_comparisons(rows, unmitigated, index_sets, confidence)
    return {
        OBJECTIVE_RECOVERY: comparisons[OBJECTIVE_DELTA],
        RECALL_RECOVERY: comparisons[RECALL_DELTA],
    }
