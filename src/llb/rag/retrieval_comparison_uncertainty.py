"""Paired uncertainty and the adopt-or-retain call for ``compare-retrieval``.

The comparison retrieves every lane over the same item order.  This module turns those aligned
metric vectors into the same shared-draw paired evidence used by the embedder bake-off, then asks
the narrower operational question: does the point-estimate winner actually separate from the
named baseline?

Recall is the primary gate.  MRR may decide the recommendation only when recall is identical on
every paired item; that prevents an earlier first hit from hiding an unresolved recall tradeoff.
Diagnostic rows are filtered by the caller before the winner is selected.
"""

from collections.abc import Sequence

from typing_extensions import NotRequired, TypedDict

from llb.rag.embedding_bakeoff_selection import (
    adjust_bakeoff_selection,
    hypothesis_key,
)
from llb.rag.embedding_bakeoff_uncertainty import (
    BAR_FIRST_HIT,
    BAR_RECALL,
    MetricVectors,
    PairedRow,
)
from llb.rag.fusion_evidence.paired import PairedComparison, reading_of, separates
from llb.rag.fusion_evidence.selection import SelectionAdjustment, selection_separates
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, format_interval

DECISION_ADOPT = "adopt"
DECISION_RETAIN = "retain"
DECISION_UNDECIDED = "undecided"
VERDICT_BARS = (BAR_RECALL, BAR_FIRST_HIT)


class RetrievalComparisonVerdict(TypedDict):
    """Recommendation for the best deployable lane against the declared baseline."""

    decision: str
    lane: str | None
    baseline: str | None
    reason: str
    selection_adjustment: NotRequired[SelectionAdjustment]


def selection_adjustment(
    vectors: dict[str, MetricVectors],
    baseline: str | None,
    eligible_lanes: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> SelectionAdjustment | None:
    """Adjust the deployable lane x metric family from which the winner was selected."""
    eligible = {
        lane: vectors[lane] for lane in eligible_lanes if lane in vectors or lane == baseline
    }
    if baseline in vectors:
        eligible[baseline] = vectors[baseline]
    return adjust_bakeoff_selection(
        eligible,
        baseline,
        VERDICT_BARS,
        resamples=resamples,
        seed=seed,
    )


def decide_verdict(
    paired: dict[str, PairedRow],
    *,
    baseline: str | None,
    winner: str | None,
    confidence: float = DEFAULT_CONFIDENCE,
    adjustment: SelectionAdjustment | None = None,
) -> RetrievalComparisonVerdict:
    """Adopt a separated point winner, otherwise retain the baseline.

    The winner is selected outside this function by the comparison's long-standing ordering:
    recall@k, then MRR, then label.  Keeping selection and inference separate makes it impossible
    for uncertainty to silently change the published point estimates.
    """
    if baseline is None or winner is None or baseline not in paired:
        return _verdict(
            DECISION_UNDECIDED,
            winner,
            baseline,
            "no scored baseline lane is available, so no paired recommendation is defined",
            adjustment,
        )
    if winner == baseline:
        return _verdict(
            DECISION_RETAIN,
            baseline,
            baseline,
            f"`{baseline}` remains the point-estimate leader among deployable lanes",
            adjustment,
        )
    row = paired[winner]
    recall = row["metrics"][BAR_RECALL]
    mrr = row["metrics"][BAR_FIRST_HIT]
    recall_separates = _selection_separates(winner, BAR_RECALL, recall, confidence, adjustment)
    recall_identical = recall["wins"] + recall["losses"] == 0
    mrr_separates = _selection_separates(winner, BAR_FIRST_HIT, mrr, confidence, adjustment)
    if recall_separates:
        reason = (
            f"`{winner}` separates from `{baseline}` on recall_at_k: {_comparison_detail(recall)}"
        )
        return _verdict(DECISION_ADOPT, winner, baseline, reason, adjustment)
    if recall_identical and mrr_separates:
        reason = (
            f"`{winner}` has itemwise-identical recall_at_k and separates from `{baseline}` "
            f"on mrr: {_comparison_detail(mrr)}"
        )
        return _verdict(DECISION_ADOPT, winner, baseline, reason, adjustment)
    reason = (
        f"`{winner}` is the point-estimate leader but does not establish a deployable gain over "
        f"`{baseline}`; recall_at_k is {_reading(recall, confidence)} "
        f"({_comparison_detail(recall)})"
    )
    if recall_identical:
        reason += f" and mrr is {_reading(mrr, confidence)} ({_comparison_detail(mrr)})"
    else:
        reason += "; MRR cannot override a non-identical unresolved recall ledger"
    return _verdict(DECISION_RETAIN, baseline, baseline, reason, adjustment)


def _selection_separates(
    lane: str,
    metric: str,
    comparison: PairedComparison,
    confidence: float,
    adjustment: SelectionAdjustment | None,
) -> bool:
    """Positive raw separation which also survives the selected family, when present."""
    return (
        "randomization_p" in comparison
        and comparison["delta"]["mean"] > 0.0
        and separates(comparison, confidence)
        and (
            adjustment is None
            or selection_separates(adjustment, hypothesis_key(lane, metric), confidence)
        )
    )


def _reading(comparison: PairedComparison, confidence: float) -> str:
    """A calibrated reading, or an explicit marker when resampling was disabled."""
    return reading_of(comparison, confidence) if "randomization_p" in comparison else "unmeasured"


def _comparison_detail(comparison: PairedComparison) -> str:
    return (
        f"delta {format_interval(comparison['delta'])}, "
        f"{comparison['wins']}/{comparison['losses']}/{comparison['ties']} win/loss/tie"
    )


def _verdict(
    decision: str,
    lane: str | None,
    baseline: str | None,
    reason: str,
    adjustment: SelectionAdjustment | None,
) -> RetrievalComparisonVerdict:
    verdict: RetrievalComparisonVerdict = {
        "decision": decision,
        "lane": lane,
        "baseline": baseline,
        "reason": reason,
    }
    if adjustment is not None:
        verdict["selection_adjustment"] = adjustment
    return verdict
