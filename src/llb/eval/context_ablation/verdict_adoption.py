"""Adopt or reject the one context lane an operator could actually ship.

`retrieved_document` is not a diagnostic: it retrieves exactly as the leaderboard lane does and
only widens the unit of context from the top-ranked chunk to its whole document, so a measured
gain here is a gain an operator captures by changing a config value. That is why it gets an
explicit adopt-or-reject call instead of a number to interpret, and why the call lives beside the
ablation verdict rather than inside it.

The cut is the shared calibrated paired interval (`llb.eval.context_ablation.verdict`), so this
module owns the DECISION and none of the statistics.
"""

from collections.abc import Mapping, Sequence

from llb.eval.context_ablation.models import (
    ADOPT_RETRIEVED_DOCUMENT,
    DERIVED_RETRIEVED_DOCUMENT_DELTA,
    DERIVED_RETRIEVED_DOCUMENT_DELTA_FITTING,
    LANE_RETRIEVED_DOCUMENT,
    REJECT_RETRIEVED_DOCUMENT,
    RETRIEVED_DOCUMENT_INCONCLUSIVE,
    RETRIEVED_DOCUMENT_NOT_MEASURED,
    DerivedComparison,
    LaneReport,
    RetrievedDocumentVerdict,
)
from llb.eval.context_ablation.verdict import (
    by_label_of,
    detail,
    long_context_entry,
    note_of,
)
from llb.rag.fusion_evidence.paired import regresses, separates
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE


def _retrieved_document_entry(
    by_label: Mapping[str, DerivedComparison],
) -> DerivedComparison | None:
    """The delta the adoption call reads: the fitting cut when either lane skipped an item."""
    return by_label.get(DERIVED_RETRIEVED_DOCUMENT_DELTA_FITTING) or by_label.get(
        DERIVED_RETRIEVED_DOCUMENT_DELTA
    )


def _captured_share(
    retrieved: DerivedComparison, long_context: DerivedComparison | None
) -> float | None:
    """The fraction of the oracle document gain this lane captured with no gold label.

    Only defined when the oracle lane actually gained: dividing by a zero or negative long-context
    delta would turn "there was nothing to capture" into a ratio nobody can read.
    """
    if long_context is None:
        return None
    oracle = long_context["paired"]["delta"]["mean"]
    if oracle <= 0.0:
        return None
    return round(retrieved["paired"]["delta"]["mean"] / oracle, 4)


def decide_retrieved_document(
    derived: Sequence[DerivedComparison],
    lanes: Mapping[str, LaneReport],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> RetrievedDocumentVerdict:
    """Adopt or reject "retrieve the chunk, send the document" as a shippable configuration.

    This lane is the only one in the ablation an operator could actually turn on, so it gets an
    explicit call rather than a number to interpret. The cut is the same calibrated paired
    interval every other verdict here reads -- adopt only on a gain that separates from zero,
    reject only on a loss that does, and say so plainly when the interval straddles it. The share
    of the oracle gap it captured is reported with the call, never folded into it: a lane can
    capture most of a gain that was not worth having.
    """
    by_label = by_label_of(derived)
    retrieved = _retrieved_document_entry(by_label)
    lane = lanes.get(LANE_RETRIEVED_DOCUMENT)
    skipped = len(lane["skipped_item_ids"]) if lane is not None else 0
    if retrieved is None:
        return {
            "decision": RETRIEVED_DOCUMENT_NOT_MEASURED,
            "reason": (
                f"the comparison did not score the {LANE_RETRIEVED_DOCUMENT} lane against rag, "
                "so there is nothing to adopt or reject"
            ),
            "n": 0,
            "delta": 0.0,
            "captured_share": None,
            "skipped": skipped,
        }
    long_context = long_context_entry(by_label)
    share = _captured_share(retrieved, long_context)
    delta = retrieved["paired"]["delta"]
    verdict: RetrievedDocumentVerdict = {
        "decision": RETRIEVED_DOCUMENT_INCONCLUSIVE,
        "reason": "",
        "n": retrieved["n"],
        "delta": delta["mean"],
        "captured_share": share,
        "skipped": skipped,
    }
    share_note = (
        f"; that is {share:.0%} of the oracle long-context gain ({detail(long_context)})"
        if share is not None and long_context is not None
        else ""
    )
    cut = note_of(retrieved, confidence=confidence)
    # `separates` is one-sided by construction ("candidate ahead"), so a measured LOSS is read off
    # the mirrored interval gate -- an adopt-or-reject call that could only ever say "adopt or
    # unclear" would not be one.
    if separates(retrieved["paired"], confidence):
        verdict["decision"] = ADOPT_RETRIEVED_DOCUMENT
        verdict["reason"] = (
            "sending the whole document the top-ranked chunk came from beats chunked retrieval "
            f"with no gold label anywhere ({detail(retrieved)}){share_note}" + cut
        )
        return verdict
    if regresses(retrieved["paired"], confidence):
        verdict["decision"] = REJECT_RETRIEVED_DOCUMENT
        verdict["reason"] = (
            "widening the unit of retrieval from chunk to document measurably HURTS "
            f"({detail(retrieved)}); keep the chunked configuration" + cut
        )
        return verdict
    verdict["reason"] = (
        f"the calibrated test does not separate this lane from chunked retrieval "
        f"({detail(retrieved)}){share_note}; keep the chunked configuration until a larger "
        "scored set separates them" + cut
    )
    return verdict


__all__ = ["decide_retrieved_document"]
