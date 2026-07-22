"""Turn the three lanes into one sentence about whether retrieval pays for itself.

The gate reads the paired INTERVAL, never the point estimate: on a few dozen items a positive mean
whose interval includes no difference is not a finding. The order below is deliberate --
`long_context_wins` is checked first because a measured long-context gain answers the operator's
question outright ("stuff the document instead"), and it can happen even when the retrieval uplift
over closed-book is itself separable from zero.

The contamination rate is reported with every decision, not folded into it. It changes what a
small uplift MEANS -- items the model already answers were never a retrieval problem -- but the
decision is still about the measured difference.
"""

from collections.abc import Mapping, Sequence

from llb.eval.context_ablation.models import (
    DERIVED_LONG_CONTEXT_DELTA,
    DERIVED_LONG_CONTEXT_DELTA_FITTING,
    DERIVED_RETRIEVAL_UPLIFT,
    ContaminationReport,
    ContextAblationVerdict,
    DerivedComparison,
    LaneReport,
    VERDICT_LONG_CONTEXT_WINS,
    VERDICT_NO_EVIDENCE,
    VERDICT_NO_RETRIEVAL_GAIN,
    VERDICT_RAG_PAYS_OFF,
    VERDICT_RETRIEVAL_INCONCLUSIVE,
)


def _by_label(derived: Sequence[DerivedComparison]) -> dict[str, DerivedComparison]:
    return {entry["label"]: entry for entry in derived}


def _detail(entry: DerivedComparison) -> str:
    delta = entry["paired"]["delta"]
    return (
        f"{entry['label']} {delta['mean']:+.3f} [{delta['lo']:+.3f}, {delta['hi']:+.3f}] "
        f"(n={entry['n']})"
    )


def _long_context_entry(by_label: Mapping[str, DerivedComparison]) -> DerivedComparison | None:
    """The long-context delta the verdict reads: the fitting subset when items were skipped.

    A skipped item scores zero, so including it would read a document that never reached the model
    as a long-context loss. The all-items delta stays in the report; the DECISION uses the
    population where the lane was actually applicable.
    """
    return by_label.get(DERIVED_LONG_CONTEXT_DELTA_FITTING) or by_label.get(
        DERIVED_LONG_CONTEXT_DELTA
    )


def decide(
    lanes: Mapping[str, LaneReport],
    derived: Sequence[DerivedComparison],
    contamination: ContaminationReport,
    *,
    baseline: str,
    n: int,
) -> ContextAblationVerdict:
    """Name the lane the evidence supports, and say what its delta amounts to."""
    verdict: ContextAblationVerdict = {
        "baseline": baseline,
        "n": n,
        "decision": VERDICT_NO_EVIDENCE,
        "reason": "",
        "contamination_rate": contamination["rate"],
        "skipped": {label: len(lane["skipped_item_ids"]) for label, lane in lanes.items()},
    }
    by_label = _by_label(derived)
    uplift = by_label.get(DERIVED_RETRIEVAL_UPLIFT)
    long_context = _long_context_entry(by_label)
    if uplift is None:
        verdict["reason"] = (
            "the comparison has no retrieval uplift to state: it needs both the "
            f"{baseline} lane and the rag lane"
        )
        return verdict
    if n == 0:
        verdict["reason"] = "no item was scored"
        return verdict
    decision, reason = _judge(uplift, long_context, contamination)
    verdict["decision"] = decision
    verdict["reason"] = reason
    return verdict


def _judge(
    uplift: DerivedComparison,
    long_context: DerivedComparison | None,
    contamination: ContaminationReport,
) -> tuple[str, str]:
    """The `(decision, reason)` for one measured ablation."""
    note = (
        f"the closed-book lane already answers {contamination['n_contaminated']}/"
        f"{contamination['n']} items ({contamination['rate']:.0%})"
    )
    if long_context is not None and long_context["paired"]["delta"]["lo"] > 0.0:
        return VERDICT_LONG_CONTEXT_WINS, (
            f"laying the whole source document into the prompt beats chunked retrieval "
            f"({_detail(long_context)}); {note}"
        )
    uplift_delta = uplift["paired"]["delta"]
    if uplift_delta["lo"] > 0.0:
        return VERDICT_RAG_PAYS_OFF, (
            f"retrieval buys a measured gain over answering from the weights "
            f"({_detail(uplift)}); {note}"
        )
    if uplift_delta["mean"] > 0.0:
        return VERDICT_RETRIEVAL_INCONCLUSIVE, (
            f"retrieval gains {uplift_delta['mean']:+.3f} objective but the interval includes no "
            f"difference ({_detail(uplift)}); a larger scored set is needed to separate the lanes, "
            f"and {note}"
        )
    return VERDICT_NO_RETRIEVAL_GAIN, (
        f"retrieval does not answer better than the model's own weights ({_detail(uplift)}); {note}"
    )
