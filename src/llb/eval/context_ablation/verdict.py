"""Turn the scored lanes into one sentence about whether retrieval pays for itself.

The gate reads the calibrated paired sign-flip p, never the point estimate. The order below is
deliberate --
`long_context_wins` is checked first because a measured long-context gain answers the operator's
question outright ("stuff the document instead"), and it can happen even when the retrieval uplift
over closed-book is itself separable from zero.

The contamination rate is reported with every decision, not folded into it. It changes what a
small uplift MEANS -- items the model already answers were never a retrieval problem -- but the
decision is still about the measured difference.

The `retrieved_document` lane gets its OWN verdict, in `verdict_adoption.py`, rather than a
branch inside this one. The ablation verdict is about what retrieval is worth on this corpus; the
adoption call is about one shippable configuration, and folding a decision an operator acts on
into a diagnostic reading is how a diagnostic quietly becomes a recommendation. Both read the same
calibrated paired cut, and the phrasing helpers below are shared so they cannot drift apart.
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
from llb.rag.fusion_evidence.stability import (
    borderline_note,
)
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE
from llb.rag.fusion_evidence.paired import (
    evidence_gate_clause,
    separates,
)


def by_label_of(derived: Sequence[DerivedComparison]) -> dict[str, DerivedComparison]:
    return {entry["label"]: entry for entry in derived}


def detail(entry: DerivedComparison) -> str:
    delta = entry["paired"]["delta"]
    return (
        f"{entry['label']} {delta['mean']:+.3f} [{delta['lo']:+.3f}, {delta['hi']:+.3f}] "
        f"(n={entry['n']})"
    )


def note_of(*entries: DerivedComparison | None, confidence: float = DEFAULT_CONFIDENCE) -> str:
    """The shared qualifier clauses over the derived deltas this verdict was decided on.

    `no_retrieval_gain` and `rag_pays_off` are two sides of one cut, so both have to say when a
    neighbouring conventional level would have landed on the other side -- and when a lane's delta
    clears zero on too few differing items for the cut to mean anything at all.
    """
    measured = [entry for entry in entries if entry is not None]
    return borderline_note(
        [(entry["label"], entry["paired"].get("stability")) for entry in measured]
    ) + evidence_gate_clause([(entry["label"], entry["paired"]) for entry in measured], confidence)


def long_context_entry(by_label: Mapping[str, DerivedComparison]) -> DerivedComparison | None:
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
    confidence: float = DEFAULT_CONFIDENCE,
) -> ContextAblationVerdict:
    """Name the lane the evidence supports over the whole scored item set."""
    return decide_population(
        derived,
        contamination,
        baseline=baseline,
        n=n,
        skipped={label: len(lane["skipped_item_ids"]) for label, lane in lanes.items()},
        confidence=confidence,
    )


def decide_population(
    derived: Sequence[DerivedComparison],
    contamination: ContaminationReport,
    *,
    baseline: str,
    n: int,
    skipped: Mapping[str, int],
    confidence: float = DEFAULT_CONFIDENCE,
) -> ContextAblationVerdict:
    """Name the lane the evidence supports over ONE population, and what its delta amounts to.

    The population is the whole run for the corpus verdict and one question type for a slice
    reading. Both are judged by the same cut on purpose: a slice whose verdict was reached by a
    softer rule than the pooled one would not be comparable to it, which is the only thing a
    per-slice reading is for.
    """
    verdict: ContextAblationVerdict = {
        "baseline": baseline,
        "n": n,
        "decision": VERDICT_NO_EVIDENCE,
        "reason": "",
        "contamination_rate": contamination["rate"],
        "skipped": dict(skipped),
    }
    by_label = by_label_of(derived)
    uplift = by_label.get(DERIVED_RETRIEVAL_UPLIFT)
    long_context = long_context_entry(by_label)
    if uplift is None:
        verdict["reason"] = (
            "the comparison has no retrieval uplift to state: it needs both the "
            f"{baseline} lane and the rag lane"
        )
        return verdict
    if n == 0:
        verdict["reason"] = "no item was scored"
        return verdict
    decision, reason = _judge(uplift, long_context, contamination, confidence)
    verdict["decision"] = decision
    verdict["reason"] = reason
    return verdict


def _judge(
    uplift: DerivedComparison,
    long_context: DerivedComparison | None,
    contamination: ContaminationReport,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[str, str]:
    """The `(decision, reason)` for one measured ablation."""
    note = (
        f"the closed-book lane already answers {contamination['n_contaminated']}/"
        f"{contamination['n']} items ({contamination['rate']:.0%})"
    )
    cut = note_of(uplift, long_context, confidence=confidence)
    if long_context is not None and separates(long_context["paired"], confidence):
        return VERDICT_LONG_CONTEXT_WINS, (
            f"laying the whole source document into the prompt beats chunked retrieval "
            f"({detail(long_context)}); {note}" + cut
        )
    uplift_delta = uplift["paired"]["delta"]
    if separates(uplift["paired"], confidence):
        return VERDICT_RAG_PAYS_OFF, (
            f"retrieval buys a measured gain over answering from the weights "
            f"({detail(uplift)}); {note}" + cut
        )
    if uplift_delta["mean"] > 0.0:
        return VERDICT_RETRIEVAL_INCONCLUSIVE, (
            f"retrieval gains {uplift_delta['mean']:+.3f} objective but the calibrated test does "
            f"not separate ({detail(uplift)}); a larger scored set is needed to separate the lanes, "
            f"and {note}" + cut
        )
    return VERDICT_NO_RETRIEVAL_GAIN, (
        f"retrieval does not answer better than the model's own weights ({detail(uplift)}); {note}"
        + cut
    )
