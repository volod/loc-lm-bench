"""Adopt-or-retain: the one sentence the bake-off's operator acts on, and why it says that.

Split out of `embedding_bakeoff_uncertainty.py` so the paired STATISTICS and the decision they
feed stay separately readable -- the same seam `embedder_adoption/verdict.py` uses.

The decision is on the calibrated paired reading and the selected candidate family, never the point
estimate: a candidate that merely leads on the mean is exactly the case this lane exists to refuse.
What the reason ADDS is how close the row it names sits to those cuts, so a `retain` reached because
every candidate missed by a mile no longer prints identically to one where a candidate sat on the
line.

Pure: the input is finished `PairedRow`s, so the whole decision is unit-tested with plain vectors.
"""

from collections.abc import Sequence

from llb.rag.embedding_bakeoff_uncertainty import (
    BAR_RECALL,
    BARS,
    DEFAULT_BARS,
    METRIC_MRR,
    METRIC_RECALL,
    BakeoffVerdict,
    PairedRow,
    bar_stability,
)
from llb.rag.fusion_evidence.stability import (
    borderline_note,
    unsettled,
)
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    format_interval,
)
from llb.rag.fusion_evidence.paired import (
    evidence_gate_clause,
    separates,
)
from llb.rag.embedding_bakeoff_selection import hypothesis_key, selection_note
from llb.rag.fusion_evidence.selection import (
    SelectionAdjustment,
    selection_separates,
)

DECISION_ADOPT = "adopt"
DECISION_RETAIN = "retain"
DECISION_UNDECIDED = "undecided"


def resolve_bars(spec: str | None) -> tuple[str, ...]:
    """Parse a comma-separated adoption-bar selection; empty/None keeps the recall@k-only default.

    `recall_at_k` is always kept: the second bar EXTENDS the decision, it never replaces the one
    unconditional reason to swap an encoder.
    """
    names = [token.strip() for token in (spec or "").split(",") if token.strip()]
    if not names:
        return DEFAULT_BARS
    unknown = [name for name in names if name not in BARS]
    if unknown:
        raise ValueError(
            f"unknown adoption bar(s) {', '.join(unknown)}: expected any of {', '.join(BARS)}"
        )
    selected = dict.fromkeys([BAR_RECALL, *names])
    return tuple(bar for bar in BARS if bar in selected)


def cleared_bars(
    paired: PairedRow,
    bars: Sequence[str] = DEFAULT_BARS,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[str]:
    """The enabled bars this candidate separates on, in `BARS` order.

    "Separates" is the shared calibrated sign-flip test plus the minimum-evidence gate.
    """
    return [bar for bar in BARS if bar in bars and separates(paired["metrics"][bar], confidence)]


def separates_from_baseline(
    paired: PairedRow,
    bars: Sequence[str] = DEFAULT_BARS,
    confidence: float = DEFAULT_CONFIDENCE,
) -> bool:
    """True when the candidate clears at least one ENABLED adoption bar."""
    return bool(cleared_bars(paired, bars, confidence))


def borderline_bars(paired: PairedRow, bars: Sequence[str] = DEFAULT_BARS) -> list[str]:
    """The enabled bars whose reading a neighbouring conventional level would change.

    Read on BOTH sides: a cleared bar that a tighter level would drop is a positive resting on the
    convention, and a missed bar that a looser level would clear is the near-miss `retain` this
    lane used to state in exactly the same words as a candidate that lost by a mile.
    """
    return [
        bar for bar in BARS if bar in bars and unsettled(bar_stability(paired, bar)) is not None
    ]


def decide_verdict(
    paired: dict[str, PairedRow],
    baseline: str | None,
    bars: Sequence[str] = DEFAULT_BARS,
    confidence: float = DEFAULT_CONFIDENCE,
    adjustment: SelectionAdjustment | None = None,
) -> BakeoffVerdict:
    """Adopt the best separated candidate, else retain the incumbent (never rank on a point gap).

    "Separated" is deliberately the strict calibrated randomization reading after the selected
    candidate x bar family is adjusted. A point-estimate leader is exactly the case this lane
    exists to refuse.

    `bars` defaults to recall@k alone. Adding `BAR_FIRST_HIT` opts the run into the scoped
    first-hit-rank bar, which an operator enables when their retrieval configuration makes rank
    binding rather than as a general widening of the recommendation.

    Either way the reason QUALIFIES the row it names when a neighbouring conventional confidence
    level would read that row differently -- so a `retain` reached because every candidate missed
    by a mile no longer prints identically to one where a candidate sat on the cut.
    """
    if baseline is None or not paired:
        return _verdict(
            DECISION_UNDECIDED,
            None,
            baseline,
            bars,
            reason=(
                "the baseline embedder was not scored in this run, so no paired delta is defined"
            ),
        )
    borderline = {
        model: marked
        for model, row in sorted(paired.items())
        if model != baseline and (marked := borderline_bars(row, bars))
    }
    per_row_cleared = {
        model: cleared_bars(row, bars, confidence)
        for model, row in paired.items()
        if model != baseline and separates_from_baseline(row, bars, confidence)
    }
    cleared = {
        model: [
            bar
            for bar in marked
            if adjustment is None
            or selection_separates(adjustment, hypothesis_key(model, bar), confidence)
        ]
        for model, marked in per_row_cleared.items()
    }
    cleared = {model: marked for model, marked in cleared.items() if marked}
    if not cleared:
        return _verdict(
            DECISION_RETAIN,
            baseline,
            baseline,
            bars,
            borderline=borderline,
            per_row_cleared=per_row_cleared,
            adjustment=adjustment,
            reason=(
                (
                    "no candidate survives the selection-adjusted adoption family "
                    if per_row_cleared
                    else "no candidate clears an adoption bar "
                )
                + f"({', '.join(bars)}) against `{baseline}`, "
                "so the ranking is not supported by this item set"
            )
            + _near_miss_note(paired, borderline)
            + _gate_note(paired, baseline, bars, confidence)
            + selection_note(adjustment),
        )
    separated = sorted(cleared, key=lambda model: _rank_key(paired[model], cleared[model]))
    winner = separated[0]
    return _verdict(
        DECISION_ADOPT,
        winner,
        baseline,
        bars,
        separated=separated,
        cleared=cleared,
        borderline=borderline,
        per_row_cleared=per_row_cleared,
        adjustment=adjustment,
        reason=_adopt_reason(winner, baseline, paired[winner], cleared[winner])
        + borderline_note(
            [(f"{winner} {bar}", bar_stability(paired[winner], bar)) for bar in cleared[winner]]
        )
        + selection_note(adjustment),
    )


def _gate_note(
    paired: dict[str, PairedRow],
    baseline: str,
    bars: Sequence[str],
    confidence: float = DEFAULT_CONFIDENCE,
) -> str:
    """The clause that tells a `retain` reached on thin evidence from one reached on wide evidence.

    Without it a candidate that leads on two of forty questions and a candidate that is genuinely
    level with the incumbent produce the same sentence.
    """
    return evidence_gate_clause(
        [
            (f"{model} {bar}", row["metrics"][bar])
            for model, row in sorted(paired.items())
            if model != baseline
            for bar in BARS
            if bar in bars
        ],
        confidence,
    )


def _near_miss_note(paired: dict[str, PairedRow], borderline: dict[str, list[str]]) -> str:
    """The clause that tells a settled `retain` apart from one a looser convention would overturn."""
    return borderline_note(
        [
            (f"{model} {bar}", bar_stability(paired[model], bar))
            for model, marked in borderline.items()
            for bar in marked
        ]
    )


def _rank_key(paired: PairedRow, cleared: Sequence[str]) -> tuple[int, float, float]:
    """Most bars cleared wins; then the larger recall gain, then the larger first-hit gain.

    Bar COUNT leads because clearing both bars is strictly more evidence than clearing either, and
    recall@k breaks the tie because it is the bar that holds in every configuration.
    """
    return (
        -len(cleared),
        -paired["metrics"][METRIC_RECALL]["delta"]["mean"],
        -paired["metrics"][METRIC_MRR]["delta"]["mean"],
    )


def _adopt_reason(winner: str, baseline: str, paired: PairedRow, cleared: Sequence[str]) -> str:
    """Name the bar(s) the winner cleared and quote each one's interval and ledger."""
    detail = "; ".join(
        f"{bar} delta {format_interval(paired['metrics'][bar]['delta'])}, "
        f"{paired['metrics'][bar]['wins']}/{paired['metrics'][bar]['losses']}/"
        f"{paired['metrics'][bar]['ties']} win/loss/tie, "
        f"sign-test p={paired['metrics'][bar]['sign_test_p']:.3f}"
        for bar in cleared
    )
    return f"`{winner}` clears {', '.join(cleared)} against `{baseline}`: {detail}"


def _verdict(
    decision: str,
    model: str | None,
    baseline: str | None,
    bars: Sequence[str],
    *,
    separated: list[str] | None = None,
    cleared: dict[str, list[str]] | None = None,
    borderline: dict[str, list[str]] | None = None,
    per_row_cleared: dict[str, list[str]] | None = None,
    adjustment: SelectionAdjustment | None = None,
    reason: str,
) -> BakeoffVerdict:
    verdict: BakeoffVerdict = {
        "decision": decision,
        "model": model,
        "baseline": baseline,
        "separated": list(separated or []),
        "bars": list(bars),
        "cleared": dict(cleared or {}),
        "borderline": dict(borderline or {}),
        "reason": reason,
    }
    if per_row_cleared is not None:
        verdict["per_row_cleared"] = per_row_cleared
    if adjustment is not None:
        verdict["selection_adjustment"] = adjustment
    return verdict
