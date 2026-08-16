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
    BakeoffVerdict,
    PairedRow,
    bar_stability,
)
from llb.rag.fusion_evidence.stability import (
    borderline_note,
    unsettled,
)
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE
from llb.rag.fusion_evidence.paired import (
    separates,
)
from llb.rag.embedding_bakeoff_reason import (
    adopt_reason,
    build_verdict,
    gate_note,
    near_miss_note,
    rank_key,
)
from llb.rag.embedding_bakeoff_selection import hypothesis_key, selection_note
from llb.rag.fusion_evidence.selection import (
    SelectionAdjustment,
    selection_separates,
)

DECISION_ADOPT = "adopt"
DECISION_RETAIN = "retain"
DECISION_UNDECIDED = "undecided"


def resolve_bars(spec: str | None, default: Sequence[str] = DEFAULT_BARS) -> tuple[str, ...]:
    """Parse a comma-separated adoption-bar selection; empty/None keeps the lane's default.

    `recall_at_k` is always kept: the second bar EXTENDS the decision, it never replaces the one
    unconditional reason to swap a component. `default` is the lane's own starting selection --
    recall@k alone for the embedder lane, both bars for the reranker lane, where rank is the
    quantity a cross-encoder can actually move.
    """
    names = [token.strip() for token in (spec or "").split(",") if token.strip()]
    if not names:
        return tuple(default)
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


def _cleared_bars_by_model(
    paired: dict[str, PairedRow],
    baseline: str,
    bars: Sequence[str],
    confidence: float,
    adjustment: SelectionAdjustment | None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Which candidates clear which bars, before and after the family-wise selection adjustment.

    Both are needed by the reason: a candidate that cleared on its own row but not in the family
    reads differently from one that never cleared at all.
    """
    per_row = {
        model: cleared_bars(row, bars, confidence)
        for model, row in paired.items()
        if model != baseline and separates_from_baseline(row, bars, confidence)
    }
    adjusted = {
        model: [
            bar
            for bar in marked
            if adjustment is None
            or selection_separates(adjustment, hypothesis_key(model, bar), confidence)
        ]
        for model, marked in per_row.items()
    }
    return per_row, {model: marked for model, marked in adjusted.items() if marked}


def _retain_reason(
    per_row_cleared: dict[str, list[str]], baseline: str, bars: Sequence[str]
) -> str:
    """Why the incumbent stands: nothing cleared, or nothing SURVIVED the adjustment."""
    opening = (
        "no candidate survives the selection-adjusted adoption family "
        if per_row_cleared
        else "no candidate clears an adoption bar "
    )
    return (
        opening + f"({', '.join(bars)}) against `{baseline}`, "
        "so the ranking is not supported by this item set"
    )


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
        return build_verdict(
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
    per_row_cleared, cleared = _cleared_bars_by_model(
        paired, baseline, bars, confidence, adjustment
    )
    if not cleared:
        return build_verdict(
            DECISION_RETAIN,
            baseline,
            baseline,
            bars,
            borderline=borderline,
            per_row_cleared=per_row_cleared,
            adjustment=adjustment,
            reason=_retain_reason(per_row_cleared, baseline, bars)
            + near_miss_note(paired, borderline)
            + gate_note(paired, baseline, bars, confidence)
            + selection_note(adjustment),
        )
    separated = sorted(cleared, key=lambda model: rank_key(paired[model], cleared[model]))
    winner = separated[0]
    return build_verdict(
        DECISION_ADOPT,
        winner,
        baseline,
        bars,
        separated=separated,
        cleared=cleared,
        borderline=borderline,
        per_row_cleared=per_row_cleared,
        adjustment=adjustment,
        reason=adopt_reason(winner, baseline, paired[winner], cleared[winner])
        + borderline_note(
            [(f"{winner} {bar}", bar_stability(paired[winner], bar)) for bar in cleared[winner]]
        )
        + selection_note(adjustment),
    )
