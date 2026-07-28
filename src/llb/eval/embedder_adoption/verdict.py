"""The keep-or-extend call on the bake-off's adoption bar, and the borderline qualifier on it.

Split out of `compare.py` so the per-cell STATISTICS and the one sentence an operator acts on stay
separately readable -- the same seam `report.py` / `roster_report.py` use for the rendering side.

Pure: the input is finished `CellReport`s, so the whole decision is unit-tested with dict cells.
"""

from collections.abc import Sequence

from llb.eval.embedder_adoption.models import (
    DECISION_EXTEND_BAR,
    DECISION_KEEP_BAR,
    DECISION_NO_EVIDENCE,
    METRIC_OBJECTIVE,
    METRIC_RECIPROCAL_RANK,
    BarVerdict,
    CellReport,
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
from llb.rag.fusion_evidence.randomization import seed_from_index_sets
from llb.rag.fusion_evidence.selection import (
    DEFAULT_SELECTION_RESAMPLES,
    SelectionAdjustment,
    selection_adjustment,
    selection_separates,
)


def adjust_selection_family(
    deltas: dict[str, list[float]],
    *,
    resamples: int,
    index_sets: list[list[int]],
) -> SelectionAdjustment | None:
    """Adjust the objective cells the `extend_bar` verdict can select from."""
    if not deltas or not index_sets:
        return None
    return selection_adjustment(
        dict(sorted(deltas.items())),
        resamples=max(resamples, DEFAULT_SELECTION_RESAMPLES),
        seed=seed_from_index_sets(index_sets),
    )


def _separated(
    cells: Sequence[CellReport], metric: str, confidence: float = DEFAULT_CONFIDENCE
) -> list[str]:
    """Cell labels whose paired delta on `metric` separates from zero on enough differing items."""
    return [cell["label"] for cell in cells if separates(cell["paired"][metric], confidence)]


def _gate_note(
    cells: Sequence[CellReport], metric: str, confidence: float = DEFAULT_CONFIDENCE
) -> str:
    """The shared insufficient-evidence clause over every cell read on `metric`."""
    return evidence_gate_clause(
        [(cell["label"], cell["paired"][metric]) for cell in cells], confidence
    )


def _qualifier(cells: Sequence[CellReport], named: Sequence[str]) -> str:
    """The shared borderline clause over the cells this verdict just named, or "" when settled.

    Without it the one-model recipe prints the same unqualified `extend_bar` / `keep_bar` sentence
    whether its deciding cell cleared zero comfortably or by nothing at all -- exactly the binary
    the roster table stopped showing.
    """
    return borderline_note(
        [(cell["label"], cell.get("stability")) for cell in cells if cell["label"] in named]
    )


def decide_bar(
    cells: Sequence[CellReport],
    *,
    baseline: str,
    candidate: str,
    confidence: float = DEFAULT_CONFIDENCE,
    adjustment: SelectionAdjustment | None = None,
) -> BarVerdict:
    """Keep recall@k as the sole adoption bar, or extend it with the scoped first-hit-rank bar.

    Read in order, because the two negative outcomes are NOT the same finding:

    1. An objective delta that clears its per-row reading AND the family of scored cells means the
       ranking advantage reached the answers in a real configuration -- that is the scope a second
       bar would carry, and it is the only result that justifies adding one.
    2. No such cell, but a reciprocal-rank delta clear of zero somewhere, is a MEASURED negative:
       the encoder does rank better inside the scored configuration and the answers do not move, so
       recall@k stays the sole bar.
    3. No cell separates on rank either. Then the premise the whole question rests on was never met
       in these configurations and the sweep decides nothing -- reporting that as "keep" would
       dress an unmet premise up as evidence.

    Whichever branch fires, the reason QUALIFIES the cell it names when a neighbouring conventional
    confidence level would read that cell differently. The decision itself never moves: `borderline`
    is a statement about how close the deciding row sits to the cut, not a fourth outcome.
    """
    per_row_answer_cells = _separated(cells, METRIC_OBJECTIVE, confidence)
    answer_cells = [
        label
        for label in per_row_answer_cells
        if adjustment is None or selection_separates(adjustment, label, confidence)
    ]
    rank_cells = _separated(cells, METRIC_RECIPROCAL_RANK, confidence)
    verdict: BarVerdict = {
        "decision": DECISION_NO_EVIDENCE,
        "baseline": baseline,
        "candidate": candidate,
        "answer_cells": answer_cells,
        "rank_cells": rank_cells,
        "borderline_cells": [
            cell["label"] for cell in cells if unsettled(cell.get("stability")) is not None
        ],
        "per_row_answer_cells": per_row_answer_cells,
        "reason": "",
    }
    if adjustment is not None:
        verdict["selection_adjustment"] = adjustment
    if answer_cells:
        detail = "; ".join(
            f"`{cell['label']}` objective "
            f"{format_interval(cell['paired'][METRIC_OBJECTIVE]['delta'])}"
            for cell in cells
            if cell["label"] in answer_cells
        )
        verdict["decision"] = DECISION_EXTEND_BAR
        verdict["reason"] = (
            f"`{candidate}` answers better than `{baseline}` in {len(answer_cells)} of "
            f"{len(cells)} cells ({detail}); a first-hit-rank gain is worth adopting under those "
            "configurations, so the bake-off gains a second bar scoped to them"
            + _qualifier(cells, answer_cells)
            + _selection_note(adjustment)
        )
        return verdict
    if rank_cells:
        best = max(cells, key=lambda cell: cell["paired"][METRIC_OBJECTIVE]["delta"]["mean"])
        verdict["decision"] = DECISION_KEEP_BAR
        verdict["reason"] = (
            f"`{candidate}` ranks the evidence earlier than `{baseline}` in "
            f"{', '.join(f'`{label}`' for label in rank_cells)} but "
            + (
                "no cell survives the objective family adjustment "
                if per_row_answer_cells
                else "no cell's objective reading separates "
            )
            + f"(best `{best['label']}` "
            f"{format_interval(best['paired'][METRIC_OBJECTIVE]['delta'])}); recall@k stays the "
            "sole adoption bar"
            + _qualifier(cells, [best["label"]])
            + _gate_note(cells, METRIC_OBJECTIVE, confidence)
            + _selection_note(adjustment)
        )
        return verdict
    verdict["reason"] = (
        f"`{candidate}` does not separate from `{baseline}` on first-hit rank in ANY scored cell, "
        "so this sweep never tested the question the second bar would answer"
        + _qualifier(cells, [cell["label"] for cell in cells])
        + _gate_note(cells, METRIC_RECIPROCAL_RANK, confidence)
        + _selection_note(adjustment)
    )
    return verdict


def _selection_note(adjustment: SelectionAdjustment | None) -> str:
    if adjustment is None:
        return ""
    details = ", ".join(
        f"{label} raw p={entry['unadjusted_p']:.4f}, adjusted p={entry['adjusted_p']:.4f}"
        for label, entry in adjustment["p_values"].items()
    )
    return (
        f"; objective selection adjustment ({adjustment['family_size']} cells, "
        f"Westfall-Young step-down max-T): {details}"
    )
