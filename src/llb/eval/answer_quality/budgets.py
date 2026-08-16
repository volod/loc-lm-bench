"""The retrieval BUDGET dimension of the answer-quality lane.

A stuck multi-hop `all-spans@k` can be a property of `top_k` rather than of the ranking: the same
rows re-scored at a larger budget carry far more of the evidence with no retrieval-side change at
all. That is a statement about what the CONTEXT carries, and it is silent about the answer, because
five times the chunks is also five times the context and part of the extra coverage arrives as
short entity mentions rather than as readable chunks.

This module turns one lane selection plus one budget selection into the `(lane x budget)` cells the
comparison scores, and names the pair whose reading answers the conversion question: the SAME
retrieval row at two budgets. The budget rides in the lane LABEL (`vector#k50`) so a cell stays a
plain lane everywhere downstream -- one run bundle, one row in every table, one entry in the item
ledger -- and the label still parses back into the retrieval knobs that produced it.
"""

from collections.abc import Sequence

from llb.eval.answer_quality.models import LaneSpec

# Separates a sweep row label from the budget it was scored at. `#` cannot occur in a sweep row
# label (`vector`, `graph/<strategy>`, `fused/<strategy>@<weight>/d<depth>/i<identity>/r<ratio>`),
# so the split is unambiguous in both directions.
BUDGET_MARKER = "#k"


def budget_label(row: str, budget: int) -> str:
    """`vector` at 50 -> `vector#k50`."""
    return f"{row}{BUDGET_MARKER}{budget}"


def split_budget_label(label: str) -> tuple[str, int | None]:
    """`vector#k50` -> `("vector", 50)`; a label with no budget suffix keeps `None`."""
    row, marker, token = label.rpartition(BUDGET_MARKER)
    if not marker:
        return label, None
    try:
        budget = int(token)
    except ValueError:
        raise ValueError(f"retrieval budget must be an integer in lane label {label!r}") from None
    if budget < 1:
        raise ValueError(f"retrieval budget must be at least 1 in lane label {label!r}")
    return row, budget


def expand_budget_lanes(lanes: Sequence[LaneSpec], budgets: Sequence[int]) -> list[LaneSpec]:
    """Every `(lane, budget)` cell, lane-major so a row's budgets sit together in every table.

    `lanes[0]` at `budgets[0]` stays first, so the comparison's baseline remains the shipped
    configuration of the baseline row and every other cell is read against it.
    """
    if not budgets:
        raise ValueError("name at least one retrieval budget")
    return [
        lane._replace(label=budget_label(lane.label, budget), top_k=budget)
        for lane in lanes
        for budget in budgets
    ]


def conversion_baselines(lanes: Sequence[LaneSpec], budgets: Sequence[int]) -> dict[str, str]:
    """Per raised-budget cell, the cell of the SAME row at the smallest compared budget.

    That pairing is the conversion question itself: everything except `top_k` is held fixed, so a
    delta between the two is the budget and nothing else.
    """
    base, *raised = budgets
    return {
        budget_label(lane.label, budget): budget_label(lane.label, base)
        for lane in lanes
        for budget in raised
    }


__all__ = [
    "BUDGET_MARKER",
    "budget_label",
    "conversion_baselines",
    "expand_budget_lanes",
    "split_budget_label",
]
