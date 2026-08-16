"""Did the retrieval budget's coverage gain reach the ANSWERS?

The retrieval-budget probe can show that a stuck multi-hop `all-spans@k` is a property of `top_k`:
the same rows re-scored at a larger budget carry much more of the evidence with no ranking change.
That is a claim about the context, and three things can happen to it end to end -- the coverage
CONVERTS into better answers, it STALLS (more evidence, same answers), or it converts nowhere and
costs a slice the extra context displaced something in.

This module names which one happened. It decides nothing new: each row's reading is the ordinary
lane judgment (`judge_lane`) with the row's own shipped-budget cell as the baseline, so a budget
verdict and a lane verdict mean the same thing by construction. What it adds is the COST scan --
every non-focus slice whose objective the extra context measurably lowered -- because a conversion
that quietly costs the factoid slice is not a conversion an operator should buy.
"""

from collections.abc import Mapping, Sequence

from llb.eval.answer_quality.budgets import split_budget_label
from llb.eval.answer_quality.models import (
    CONVERSION_CONVERTED,
    CONVERSION_STALLED,
    METRIC_OBJECTIVE,
    VERDICT_ANSWER_GAIN,
    VERDICT_INCONCLUSIVE,
    VERDICT_NO_EVIDENCE,
    VERDICT_NO_GAIN,
    VERDICT_RETRIEVAL_ONLY,
    BudgetConversion,
    CrossReading,
    RowConversion,
)
from llb.eval.answer_quality.verdict import judge_lane
from llb.rag.fusion_evidence.paired import regresses
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE

# Row outcome -> the headline the sweep gets when it is the strongest one measured.
_HEADLINE = {
    VERDICT_ANSWER_GAIN: CONVERSION_CONVERTED,
    VERDICT_RETRIEVAL_ONLY: CONVERSION_STALLED,
    VERDICT_INCONCLUSIVE: VERDICT_INCONCLUSIVE,
    VERDICT_NO_GAIN: VERDICT_NO_GAIN,
}
_STRENGTH = list(_HEADLINE)


def cost_slices(
    reading: CrossReading, focus_slice: str, confidence: float = DEFAULT_CONFIDENCE
) -> list[str]:
    """Question types whose objective the raised budget measurably LOWERED."""
    return sorted(
        name
        for name, entry in reading["slices"].items()
        if name != focus_slice
        and regresses(entry["paired_vs_baseline"][METRIC_OBJECTIVE], confidence)
    )


def _row_conversion(
    reading: CrossReading, *, focus_slice: str, coverage: str, confidence: float
) -> RowConversion:
    label, base_label = reading["label"], reading["base_lane"]
    row, budget = split_budget_label(label)
    _, base_budget = split_budget_label(base_label)
    decision, reason = judge_lane(reading, label, base_label, focus_slice, coverage, confidence)
    costs = cost_slices(reading, focus_slice, confidence)
    if costs:
        reason += (
            f" COST: the raised budget lowers the objective on {', '.join(costs)} by an interval "
            "clear of zero, so this coverage is not free"
        )
    return {
        "row": row,
        "lane": label,
        "base_lane": base_label,
        "budget": budget or 0,
        "base_budget": base_budget or 0,
        "decision": decision,
        "reason": reason,
        "cost_slices": costs,
    }


def budget_conversion(
    cross_readings: Mapping[str, CrossReading],
    *,
    budgets: Sequence[int],
    focus_slice: str,
    coverage: str,
    confidence: float = DEFAULT_CONFIDENCE,
) -> BudgetConversion:
    """The per-row conversion readings plus the one sentence the sweep is read on."""
    rows = sorted(
        (
            _row_conversion(
                reading, focus_slice=focus_slice, coverage=coverage, confidence=confidence
            )
            for reading in cross_readings.values()
        ),
        key=lambda row: (row["budget"], row["row"]),
    )
    return {
        "budgets": list(budgets),
        "focus_slice": focus_slice,
        "coverage_metric": coverage,
        "decision": _headline(rows),
        "reason": _reason(rows, focus_slice),
        "rows": rows,
    }


def _headline(rows: Sequence[RowConversion]) -> str:
    """The strongest outcome any row reached -- a conversion anywhere is the sweep's answer."""
    reached = [row["decision"] for row in rows if row["decision"] in _HEADLINE]
    if not reached:
        return VERDICT_NO_EVIDENCE
    return _HEADLINE[min(reached, key=_STRENGTH.index)]


def _reason(rows: Sequence[RowConversion], focus_slice: str) -> str:
    if not rows:
        return "no row was scored at a second retrieval budget"
    strongest = min(rows, key=lambda row: _STRENGTH.index(row["decision"]))
    others = [row for row in rows if row is not strongest]
    detail = "; ".join(f"`{row['row']}` {row['decision']}" for row in others)
    lead = (
        f"on {focus_slice}, `{strongest['row']}` at k={strongest['budget']} is "
        f"{strongest['decision']} against its own k={strongest['base_budget']} "
        f"-- {strongest['reason']}"
    )
    return lead + (f". Other rows: {detail}" if detail else "")


__all__ = ["budget_conversion", "cost_slices"]
