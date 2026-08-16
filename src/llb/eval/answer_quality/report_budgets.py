"""Render the budget-conversion section of the answer-quality artifact.

The lane tables read every cell against ONE baseline, which answers "is this configuration better
than the shipped one". The budget question is the other pairing -- the same retrieval row at two
budgets -- so it gets its own table: what the extra budget bought on the focus slice, what it cost
elsewhere, and what it charged for the privilege in context characters.
"""

from collections.abc import Mapping

from llb.eval.answer_quality.models import (
    METRIC_CONTEXT_CHARS,
    METRIC_OBJECTIVE,
    METRIC_PROMPT_TOKENS,
    AnswerQualityReport,
    BudgetConversion,
    CrossReading,
    RowConversion,
)
from llb.rag.fusion_evidence.paired import format_randomization_p
from llb.rag.fusion_evidence.stats import format_interval

_HEADERS = {
    METRIC_OBJECTIVE: "objective delta",
    METRIC_CONTEXT_CHARS: "context chars delta",
    METRIC_PROMPT_TOKENS: "prompt tokens delta",
}


def _row_line(
    row: RowConversion, reading: CrossReading, focus_slice: str, metrics: list[str]
) -> str:
    entry = reading["slices"].get(focus_slice)
    prefix = f"| `{row['row']}` | {row['budget']} | "
    if entry is None:
        return prefix + " | ".join(["-"] * (len(metrics) + 3)) + " |"
    paired = entry["paired_vs_baseline"]
    objective = paired[METRIC_OBJECTIVE]
    return (
        prefix
        + " | ".join(format_interval(paired[metric]["delta"]) for metric in metrics)
        + f" | {objective['wins']}/{objective['losses']}/{objective['ties']} "
        + f"| {objective['sign_test_p']:.3f} | {format_randomization_p(objective)} |"
    )


def _table(
    conversion: BudgetConversion, readings: Mapping[str, CrossReading], metrics: list[str]
) -> list[str]:
    base = conversion["rows"][0]["base_budget"] if conversion["rows"] else 0
    header = " | ".join(_HEADERS.get(metric, f"{metric} delta") for metric in metrics)
    focus = conversion["focus_slice"]
    lines = [
        f"Each delta is the SAME retrieval row at its raised budget minus itself at k={base}, on "
        f"the `{focus}` slice.",
        "",
        f"| row | k | {header} | w/l/t | sign p | rand p |",
        "| --- | ---: | " + " | ".join(["---:"] * len(metrics)) + " | :-: | ---: | ---: |",
    ]
    lines += [_row_line(row, readings[row["lane"]], focus, metrics) for row in conversion["rows"]]
    lines.append("")
    return lines


def budget_section(report: AnswerQualityReport) -> list[str]:
    """The whole section, or nothing at all when the run scored a single budget."""
    conversion = report.get("budget_conversion")
    readings = report.get("cross_readings")
    if not conversion or not readings:
        return []
    metrics = [
        metric
        for metric in (
            METRIC_OBJECTIVE,
            conversion["coverage_metric"],
            METRIC_CONTEXT_CHARS,
            METRIC_PROMPT_TOKENS,
        )
        if metric in report["metrics"]
    ]
    budgets = ", ".join(str(budget) for budget in conversion["budgets"])
    lines = [
        f"### Budget conversion (k = {budgets})",
        "",
        f"- verdict: **{conversion['decision']}** -- {conversion['reason']}",
        "",
    ]
    lines += _table(conversion, readings, metrics)
    lines.append("Per-row readings:")
    lines.append("")
    lines += [
        f"- `{row['row']}` k={row['budget']} vs k={row['base_budget']}: "
        f"**{row['decision']}** -- {row['reason']}"
        for row in conversion["rows"]
    ]
    lines.append("")
    return lines


__all__ = ["budget_section"]
