"""The two tables the ablation artifact repeats: derived deltas, and metrics per lane.

Both are rendered over a POPULATION -- the whole scored set, or one question-type slice -- so they
live here rather than in the composer: the pooled table and a slice's table are the same table, and
a reader comparing a slice against the pool has to be reading identical columns to do it at all.
"""

from collections.abc import Sequence

from llb.eval.context_ablation.models import (
    DERIVED_LONG_CONTEXT_DELTA,
    DERIVED_LONG_CONTEXT_DELTA_FITTING,
    DERIVED_ORACLE_DOCUMENT_GAP,
    DERIVED_ORACLE_DOCUMENT_GAP_FITTING,
    DERIVED_RETRIEVAL_UPLIFT,
    DERIVED_RETRIEVED_DOCUMENT_DELTA,
    DERIVED_RETRIEVED_DOCUMENT_DELTA_FITTING,
    METRIC_OBJECTIVE,
    METRIC_RETRIEVAL_HIT,
    METRIC_TOKEN_F1,
    METRICS,
    ContextAblationReport,
    DerivedComparison,
)
from llb.rag.fusion_evidence.slices import SliceReport
from llb.rag.fusion_evidence.stability import format_reading
from llb.rag.fusion_evidence.stats import format_interval
from llb.rag.fusion_evidence.paired import format_randomization_p

_HEADERS = {
    METRIC_OBJECTIVE: "objective",
    METRIC_TOKEN_F1: "token F1",
    METRIC_RETRIEVAL_HIT: "recall@k",
}

DERIVED_NOTES = {
    DERIVED_RETRIEVAL_UPLIFT: "how much of the RAG score retrieval paid for",
    DERIVED_LONG_CONTEXT_DELTA: "whole-document stuffing versus chunked retrieval (ORACLE)",
    DERIVED_LONG_CONTEXT_DELTA_FITTING: "the same delta over items the pair did not skip",
    DERIVED_RETRIEVED_DOCUMENT_DELTA: "the part of that gap an operator captures, no gold label",
    DERIVED_RETRIEVED_DOCUMENT_DELTA_FITTING: "the same delta over items the pair did not skip",
    DERIVED_ORACLE_DOCUMENT_GAP: "what was left that only the gold label could supply",
    DERIVED_ORACLE_DOCUMENT_GAP_FITTING: "the same delta over items the pair did not skip",
}


def derived_table(
    entries: Sequence[DerivedComparison],
    *,
    heading: str = "### Derived numbers",
    empty_note: str = "No derived delta is available: the comparison scored one lane.",
) -> list[str]:
    """The paired candidate-minus-reference deltas measurable over one population."""
    lines = [heading, ""]
    if not entries:
        lines.extend([empty_note, ""])
        return lines
    lines.append(
        "| delta | candidate - reference | n | value | w/l/t | sign p | rand p | reading "
        "| reads as |"
    )
    lines.append("| --- | --- | ---: | ---: | :-: | ---: | ---: | :-: | --- |")
    for entry in entries:
        paired = entry["paired"]
        stability = paired.get("stability")
        lines.append(
            f"| `{entry['label']}` | `{entry['candidate']}` - `{entry['reference']}` "
            f"| {entry['n']} | {format_interval(paired['delta'])} "
            f"| {paired['wins']}/{paired['losses']}/{paired['ties']} "
            f"| {paired['sign_test_p']:.3f} | {format_randomization_p(paired)} "
            f"| {format_reading(stability, stability['reading']) if stability else '-'} "
            f"| {DERIVED_NOTES.get(entry['label'], entry['population'])} |"
        )
    lines.append("")
    return lines


def metric_table(
    report: ContextAblationReport, pick: str | None, title: str, note: str
) -> list[str]:
    """One row-per-lane table of `mean [lo, hi]` per metric, plus the paired objective delta."""
    lines = [f"### {title}", ""]
    selected: dict[str, SliceReport] = {}
    for label, lane in report["lanes"].items():
        entry = lane["overall"] if pick is None else lane["slices"].get(pick)
        if entry is not None:
            selected[label] = entry
    n = next((entry["n"] for entry in selected.values()), 0)
    lines.extend([f"{note} (n={n}, {report['confidence']:.0%} bootstrap CI)", ""])
    if n == 0:
        lines.extend(["No item falls in this slice, so no metric is measured here.", ""])
        return lines
    header = " | ".join(_HEADERS.get(metric, metric) for metric in METRICS)
    lines.append(
        f"| lane | {header} | objective delta vs {report['baseline']} | w/l/t | sign p | rand p |"
    )
    lines.append("| --- | " + " | ".join(["---:"] * len(METRICS)) + " | ---: | :-: | ---: | ---: |")
    for label in sorted(selected):
        entry = selected[label]
        cells = [format_interval(entry["metrics"][metric]) for metric in METRICS]
        paired = entry["paired_vs_baseline"][METRIC_OBJECTIVE]
        lines.append(
            f"| {label} | "
            + " | ".join(cells)
            + f" | {format_interval(paired['delta'])} "
            + f"| {paired['wins']}/{paired['losses']}/{paired['ties']} "
            + f"| {paired['sign_test_p']:.3f} | {format_randomization_p(paired)} |"
        )
    lines.append("")
    return lines


__all__ = ["DERIVED_NOTES", "derived_table", "metric_table"]
