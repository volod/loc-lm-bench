"""Render the ablation one question type at a time.

The summary table is the point of the section: one row per question type, carrying the two numbers
the ablation exists to produce and the reading each slice reached on its own items. An operator
scanning it learns WHICH questions retrieval pays for on this corpus, which the pooled average
cannot say -- and, just as usefully, which slices it fails to pay for.

Each slice's own metric and derived tables follow underneath, because a row that says "does not
separate" is only actionable next to the n and the win/loss ledger it was read off.
"""

from collections.abc import Sequence

from llb.eval.context_ablation.models import (
    DERIVED_RETRIEVAL_UPLIFT,
    ContextAblationReport,
    DerivedComparison,
    SliceReading,
)
from llb.eval.context_ablation.report_tables import derived_table, metric_table
from llb.eval.context_ablation.verdict import by_label_of, long_context_entry
from llb.rag.fusion_evidence.stats import format_interval

_SUMMARY_NOTE = (
    "One row per question type, each decided on its OWN items: a slice reading is diagnostic and "
    "the corpus decision stays the pooled verdict above. `long_context_delta` is stated over the "
    "items the pair did not skip, and its n is the one in that column."
)

_NO_SLICES = (
    "No question type is reported: this gold set ships no `needle_items.jsonl` / "
    "`item_provenance.jsonl` sidecar, so every item is pooled into one number."
)


def _delta_cell(entry: DerivedComparison | None) -> str:
    """The delta, or `not measurable` when the pair kept no item of this slice.

    A lane that skipped every item of a question type produced no population to compare on, which
    is not the same as a delta of zero -- and zero is what a mean over nothing formats as.
    """
    if entry is None:
        return "-"
    if entry["n"] == 0:
        return "not measurable"
    return format_interval(entry["paired"]["delta"])


def _ledger_cell(entry: DerivedComparison | None) -> str:
    if entry is None or entry["n"] == 0:
        return "-"
    paired = entry["paired"]
    return f"{paired['wins']}/{paired['losses']}/{paired['ties']}"


def _summary_row(reading: SliceReading) -> str:
    by_label = by_label_of(reading["derived"])
    uplift = by_label.get(DERIVED_RETRIEVAL_UPLIFT)
    long_context = long_context_entry(by_label)
    contamination = reading["contamination"]
    long_context_n = "-" if long_context is None else str(long_context["n"])
    return (
        f"| {reading['slice']} | {reading['n']} "
        f"| {contamination['n_contaminated']}/{contamination['n']} "
        f"| {_delta_cell(uplift)} | {_ledger_cell(uplift)} "
        f"| {_delta_cell(long_context)} | {long_context_n} "
        f"| {reading['verdict']['decision']} |"
    )


def slice_summary(readings: Sequence[SliceReading], confidence: float) -> list[str]:
    """The one table that answers "which questions does retrieval pay for on this corpus"."""
    lines = ["### Per-slice reading", ""]
    if not readings:
        lines.extend([_NO_SLICES, ""])
        return lines
    lines.extend([f"{_SUMMARY_NOTE} ({confidence:.0%} bootstrap CI)", ""])
    lines.append(
        "| slice | n | closed-book matches | `retrieval_uplift` | w/l/t "
        "| `long_context_delta` | n | reading |"
    )
    lines.append("| --- | ---: | ---: | ---: | :-: | ---: | ---: | --- |")
    lines.extend(_summary_row(reading) for reading in readings)
    lines.append("")
    return lines


def slice_sections(report: ContextAblationReport) -> list[str]:
    """The summary table, then every slice's own metrics and derived deltas."""
    readings = report.get("slice_readings", [])
    lines = slice_summary(readings, report["confidence"])
    for reading in readings:
        name = reading["slice"]
        lines += metric_table(report, name, f"Slice: {name}", "Question-type slice")
        lines += derived_table(
            reading["derived"],
            heading=f"#### Derived numbers -- {name}",
            empty_note="No derived delta is measurable on this slice.",
        )
        lines.append(f"Reading: {reading['verdict']['reason'] or 'no reading was reached'}.")
        lines.append("")
    return lines


__all__ = ["slice_sections", "slice_summary"]
