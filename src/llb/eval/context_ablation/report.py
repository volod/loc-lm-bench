"""Render the context ablation as ASCII Markdown (AGENTS.md: no Unicode, no box-drawing).

The derived table comes FIRST, before the per-lane means: an operator reading this artifact wants
"did retrieval pay for itself", not three numbers to subtract by hand. The per-lane table, the
question-type slices, and the flagged-item ledger follow as the evidence behind it.
"""

from collections.abc import Mapping, Sequence

from llb.eval.context_ablation.models import (
    METRIC_OBJECTIVE,
    METRIC_RETRIEVAL_HIT,
    METRIC_TOKEN_F1,
    METRICS,
    ContextAblationReport,
    DerivedComparison,
    ItemOutcome,
)
from llb.rag.fusion_evidence.slices import SliceReport
from llb.rag.fusion_evidence.stats import format_interval

_HEADERS = {
    METRIC_OBJECTIVE: "objective",
    METRIC_TOKEN_F1: "token F1",
    METRIC_RETRIEVAL_HIT: "recall@k",
}

_DERIVED_NOTES = {
    "retrieval_uplift": "how much of the RAG score retrieval paid for",
    "long_context_delta": "whole-document stuffing versus chunked retrieval",
    "long_context_delta_fitting": "the same delta over items the long-context lane did not skip",
}

# Item rows worth printing: at a few dozen items the flagged ones ARE the evidence, and a full
# ledger of every scored item belongs in `comparison.json`, not in the narrative artifact.
_LEDGER_NOTE = (
    "Only flagged items are listed: `contaminated` = the closed-book lane already matched the "
    "reference; `skipped` = a lane's context did not fit the model window. Every scored item is "
    "in `comparison.json`."
)


def _derived_table(entries: Sequence[DerivedComparison]) -> list[str]:
    lines = ["### Derived numbers", ""]
    if not entries:
        lines.extend(["No derived delta is available: the comparison scored one lane.", ""])
        return lines
    lines.append("| delta | candidate - reference | n | value | w/l/t | sign p | reads as |")
    lines.append("| --- | --- | ---: | ---: | :-: | ---: | --- |")
    for entry in entries:
        paired = entry["paired"]
        lines.append(
            f"| `{entry['label']}` | `{entry['candidate']}` - `{entry['reference']}` "
            f"| {entry['n']} | {format_interval(paired['delta'])} "
            f"| {paired['wins']}/{paired['losses']}/{paired['ties']} "
            f"| {paired['sign_test_p']:.3f} "
            f"| {_DERIVED_NOTES.get(entry['label'], entry['population'])} |"
        )
    lines.append("")
    return lines


def _metric_table(
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
    lines.append(f"| lane | {header} | objective delta vs {report['baseline']} | w/l/t | sign p |")
    lines.append("| --- | " + " | ".join(["---:"] * len(METRICS)) + " | ---: | :-: | ---: |")
    for label in sorted(selected):
        entry = selected[label]
        cells = [format_interval(entry["metrics"][metric]) for metric in METRICS]
        paired = entry["paired_vs_baseline"][METRIC_OBJECTIVE]
        lines.append(
            f"| {label} | "
            + " | ".join(cells)
            + f" | {format_interval(paired['delta'])} "
            + f"| {paired['wins']}/{paired['losses']}/{paired['ties']} "
            + f"| {paired['sign_test_p']:.3f} |"
        )
    lines.append("")
    return lines


def _flags(item: ItemOutcome, skipped_by_item: Mapping[str, list[str]]) -> str:
    flags = ["contaminated"] if item["contaminated"] else []
    flags.extend(f"skipped:{label}" for label in skipped_by_item.get(item["item_id"], []))
    return ", ".join(flags)


def _item_table(report: ContextAblationReport) -> list[str]:
    """Per-item objective of every lane, for the contaminated and skipped items."""
    skipped_by_item: dict[str, list[str]] = {}
    for label, lane in sorted(report["lanes"].items()):
        for item_id in lane["skipped_item_ids"]:
            skipped_by_item.setdefault(item_id, []).append(label)
    flagged = [
        item
        for item in report["items"]
        if item["contaminated"] or item["item_id"] in skipped_by_item
    ]
    lines = ["### Flagged items", ""]
    if not flagged:
        lines.extend(
            [
                "No item was flagged: the closed-book lane matched nothing and nothing was "
                "skipped.",
                "",
            ]
        )
        return lines
    labels = sorted(report["lanes"])
    lines.append("| item | " + " | ".join(labels) + " | flags |")
    lines.append("| --- | " + " | ".join([":-:"] * len(labels)) + " | --- |")
    for item in flagged:
        cells = [f"{item['lanes'][label][METRIC_OBJECTIVE]:.2f}" for label in labels]
        lines.append(
            f"| {item['item_id']} | " + " | ".join(cells) + f" | {_flags(item, skipped_by_item)} |"
        )
    lines.extend(["", "Each cell is the lane's `objective`. " + _LEDGER_NOTE, ""])
    return lines


def _lane_list(report: ContextAblationReport) -> list[str]:
    lines = []
    for label, lane in sorted(report["lanes"].items()):
        skipped = len(lane["skipped_item_ids"])
        suffix = f" -- {skipped} item(s) skipped (context did not fit)" if skipped else ""
        lines.append(f"  - `{label}`{suffix}")
        lines.extend(f"    - `{run_dir}`" for run_dir in lane["run_dirs"])
    return lines


def format_report(
    report: ContextAblationReport,
    *,
    metadata: Mapping[str, object] | None = None,
    title: str = "RAG versus long context",
) -> str:
    """The full Markdown artifact: verdict, derived deltas, lanes, slices, flagged items."""
    verdict = report["verdict"]
    contamination = report["contamination"]
    meta = dict(metadata or {})
    lines = [f"# {title}", ""]
    for key in ("model", "backend", "split", "grounding", "goldset", "corpus"):
        if key in meta:
            lines.append(f"- {key}: `{meta[key]}`")
    lines += [
        f"- baseline lane: `{report['baseline']}`",
        f"- scored items: {report['n']} (identical item set in every lane)",
        f"- closed-book matches: {contamination['n_contaminated']}/{contamination['n']} "
        f"({contamination['rate']:.1%}) -- parametric knowledge or corpus contamination",
        f"- bootstrap: {report['resamples']} resamples, seed {report['seed']}",
        f"- verdict: **{verdict['decision']}**"
        + (f" -- {verdict['reason']}" if verdict["reason"] else ""),
        "- scored lanes:",
    ]
    lines += _lane_list(report)
    lines.append("")
    lines += _derived_table(report["derived"])
    lines += _metric_table(report, None, "Per lane", "Every scored item")
    other = sorted({name for lane in report["lanes"].values() for name in lane["slices"]})
    for name in other:
        lines += _metric_table(report, name, f"Slice: {name}", "Question-type slice")
    lines += _item_table(report)
    return "\n".join(lines) + "\n"
