"""Compose the context ablation as ASCII Markdown (AGENTS.md: no Unicode, no box-drawing).

The derived table comes FIRST, before the per-lane means: an operator reading this artifact wants
"did retrieval pay for itself", not three numbers to subtract by hand. The per-lane table, the
question-type slices, and the flagged-item ledger follow as the evidence behind it.

The tables themselves live in `report_tables.py` (they are rendered over the pooled set and over
each slice alike) and the slice section in `report_slices.py`; this module is the running order.
"""

from collections.abc import Mapping

from llb.backends.served_window import BUDGET_SOURCE_SERVED
from llb.eval.context_ablation.models import (
    METRIC_OBJECTIVE,
    RETRIEVED_DOCUMENT_NOT_MEASURED,
    ContextAblationReport,
    ContextAblationVerdict,
    ItemOutcome,
    LaneReport,
)
from llb.rag.fusion_evidence.evidence_gate import (
    evidence_gate_summary,
)
from llb.rag.fusion_evidence.stability import boundary_table
from llb.rag.fusion_evidence.paired import gated_readings
from llb.eval.context_ablation.report_power import power_section
from llb.eval.context_ablation.report_slices import slice_sections
from llb.eval.context_ablation.report_stability import stability_note, stability_section
from llb.eval.context_ablation.report_tables import derived_table, metric_table

# Item rows worth printing: at a few dozen items the flagged ones ARE the evidence, and a full
# ledger of every scored item belongs in `comparison.json`, not in the narrative artifact.
_LEDGER_NOTE = (
    "Only flagged items are listed: `contaminated` = the closed-book lane already matched the "
    "reference; `skipped` = a lane's context did not fit the model window. Every scored item is "
    "in `comparison.json`."
)


def _gate_summary(report: ContextAblationReport) -> list[str]:
    """How many of the derived deltas the minimum-evidence gate relabeled."""
    gated, total = gated_readings(
        [entry["paired"] for entry in report["derived"]], report["confidence"]
    )
    return evidence_gate_summary(gated, total, report["confidence"], subject="derived delta")


def _boundary_section(report: ContextAblationReport) -> list[str]:
    """How close each derived delta sits to the cut the ablation verdict is taken from.

    Both the retrieval uplift and long-context delta use the calibrated paired cut, so either
    verdict can rest on a row a neighbouring convention would flip.
    """
    rows = [
        (f"`{entry['label']}`", stability)
        for entry in report["derived"]
        if (stability := entry["paired"].get("stability")) is not None
    ]
    return boundary_table(
        rows,
        title="How close each derived delta sits to the cut",
        key_header="delta",
        subject="the candidate lane",
        confidence=report["confidence"],
    )


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


def _adoption_lines(verdict: ContextAblationVerdict) -> list[str]:
    """The adopt-or-reject call on `retrieved_document`, stated apart from the ablation verdict.

    An operator can act on this one, so it is not buried in the derived table: it names the
    decision, the delta it rests on, and how much of the oracle gap the lane captured.
    """
    adoption = verdict.get("retrieved_document")
    if adoption is None or adoption["decision"] == RETRIEVED_DOCUMENT_NOT_MEASURED:
        return []
    share = adoption["captured_share"]
    captured = f", capturing {share:.0%} of the oracle gap" if share is not None else ""
    skipped = f", {adoption['skipped']} item(s) skipped" if adoption["skipped"] else ""
    return [
        f"- retrieved-document lane: **{adoption['decision']}** "
        f"({adoption['delta']:+.3f} objective vs rag on n={adoption['n']}{captured}{skipped})"
        + (f" -- {adoption['reason']}" if adoption["reason"] else "")
    ]


def _window_note(lane: LaneReport) -> str:
    """Which window this lane's skip count was measured against.

    A skip count is unreadable without it: "3 items skipped" against a declared 32k window and
    against the 4k one Ollama turned out to be serving are different findings, and the second is
    the one that would otherwise have been a silently truncated document reported as delivered.
    """
    binding = lane.get("context_window")
    if not binding:
        return ""
    bound, other_label, other = (
        (binding["served_max_model_len"], "declared", binding["declared_max_model_len"])
        if binding["budget_source"] == BUDGET_SOURCE_SERVED
        else (binding["declared_max_model_len"], "served", binding["served_max_model_len"])
    )
    if not bound:
        return " -- window unbounded (nothing could be skipped for size)"
    against = f", {other_label} {other}" if other and other != bound else ""
    return f" -- window {bound} tokens ({binding['budget_source']}{against})"


def _lane_list(report: ContextAblationReport) -> list[str]:
    lines = []
    for label, lane in sorted(report["lanes"].items()):
        skipped = len(lane["skipped_item_ids"])
        suffix = f" -- {skipped} item(s) skipped (context did not fit)" if skipped else ""
        lines.append(f"  - `{label}`{suffix}{_window_note(lane)}")
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
        f"({contamination['rate']:.1%}{stability_note(report)}) -- parametric knowledge or "
        "corpus contamination",
        f"- bootstrap: {report['resamples']} resamples, seed {report['seed']}",
        f"- verdict: **{verdict['decision']}**"
        + (f" -- {verdict['reason']}" if verdict["reason"] else ""),
        *_adoption_lines(verdict),
        "- scored lanes:",
    ]
    lines += _lane_list(report)
    lines.append("")
    lines += power_section(report)
    lines += derived_table(report["derived"])
    lines += _gate_summary(report)
    lines += _boundary_section(report)
    lines += stability_section(report)
    lines += metric_table(report, None, "Per lane", "Every scored item")
    lines += slice_sections(report)
    lines += _item_table(report)
    return "\n".join(lines) + "\n"
