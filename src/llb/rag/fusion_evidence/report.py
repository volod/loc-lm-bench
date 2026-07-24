"""Render the fusion-evidence report as ASCII Markdown (AGENTS.md: no Unicode, no box-drawing).

Three tables, in the order a reviewer needs them: the focus slice with uncertainty (is there a
multi-hop gain at all?), the overall table (did it cost anything?), and the item-level paired
ledger (with n in the low tens, the individual items ARE the evidence).
"""

from llb.rag.fusion_evidence.models import (
    METRIC_ALL_SPANS,
    METRIC_COVERAGE,
    METRIC_MRR,
    METRIC_RECALL,
    FusionEvidenceReport,
)
from llb.rag.fusion_evidence.slices import SliceReport
from llb.rag.fusion_evidence.stats import format_interval

_HEADLINE_METRICS = (METRIC_RECALL, METRIC_ALL_SPANS, METRIC_COVERAGE, METRIC_MRR)
_HEADERS = {
    METRIC_RECALL: "recall@k",
    METRIC_ALL_SPANS: "all-spans@k",
    METRIC_COVERAGE: "span coverage",
    METRIC_MRR: "MRR",
}


def _metric_table(
    report: FusionEvidenceReport, pick: str | None, title: str, note: str
) -> list[str]:
    """One row-per-backend table of `mean [lo, hi]` per metric, plus the paired recall delta."""
    lines = [f"### {title}", ""]
    selected: dict[str, SliceReport] = {}
    for label, row in report["rows"].items():
        slice_report = row["overall"] if pick is None else row["slices"].get(pick)
        if slice_report is not None:
            selected[label] = slice_report
    n = next((entry["n"] for entry in selected.values()), 0)
    lines.append(f"{note} (n={n}, k={report['k']}, {report['confidence']:.0%} bootstrap CI)")
    lines.append("")
    if n == 0:
        # An all-zero table reads like a measured result; say plainly that nothing was scored.
        lines.extend(["No item falls in this slice, so no metric is measured here.", ""])
        return lines
    header = " | ".join(_HEADERS[metric] for metric in _HEADLINE_METRICS)
    lines.append(f"| row | {header} | recall delta vs {report['baseline']} | w/l/t | sign p |")
    lines.append(
        "| --- | " + " | ".join(["---:"] * len(_HEADLINE_METRICS)) + " | ---: | :-: | ---: |"
    )
    for label in sorted(selected):
        entry = selected[label]
        cells = [format_interval(entry["metrics"][metric]) for metric in _HEADLINE_METRICS]
        paired = entry["paired_vs_baseline"][METRIC_RECALL]
        lines.append(
            f"| {label} | "
            + " | ".join(cells)
            + f" | {format_interval(paired['delta'])} "
            + f"| {paired['wins']}/{paired['losses']}/{paired['ties']} "
            + f"| {paired['sign_test_p']:.3f} |"
        )
    lines.append("")
    return lines


def _item_table(report: FusionEvidenceReport) -> list[str]:
    """Per-item recall / all-spans outcome of every row on the focus slice."""
    items = report["focus_items"]
    lines = [f"### Item-level outcomes ({report['focus_slice']})", ""]
    if not items:
        lines.extend([f"No {report['focus_slice']} item was scored.", ""])
        return lines
    labels = sorted(report["rows"])
    lines.append("| item | spans | " + " | ".join(labels) + " |")
    lines.append("| --- | ---: | " + " | ".join([":-:"] * len(labels)) + " |")
    for item in items:
        cells = [
            f"{int(item['rows'][label][METRIC_RECALL])}/"
            f"{int(item['rows'][label][METRIC_ALL_SPANS])}"
            for label in labels
        ]
        lines.append(f"| {item['item_id']} | {item['n_spans']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Each cell is `recall@k / all-spans@k` (1 = hit, 0 = miss).")
    lines.append("")
    return lines


def _agreement_table(report: FusionEvidenceReport) -> list[str]:
    """Per fused row: how often the two lanes vouch for the SAME candidate.

    Fusion can only reward agreement it can see, so this table is what a span-identity policy is
    read against -- and it is also the precondition for candidate depth to matter at all.
    """
    measured = {
        label: row["agreement"] for label, row in report["rows"].items() if "agreement" in row
    }
    if not measured:
        return []
    lines = [
        "### Cross-lane agreement",
        "",
        "Candidates BOTH lanes returned in the fused pool, per question.",
        "",
        "| row | questions with a shared candidate | share | mean shared candidates |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in sorted(measured):
        entry = measured[label]
        lines.append(
            f"| {label} | {entry['questions_with_shared_candidate']}/{entry['questions']} "
            f"| {entry['share_of_questions']:.3f} | {entry['mean_shared_candidates']:.3f} |"
        )
    lines.append("")
    return lines


def _routing_table(report: FusionEvidenceReport) -> list[str]:
    """Show how routed rows divided the scored questions and which signal supplied the route."""
    measured = {label: row["routing"] for label, row in report["rows"].items() if "routing" in row}
    if not measured:
        return []
    lines = [
        "### Question routing",
        "",
        "| row | graph | vector | sidecar | heuristic |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in sorted(measured):
        entry = measured[label]
        lines.append(
            f"| {label} | {entry['graph_questions']} | {entry['vector_questions']} "
            f"| {entry['sidecar_questions']} | {entry['heuristic_questions']} |"
        )
    lines.append("")
    lines.extend(
        [
            "Routes by question-type slice:",
            "",
            "| row | slice | graph | vector |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for label in sorted(measured):
        for name, counts in sorted(measured[label]["slices"].items()):
            lines.append(
                f"| {label} | {name} | {counts['graph_questions']} | {counts['vector_questions']} |"
            )
    lines.append("")
    return lines


def format_report(report: FusionEvidenceReport, *, title: str = "Graph-vector fusion") -> str:
    """The full Markdown artifact: verdict, focus slice, overall, other slices, item ledger."""
    verdict = report["verdict"]
    lines = [
        f"# {title}: multi-hop evidence",
        "",
        f"- baseline row: `{report['baseline']}`",
        f"- scored items: {report['n']} (k={report['k']})",
        f"- focus slice: `{verdict['focus_slice']}` (n={verdict['focus_n']})",
        f"- bootstrap: {report['resamples']} resamples, seed {report['seed']}",
        f"- verdict: **{verdict['decision']}**"
        + (f" -- {verdict['reason']}" if verdict["reason"] else ""),
        "",
    ]
    lines += _metric_table(
        report,
        report["focus_slice"],
        f"Focus slice: {report['focus_slice']}",
        "Questions whose answer needs evidence from more than one span",
    )
    lines += _metric_table(report, None, "Overall", "Every scored item")
    other = [
        name
        for row in report["rows"].values()
        for name in row["slices"]
        if name != report["focus_slice"]
    ]
    for name in sorted(set(other)):
        lines += _metric_table(report, name, f"Slice: {name}", "Context slice")
    lines += _agreement_table(report)
    lines += _routing_table(report)
    lines += _floor_section(report)
    lines += _item_table(report)
    return "\n".join(lines)


def _floor_section(report: FusionEvidenceReport) -> list[str]:
    """The measurement floor per swept row, when it was measured.

    A sweep publishes a dozen rows that differ in the third decimal; the floor is what separates
    "this weight retrieves more" from "this weight breaks ties differently". The focus slice gets
    its own block, because the verdict is read there and a floor over every item does not bound
    the band of a slice a third its size.
    """
    floor = report.get("noise_floor")
    if floor is None:
        return []
    from llb.rag.noise_floor_report import render_noise_floor_markdown

    lines = render_noise_floor_markdown(floor, scored="every item")
    focus = report.get("noise_floor_focus")
    if focus is not None:
        lines += render_noise_floor_markdown(
            focus,
            title=f"Measurement floor: {report['focus_slice']}",
            scored=f"{report['focus_slice']} items",
        )
    return lines
