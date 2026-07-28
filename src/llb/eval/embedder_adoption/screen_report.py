"""Render embedder screen-study artifacts."""

from collections.abc import Mapping

from llb.eval.embedder_adoption.screen_models import ScreenReport
from llb.rag.fusion_evidence.evidence_gate import reading_label


def format_screen(
    report: ScreenReport,
    *,
    metadata: Mapping[str, object] | None = None,
    title: str = "Embedder adoption bar: what does a per-model screen cost?",
) -> str:
    """Render the screen cost verdict and agreement curves as Markdown."""
    verdict = report["verdict"]
    meta = dict(metadata or {})
    lines = [f"# {title}", ""]
    for key in ("split", "grounding", "goldset", "corpus"):
        if key in meta:
            lines.append(f"- {key}: `{meta[key]}`")
    lines += [
        f"- encoder pair: `{report['candidate']}` vs `{report['baseline']}`",
        f"- focus cell: `{report['focus_cell']}` (the reranker question)",
        f"- full item set: {verdict['full_n']} items; "
        f"full grid {verdict['bundles_full_grid']} bundles, focus cell alone "
        f"{verdict['bundles_focus_cell']}",
        f"- study: {report['draws']} subsamples per size, {report['resamples']} resamples each, "
        f"seed {report['seed']}, target agreement {verdict['target_agreement']:.0%}",
        f"- verdict: **{verdict['decision']}** -- {verdict['reason']}",
        "",
        "### Screen agreement with the full-set reading",
        "",
        "How often a screen of N items reproduces the same `k10+rerank` reading the whole ledger "
        "gives. A smaller ledger can either lose or invent a calibrated separation, so agreement "
        "with the full reading is measured in both directions.",
        "",
        "| model | full reading | "
        + " | ".join(f"n={size}" for size in report["sizes"])
        + " | min size |",
        "| --- | :-: | " + " | ".join(["---:"] * len(report["sizes"])) + " | :-: |",
    ]
    for entry in report["models"]:
        by_size = {curve["size"]: curve["agreement"] for curve in entry["sizes"]}
        cells = " | ".join(
            f"{by_size[size]:.0%}" if size in by_size else "-" for size in report["sizes"]
        )
        floor = str(entry["min_size"]) if entry["min_size"] is not None else "none"
        reading = reading_label(entry["full_reading"])
        lines.append(f"| `{entry['model'].split('/')[-1]}` | {reading} | {cells} | {floor} |")
    lines += [
        "",
        "Every row's re-derived full-set reading matches the reading its own sweep recorded, so "
        "the curves above are estimated from vectors that reproduce the published answer.",
        "",
    ]
    return "\n".join(lines) + "\n"


def format_screen_summary(report: ScreenReport) -> str:
    """Render a compact ASCII terminal summary."""
    verdict = report["verdict"]
    lines = [
        f"[compare-adoption-screen] focus={report['focus_cell']} "
        f"full_n={verdict['full_n']} models={len(report['models'])}",
        f"  {'model'.ljust(34)} {'reading':>10} "
        + " ".join(f"{'n=' + str(size):>6}" for size in report["sizes"])
        + "   min",
    ]
    for entry in report["models"]:
        by_size = {curve["size"]: curve["agreement"] for curve in entry["sizes"]}
        cells = " ".join(
            f"{by_size[size]:>5.0%}" if size in by_size else f"{'-':>6}" for size in report["sizes"]
        )
        floor = str(entry["min_size"]) if entry["min_size"] is not None else "none"
        lines.append(
            f"  {entry['model'].split('/')[-1][:34].ljust(34)} "
            f"{reading_label(entry['full_reading']):>10} {cells}   {floor:>4}"
        )
    lines.append(f"  verdict: {verdict['decision'].upper()} -- {verdict['reason']}")
    return "\n".join(lines)
