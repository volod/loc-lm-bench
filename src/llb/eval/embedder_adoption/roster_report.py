"""Render the roster property reading as ASCII Markdown (AGENTS.md: no Unicode, no box-drawing).

Split out of `roster.py` so the reading logic and its presentation stay separately readable, the
same seam `embedding_bakeoff_report.py` uses for the bake-off.

A reading that a looser-but-conventional confidence level would change is marked `(borderline)`,
and the focus cell also prints `p_positive` -- the continuous quantity the reading thresholds -- so
a row sitting on the cut cannot be read as settled evidence (`llb.eval.embedder_adoption.stability`).
"""

from collections.abc import Mapping

from llb.eval.embedder_adoption.roster_models import (
    NUMERIC_PROPERTIES,
    PROPERTIES,
    PROPERTY_FAMILY,
    PROPERTY_PARAMS,
    RosterReport,
)
from llb.rag.fusion_evidence.evidence_gate import (
    reading_label,
)
from llb.rag.fusion_evidence.stability import (
    boundary_table,
    format_reading,
)


def _profile_cell(profile: Mapping[str, object], name: str) -> str:
    value = profile.get(name)
    return "-" if value is None else f"{value:g}" if isinstance(value, float) else str(value)


def _boundary_table(report: RosterReport) -> list[str]:
    """How close each model's FOCUS-cell reading sits to the cut that produced it.

    The same table the SWEEP report renders per cell, so the roster and a one-model run place a
    knife-edge row on one shared scale.
    """
    focus = next(cell for cell in report["cells"] if cell["label"] == report["focus_cell"])
    stability = focus.get("stability") or {}
    rows = [
        (f"`{model.split('/')[-1]}`", stability[model])
        for model in report["models"]
        if model in stability
    ]
    return boundary_table(
        rows,
        title=f"How close `{report['focus_cell']}` sits to the cut",
        key_header="model",
        subject=f"`{report['candidate']}`",
    )


def format_roster(
    report: RosterReport,
    *,
    metadata: Mapping[str, object] | None = None,
    title: str = "Embedder adoption bar: is the reranker gain predictable in advance?",
) -> str:
    """The roster Markdown artifact: the verdict, the per-cell readings, the property table."""
    verdict = report["verdict"]
    meta = dict(metadata or {})
    lines = [f"# {title}", ""]
    for key in ("split", "grounding", "goldset", "corpus"):
        if key in meta:
            lines.append(f"- {key}: `{meta[key]}`")
    lines += [
        f"- encoder pair: `{report['candidate']}` vs `{report['baseline']}`",
        f"- scored items: {report['n']} (identical set and seed in every sweep)",
        f"- roster: {len(report['models'])} models",
        f"- focus cell: `{report['focus_cell']}`",
        f"- verdict: **{verdict['decision']}** -- {verdict['reason']}",
        "",
        "### Per-cell reading by model",
        "",
        "| model | "
        + " | ".join(f"`{cell['label']}`" for cell in report["cells"])
        + " | sweep verdict |",
        "| --- | " + " | ".join([":-:"] * len(report["cells"])) + " | :-: |",
    ]
    for model in report["models"]:
        readings = " | ".join(
            format_reading(cell.get("stability", {}).get(model), cell["readings"][model])
            for cell in report["cells"]
        )
        lines.append(f"| `{model}` | {readings} | {report['verdicts'][model]} |")
    lines += [
        "",
        "`answer` = the rank gain reached the answer (objective interval clears zero); "
        "`rank only` = the encoder ranks earlier but the answer does not move; `neither` = no "
        "separation; `insufficient evidence` = the interval clears zero on too few differing items "
        "for the reporting level to be reachable. `(borderline)` marks a reading that a 90% or a "
        "97.5% interval would change -- the cut decided it, not the evidence.",
        "",
    ]
    lines += _boundary_table(report)
    lines += [
        "### Declared model properties",
        "",
        "| model | "
        + " | ".join(f"{name}" for name in PROPERTIES)
        + f" | `{report['focus_cell']}` |",
        # Numeric properties read right-aligned, categorical ones centred.
        "| --- | "
        + " | ".join("---:" if name in NUMERIC_PROPERTIES else ":-:" for name in PROPERTIES)
        + " | :-: |",
    ]
    focus = next(cell for cell in report["cells"] if cell["label"] == report["focus_cell"])
    for model in report["models"]:
        profile = report["profiles"].get(model, {})
        values = " | ".join(_profile_cell(profile, name) for name in PROPERTIES)
        reading = reading_label(focus["readings"][model])
        lines.append(f"| `{model}` | {values} | {reading} |")
    lines += ["", "### Property separation", ""]
    if not verdict["separations"]:
        lines.append("No property was tested: the focus cell has no split to explain.")
    for entry in verdict["separations"]:
        mark = "SEPARATES" if entry["separates"] else "does not separate"
        lines.append(f"- `{entry['property']}` -- **{mark}**: {entry['reason']}")
        if entry["missing"]:
            lines.append(f"  - undeclared for: {', '.join(f'`{m}`' for m in entry['missing'])}")
    lines.append("")
    return "\n".join(lines) + "\n"


def format_roster_summary(report: RosterReport) -> str:
    """One-screen ASCII summary for the terminal."""
    verdict = report["verdict"]
    focus = next(cell for cell in report["cells"] if cell["label"] == report["focus_cell"])
    lines = [
        f"[compare-adoption-roster] n={report['n']} models={len(report['models'])} "
        f"focus={report['focus_cell']}",
        f"  {'model'.ljust(34)} {'params':>7} {'family':>12}   {report['focus_cell']}",
    ]
    for model in report["models"]:
        profile = report["profiles"].get(model, {})
        reading = format_reading(focus.get("stability", {}).get(model), focus["readings"][model])
        lines.append(
            f"  {model.split('/')[-1][:34].ljust(34)} "
            f"{_profile_cell(profile, PROPERTY_PARAMS):>7} "
            f"{_profile_cell(profile, PROPERTY_FAMILY):>12}   {reading}"
        )
    lines.append(f"  verdict: {verdict['decision'].upper()} -- {verdict['reason']}")
    return "\n".join(lines)
