"""Render an audit as `findings.jsonl` + `report.md` + `summary.json`.

The Markdown is the operator's read: what the corpus looks like, what each tier cost, and the
worst offenders first. `findings.jsonl` is the machine read and the input a resolution lane
consumes -- one JSON object per claim pair, with both sides' exact offsets.
"""

import json
from pathlib import Path

from llb.conflicts.constants import (
    FINDINGS_FILE,
    REL_CONTRADICTS,
    REL_SUPERSEDED_BY,
    REPORT_FILE,
    SUMMARY_FILE,
    TREE_META_FILE,
)
from llb.conflicts.models import AuditResult, Finding

# Findings whose relation means "someone must decide", listed first in the report.
ACTIONABLE = (REL_CONTRADICTS, REL_SUPERSEDED_BY)
_EXCERPT = 160


def _excerpt(text: str) -> str:
    """One-line excerpt safe to drop into a Markdown table cell."""
    flat = " ".join(text.split())
    if len(flat) > _EXCERPT:
        flat = flat[: _EXCERPT - 1].rstrip() + "…"
    return flat.replace("|", "\\|")


def _sort_key(finding: Finding) -> tuple[int, float, str]:
    """Actionable relations first, then by descending score, then stably by claim identity."""
    priority = 0 if finding.relation in ACTIONABLE else 1
    return (priority, -finding.score, str(finding.key()))


def render_report(result: AuditResult) -> str:
    """The operator-facing Markdown report."""
    lines = [
        "# Corpus conflict audit",
        "",
        f"- corpus: `{result.corpus_root}`",
        f"- effort: `{result.effort}`",
        f"- documents: {result.n_docs}",
        f"- findings: {len(result.findings)}",
        "",
    ]
    lines += _relations_section(result)
    lines += _tiers_section(result)
    lines += _needles_section(result)
    lines += _findings_section(result)
    return "\n".join(lines)


def _relations_section(result: AuditResult) -> list[str]:
    counts = result.relation_counts()
    if not counts:
        return ["No conflicting, duplicated, or subsumed claims were found.", ""]
    lines = ["## Relations", "", "| relation | findings |", "| --- | --- |"]
    lines += [f"| `{relation}` | {count} |" for relation, count in counts.items()]
    lines.append("")
    return lines


def _tiers_section(result: AuditResult) -> list[str]:
    if not result.tiers:
        return []
    lines = [
        "## Tiers",
        "",
        "| tier | candidate pairs | findings | seconds |",
        "| --- | --- | --- | --- |",
    ]
    for stat in result.tiers:
        lines.append(
            f"| `{stat.tier}` | {stat.candidate_pairs} | {stat.findings} | {stat.seconds:.2f} |"
        )
    lines.append("")
    semantic = next((s for s in result.tiers if "pruned_fraction" in s.extra), None)
    if semantic is not None:
        extra = semantic.extra
        lines += [
            f"The semantic prefix tree examined {extra.get('pairs_examined')} of "
            f"{extra.get('exhaustive_pairs')} possible chunk pairs "
            f"({float(extra.get('pruned_fraction', 0.0)) * 100:.1f}% pruned) at cosine "
            f">= {extra.get('cos_threshold')}. Pruning is exact: the surviving pairs are the "
            "same ones an all-pairs scan would return.",
            "",
        ]
    return lines


def _needles_section(result: AuditResult) -> list[str]:
    needles = result.needles
    if not needles:
        return []
    lines = [
        "## Needle ambiguity",
        "",
        f"- gold items checked: {needles.get('items')}",
        f"- answerable from more than one document: {needles.get('ambiguous_items')} "
        f"({float(needles.get('non_unique_needle_fraction', 0.0)) * 100:.1f}%)",
    ]
    unlocated = int(needles.get("unlocated_items", 0) or 0)
    if unlocated:
        lines.append(f"- gold spans that matched no chunk (not scored for ambiguity): {unlocated}")
    lines.append("")
    return lines


def _findings_section(result: AuditResult) -> list[str]:
    if not result.findings:
        return []
    lines = [
        "## Findings",
        "",
        "Actionable relations first. Offsets are exact character positions in the source "
        "document; `~` marks a claim whose quote could not be located, where the span falls back "
        "to the enclosing chunk.",
        "",
        "| relation | tier | score | newer | A | B |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for finding in sorted(result.findings, key=_sort_key):
        newer = finding.staleness.newer_side or "-"
        lines.append(
            f"| `{finding.relation}` | `{finding.tier}` | {finding.score:.3f} | {newer} "
            f"| {_side(finding, 'a')} | {_side(finding, 'b')} |"
        )
    lines.append("")
    return lines


def _side(finding: Finding, side: str) -> str:
    ref = finding.a if side == "a" else finding.b
    mark = "" if ref.offsets_exact else "~"
    return f"`{ref.doc_id}`{mark} [{ref.char_start}:{ref.char_end}]<br>{_excerpt(ref.text)}"


def write_audit(out_dir: Path | str, result: AuditResult) -> dict[str, Path]:
    """Persist the three artifacts; returns their paths by name."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    findings_path = out / FINDINGS_FILE
    with findings_path.open("w", encoding="utf-8") as handle:
        for finding in sorted(result.findings, key=_sort_key):
            handle.write(json.dumps(finding.payload(), ensure_ascii=False) + "\n")
    report_path = out / REPORT_FILE
    report_path.write_text(render_report(result), encoding="utf-8")
    summary_path = out / SUMMARY_FILE
    summary_path.write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths = {"findings": findings_path, "report": report_path, "summary": summary_path}
    if result.tree_meta:
        tree_path = out / TREE_META_FILE
        tree_path.write_text(
            json.dumps(result.tree_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths["tree_meta"] = tree_path
    return paths
