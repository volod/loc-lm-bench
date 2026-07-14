"""Focused position probe report implementation."""

import json

from pathlib import Path

from llb.eval.position_probe import ProbeReport


def render_report(report: ProbeReport) -> str:
    """ASCII Markdown report (AGENTS.md: no box-drawing, no emojis)."""
    lines = [
        "# Context-position probe (lost-in-the-middle)",
        "",
        f"- model: `{report.model}` (backend: {report.backend})",
        f"- k: {report.k} (gold chunk at head/middle/tail among real retrieved distractors)",
        f"- items probed: {report.n_items}"
        + (f" (skipped: {report.skipped})" if report.skipped else ""),
        "",
        "| position | n | mean objective | 95% CI |",
        "| --- | --- | --- | --- |",
    ]
    for p in report.positions:
        ci = f"[{p.ci[0]:.3f}, {p.ci[1]:.3f}]" if p.ci else "n/a"
        lines.append(f"| {p.position} | {p.n} | {p.mean_score:.3f} | {ci} |")
    lines += [
        "",
        f"Recommended `context_order` for `{report.model}`: **{report.recommendation}**",
        f"({report.recommendation_note})",
        "",
    ]
    return "\n".join(lines)


def write_probe(report: ProbeReport, out_dir: Path) -> dict[str, str]:
    """Persist `report.md` + `cases.jsonl` under the probe run dir; returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    cases_path = out_dir / "cases.jsonl"
    report_path.write_text(render_report(report), encoding="utf-8")
    with cases_path.open("w", encoding="utf-8") as fh:
        for row in report.rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"report": str(report_path), "cases": str(cases_path)}
