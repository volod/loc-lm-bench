"""Focused position probe report implementation."""

import json

from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.eval.position_probe import POSITION_HEAD, POSITION_TAIL, ProbeReport
from llb.rag.fusion_evidence.evidence_gate import READING_FLAT, READING_SEPARATED

PROBE_JSON = "probe.json"


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


def probe_reading(report: ProbeReport) -> str:
    """`separated` when the head and tail CIs clear each other, `flat` when they overlap.

    The same cut the recommendation note already states in prose, named in the repository's shared
    reading vocabulary so a consumer does not have to read the sentence to learn what was decided.
    """
    by_pos = {p.position: p for p in report.positions}
    head, tail = by_pos.get(POSITION_HEAD), by_pos.get(POSITION_TAIL)
    if head is None or tail is None or not head.ci or not tail.ci:
        return READING_FLAT
    overlap = head.ci[0] <= tail.ci[1] and tail.ci[0] <= head.ci[1]
    return READING_FLAT if overlap else READING_SEPARATED


def probe_payload(report: ProbeReport) -> JsonObject:
    """The probe summary a machine consumer reads: what was probed, what each position scored,
    what is recommended, and how the head-versus-tail comparison read."""
    return {
        "model": report.model,
        "backend": report.backend,
        "k": report.k,
        "n_items": report.n_items,
        "skipped": dict(report.skipped),
        "positions": [
            {
                "position": p.position,
                "n": p.n,
                "mean_objective": p.mean_score,
                "ci": list(p.ci) if p.ci else None,
            }
            for p in report.positions
        ],
        "recommendation": report.recommendation,
        "recommendation_note": report.recommendation_note,
        "verdict": probe_reading(report),
        "retrieval_fingerprint": report.retrieval_fingerprint,
    }


def write_probe(report: ProbeReport, out_dir: Path) -> dict[str, str]:
    """Persist `report.md` + `cases.jsonl` + `probe.json` under the probe run dir; returns paths.

    `probe.json` exists because the recommendation is consumed by other reports (the composed agent
    operating profile reads it): re-deriving a decision by parsing a prose sentence out of the
    Markdown would make the report's wording load-bearing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    cases_path = out_dir / "cases.jsonl"
    probe_path = out_dir / PROBE_JSON
    report_path.write_text(render_report(report), encoding="utf-8")
    probe_path.write_text(
        json.dumps(probe_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with cases_path.open("w", encoding="utf-8") as fh:
        for row in report.rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"report": str(report_path), "cases": str(cases_path), "probe": str(probe_path)}
