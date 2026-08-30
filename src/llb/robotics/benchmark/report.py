"""Persist a self-contained operation bundle and render its concise human report."""

import json
from pathlib import Path
from typing import Any

from llb.core.fsutil import atomic_write_text


def _metric(name: str, row: dict[str, Any]) -> str:
    rate = row.get("rate")
    shown = "n/a" if rate is None else f"{float(rate):.3f}"
    return f"| {name} | {row['hits']}/{row['n']} | {shown} |"


def format_report(report: dict[str, Any]) -> str:
    verdict = report["paired_verdict"]
    lines = [
        "# Robotics RAG operation benchmark",
        "",
        f"- Verdict: `{verdict['decision']}`",
        f"- Model: `{report['model']}`",
        f"- Backend: `{report['backend']}`",
        f"- Cases per model lane: {report['design']['minimum_evidence_count']}",
        "- Scope: protocol-neutral emulator only; no physical certification",
        "",
        "## Lane metrics",
        "",
        "| Lane / metric | Count | Rate |",
        "| --- | ---: | ---: |",
    ]
    for lane in ("no_retrieval", "retrieval", "reference"):
        metrics = report["lanes"][lane]
        for key in ("task_completion", "appropriate_refusal", "operational_success"):
            lines.append(_metric(f"{lane} / {key}", metrics[key]))
    lines += [
        "",
        "## Paired retrieval reading",
        "",
        "| Metric | Delta | Interval | W/L/T |",
        "| --- | ---: | --- | ---: |",
    ]
    for key in ("task_completion", "appropriate_refusal", "operational_success"):
        metric = verdict[key]
        interval = metric["interval"]
        lines.append(
            f"| {key} | {metric['delta']:.3f} | [{interval[0]:.3f}, {interval[1]:.3f}] | "
            f"{metric['wins']}/{metric['losses']}/{metric['ties']} |"
        )
    lines += [
        "",
        f"Mandatory safety gate: `{verdict['mandatory_safety_gate_passed']}`. ",
        f"Unsafe proposal regression: `{verdict['unsafe_proposal_regression']}`. ",
        "The finite emulator ledger is not a safety proof or hardware authorization.",
        "",
    ]
    return "\n".join(lines)


def write_bundle(
    run_dir: Path,
    report: dict[str, Any],
    lane_rows: dict[str, list[dict[str, Any]]],
) -> None:
    for lane, rows in lane_rows.items():
        atomic_write_text(
            run_dir / "transcripts" / f"{lane}.jsonl",
            "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        )
    atomic_write_text(
        run_dir / "report.json",
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(run_dir / "report.md", format_report(report))
