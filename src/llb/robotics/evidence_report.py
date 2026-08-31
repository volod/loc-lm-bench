"""Render machine- and human-readable HFlow bridge reports."""

import json
from pathlib import Path

from llb.core.fsutil import atomic_write_text


def write_evidence_report(path: Path, report: dict[str, object]) -> None:
    atomic_write_text(
        path / "report.json",
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    counts = report["admission_counts"]
    assert isinstance(counts, dict)
    lines = [
        "# HFlow robotics evidence bridge report",
        "",
        f"- Verdict: `{report['verdict']}`",
        f"- Projections: {report['projection_count']}",
        f"- Episodes opened with standard MCAP: {report['episode_count']}",
        f"- Corpus fingerprint: `{report['corpus_fingerprint']}`",
        "",
        "## Admission ledger",
        "",
    ]
    for admission, count in sorted(counts.items()):
        lines.append(f"- `{admission}`: {count}")
    atomic_write_text(path / "report.md", "\n".join(lines) + "\n")
