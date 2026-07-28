"""Read-only re-decision of recorded paired artifacts."""

import json
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.core.paths import resolve_data_dir
from llb.core.store_generations import generation_timestamp


@app.command("audit-paired-readings")
def audit_paired_readings_cmd(
    data_dir: Optional[Path] = typer.Option(None, help="artifact root (default: DATA_DIR)"),
    out_dir: Optional[Path] = typer.Option(
        None, help="output directory (default: DATA_DIR/paired-reading-audit/<timestamp>)"
    ),
) -> None:
    """Re-read vector-backed artifacts without inference and name every changed verdict."""
    from llb.rag.paired_reading_audit import audit_paired_readings
    from llb.rag.paired_reading_audit_report import format_audit

    root = resolve_data_dir(data_dir)
    target = out_dir or root / "paired-reading-audit" / generation_timestamp()
    report = audit_paired_readings(root)
    target.mkdir(parents=True, exist_ok=True)
    (target / "audit.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    (target / "report.md").write_text(format_audit(report), encoding="utf-8")
    typer.echo(
        f"[audit-paired-readings] {report['artifacts']} artifacts, "
        f"{report['comparisons']} comparisons, "
        f"{len(report['reading_changes'])} reading changes -> {target / 'report.md'}"
    )
