"""CLI entry points for offline and upstream HFlow evidence bridge checks."""

from pathlib import Path
from typing import Optional

import typer

from llb.bench.common import new_run_timestamp
from llb.cli.app import app
from llb.cli.helpers import cli_error
from llb.core.paths import PROJECT_ROOT, resolve_data_dir

DEFAULT_HFLOW_FIXTURE = PROJECT_ROOT / "samples" / "robotics" / "hflow"


@app.command("robotics-evidence-bridge")
def robotics_evidence_bridge(
    fixture_dir: Path = typer.Option(
        DEFAULT_HFLOW_FIXTURE,
        help="Pinned, network-free HFlow projection fixture.",
    ),
    data_dir: Optional[Path] = typer.Option(None, help="Artifact root (default: DATA_DIR)."),
) -> None:
    """Replay a pinned HFlow Parquet manifest into the canonical text corpus."""
    from llb.robotics.evidence_bridge import run_evidence_bridge

    try:
        output_dir, report = run_evidence_bridge(fixture_dir, data_dir=data_dir)
    except (RuntimeError, ValueError) as exc:
        cli_error(str(exc), code=1)
    typer.echo(f"[ok] robotics evidence {report['verdict']} -> {output_dir}")


@app.command("robotics-hflow-integration")
def robotics_hflow_integration(
    data_dir: Optional[Path] = typer.Option(None, help="Artifact root (default: DATA_DIR)."),
    export_fixture: Optional[Path] = typer.Option(
        None,
        help="Optional new destination for the portable fixture.",
    ),
) -> None:
    """Run pinned HFlow app.test()/curate and replay the resulting bridge fixture."""
    from llb.robotics.hflow_integration import run_hflow_integration

    _run_id, run_timestamp = new_run_timestamp()
    output_dir = resolve_data_dir(data_dir) / "robotics-evidence" / run_timestamp
    try:
        report = run_hflow_integration(output_dir, export_fixture)
    except (RuntimeError, ValueError) as exc:
        cli_error(str(exc), code=1)
    typer.echo(f"[ok] pinned HFlow integration {report['verdict']} -> {output_dir}")
