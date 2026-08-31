"""CLI entry point for the offline robotics boundary contract check."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import cli_error
from llb.core.paths import PROJECT_ROOT

DEFAULT_FIXTURE_DIR = PROJECT_ROOT / "samples" / "robotics" / "contracts"


@app.command("robotics-contract-check")
def robotics_contract_check(
    fixture_dir: Path = typer.Option(
        DEFAULT_FIXTURE_DIR,
        help="Pinned, network-free robotics fixture directory.",
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        help="Artifact root (default: DATA_DIR from the project environment).",
    ),
) -> None:
    """Validate upstream pins and replay the protocol-neutral fake boundary."""
    from llb.robotics.check import run_contract_check

    try:
        output_dir, report = run_contract_check(fixture_dir, data_dir)
    except ValueError as exc:
        cli_error(str(exc), code=1)
    typer.echo(f"[ok] robotics contract {report['compatibility_label']} -> {output_dir}")
