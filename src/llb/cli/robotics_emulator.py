"""CLI entry point for the deterministic robotics device emulator."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import cli_error
from llb.core.paths import PROJECT_ROOT

DEFAULT_FIXTURE = PROJECT_ROOT / "samples" / "robotics" / "emulator" / "scenarios.json"


@app.command("robotics-emulator-check")
def robotics_emulator_check(
    fixture: Path = typer.Option(DEFAULT_FIXTURE, help="Committed emulator scenario fixture."),
    data_dir: Optional[Path] = typer.Option(
        None, help="Artifact root (default: DATA_DIR from the project environment)."
    ),
) -> None:
    """Replay the side-effect gate, faults, locks, and no-retry scenarios."""
    from llb.robotics.emulator_run import run_emulator_check

    try:
        output_dir, report = run_emulator_check(fixture, data_dir)
    except ValueError as exc:
        cli_error(str(exc), code=1)
    typer.echo(
        f"[ok] robotics emulator {report['scenario_count']} scenarios, "
        f"{report['forbidden_adapter_invocations']} forbidden invocations -> {output_dir}"
    )
