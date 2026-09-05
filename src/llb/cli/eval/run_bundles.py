"""Check one published run bundle against the contracts it declares."""

from pathlib import Path

import typer

from llb.cli.app import app
from llb.cli.artifact_readings import member_line


@app.command("check-run-bundle")
def check_run_bundle(
    run: Path = typer.Argument(..., help="published run bundle directory"),
) -> None:
    """Validate every member of one run bundle, member by member.

    Run it before a board, a study, or an external consumer reads the bundle. The manifest is
    resolved through its contract first -- a pre-contract bundle is read at the version the family
    declares its history to be -- and everything else is then read through what THAT manifest
    declares: the score rows against their contract or their published column set, the retrieval
    sidecar against its row family, and each additional artifact against the contract or the
    human-report exemption it was published under. Every refusal is reported, not only the first.
    """
    from llb.artifacts.errors import ArtifactContractError
    from llb.artifacts.run_bundle.datasets import run_bundle_manifest
    from llb.artifacts.run_bundle.survey import survey_run_bundle

    try:
        manifest = run_bundle_manifest(run)
    except (FileNotFoundError, ArtifactContractError, ValueError) as exc:
        typer.echo(f"[check-run-bundle] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    readings = survey_run_bundle(run, manifest)
    for reading in readings:
        typer.echo(member_line(reading))
    refused = [reading for reading in readings if reading.refusal]
    typer.echo(
        f"[check-run-bundle] {len(readings) - len(refused)}/{len(readings)} member(s) readable "
        f"in {manifest.dataset_id} at {run}"
    )
    if refused:
        raise typer.Exit(code=1)
