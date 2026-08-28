"""Generation-swap invalidation command: what a proposed adoption makes stale, listed in advance."""

from pathlib import Path

import typer

from llb.cli.app import app
from llb.cli.helpers import cli_error
from llb.cli.models.families import DEFAULT_MANIFEST


@app.command("report-generation-invalidation")
def report_generation_invalidation_cmd(
    family: str = typer.Argument(..., help="family id whose generation is being swapped"),
    generation: str = typer.Argument(..., help="the generation proposed for adoption"),
    manifest: Path = typer.Option(DEFAULT_MANIFEST, help="candidate-models YAML manifest"),
    root: Path = typer.Option(
        None, help="repo root the committed evidence and delivered docs are read from"
    ),
    as_json: bool = typer.Option(False, "--json", help="print the report as JSON"),
    strict: bool = typer.Option(
        False, help="exit non-zero when the swap invalidates anything (for automation)"
    ),
) -> None:
    """List every committed aggregate, published value, and baseline row a swap would invalidate."""
    from llb.backends.invalidation import render_json, render_text, report_invalidation
    from llb.backends.roster import load_register
    from llb.core.paths import PROJECT_ROOT

    try:
        register = load_register(manifest)
    except ValueError as exc:
        cli_error(str(exc))

    try:
        report = report_invalidation(register, family, generation, root=root or PROJECT_ROOT)
    except ValueError as exc:
        cli_error(str(exc))

    typer.echo(render_json(report) if as_json else render_text(report))
    if strict and report.invalidated:
        raise typer.Exit(code=1)
