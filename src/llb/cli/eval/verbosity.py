"""Verbosity-sensitivity study over finalized RAG run bundles."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


@app.command("analyze-verbosity")
def analyze_verbosity_cmd(
    run_dir: list[Path] = typer.Option(
        ..., "--run-dir", help="run-eval bundle; repeat once per model on the fixed item set"
    ),
    out_dir: Optional[Path] = typer.Option(
        None, help="output dir (default: $DATA_DIR/verbosity-sensitivity/<timestamp>)"
    ),
) -> None:
    """Compare F1, fact coverage, found-rate, and the declared answer-format policy."""
    from llb.bench.common import new_run_timestamp
    from llb.core.paths import resolve_data_dir
    from llb.eval.verbosity_sensitivity import analyze, render, write

    try:
        report = analyze(run_dir)
        if out_dir is None:
            _, stamp = new_run_timestamp()
            out_dir = resolve_data_dir() / "verbosity-sensitivity" / stamp
        paths = write(report, out_dir)
    except (OSError, ValueError) as exc:
        typer.echo(f"[analyze-verbosity] error: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(render(report))
    typer.echo(f"[analyze-verbosity] report -> {paths['report']}")
    typer.echo(f"[analyze-verbosity] JSON   -> {paths['json']}")
