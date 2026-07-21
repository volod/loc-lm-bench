"""Unified human-review workbench command."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import cli_error


@app.command("review")
def review_cmd(
    path: Path = typer.Argument(..., help="existing review ledger or run directory"),
    start: Optional[int] = typer.Option(None, min=1, help="one-based record to open"),
) -> None:
    """Auto-detect an existing ledger and open the unified Textual workbench."""
    try:
        from llb.review.workbench import run_workbench
    except ImportError:
        cli_error('review workbench unavailable; install with: uv pip install -e ".[review]"')
    try:
        adapter = run_workbench(path, start=start)
    except (OSError, ValueError) as exc:
        cli_error(str(exc))
    progress = adapter.progress(max(0, min((start or 1) - 1, len(adapter) - 1)))
    typer.echo(f"[review] {adapter.kind}: {progress.reviewed}/{progress.total} reviewed -> {path}")
