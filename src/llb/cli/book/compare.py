"""`llb book-ocr compare`: choose a book-ocr config from accepted pilot pages."""

import json
from pathlib import Path

import typer
import yaml

from llb.cli.book import book_app
from llb.core.fsutil import atomic_write_text


@book_app.command("compare")
def compare_cmd(
    manifest: Path = typer.Argument(..., help="comparison manifest"),
    out: Path = typer.Option(..., "--out", help="write the JSON report here"),
    config_out: Path = typer.Option(..., "--config-out", help="write the book-ocr config here"),
) -> None:
    """Score accepted pilot pages and write the report and the selected config.

    Exit 2 when the reference is unaccepted or the baseline language disagrees.
    A comparison where no candidate wins still exits 0: that result is the finding.
    """
    from llb.prep.book.compare.manifest import run_manifest
    from llb.prep.book.compare.summary import summary_lines

    try:
        report, config = run_manifest(manifest)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=True, indent=2) + "\n"
    atomic_write_text(out, payload)
    atomic_write_text(config_out, yaml.safe_dump(config, sort_keys=False))
    typer.echo(f"[book-ocr] compare -> {out}")
    typer.echo("\n".join(summary_lines(report)))
