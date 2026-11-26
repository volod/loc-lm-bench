"""`llb book-ocr score`: quality metrics against a reference transcription."""

import json
from pathlib import Path

import typer

from llb.cli.book import book_app
from llb.core.fsutil import atomic_write_text


@book_app.command("score")
def score_cmd(
    run_dir: Path | None = typer.Argument(None, help="a book-ocr run directory"),
    reference: Path = typer.Option(..., "--reference", help="reference transcription directory"),
    citations: Path | None = typer.Option(
        None, "--citations", help="existing PDF lane <doc>.citations.json beside its markdown"
    ),
    engine: str | None = typer.Option(None, "--engine", help="score this ledger engine only"),
    acceptance: Path | None = typer.Option(
        None,
        "--acceptance",
        help="verify a finalized pilot reference against this acceptance record",
    ),
    out: Path | None = typer.Option(None, "--out", help="write the JSON report to this path"),
    as_json: bool = typer.Option(False, "--json", help="print the JSON report"),
) -> None:
    """Report per-page and per-class recovery metrics. Writes nothing unless --out is set."""
    from llb.prep.book.metrics.score import score_reference, summary_lines

    if run_dir is None and citations is None:
        typer.echo("[error] score needs a run directory or --citations", err=True)
        raise typer.Exit(code=2)
    try:
        if acceptance is not None:
            from llb.prep.book.config import read_config
            from llb.prep.book.metrics.reference import verify_reference_digest

            source = read_config(run_dir).source.sha256 if run_dir is not None else None
            verify_reference_digest(reference, acceptance, source)
        report = score_reference(reference, run_dir, citations, engine)
    except (OSError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=True, indent=2) + "\n"
    if out is not None:
        atomic_write_text(out, payload)
        typer.echo(f"[book-ocr] score -> {out}")
    if as_json:
        typer.echo(payload)
    else:
        typer.echo("\n".join(summary_lines(report)))
