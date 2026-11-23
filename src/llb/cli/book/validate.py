"""`llb book-ocr validate`: prove a rendered vault and write its processing report."""

from pathlib import Path

import typer

from llb.cli.book import book_app

_EXIT_FAILED = 1
_EXIT_REFUSED = 2


@book_app.command("validate")
def validate_cmd(
    book_dir: Path = typer.Argument(..., help="a rendered reading vault"),
    run_dir: Path | None = typer.Option(
        None, "--run", help="book-ocr run, when the vault was published outside it"
    ),
) -> None:
    """Check links, pages, assets, and markers. Write the processing report."""
    from llb.prep.book.constants import METADATA_DIRNAME, PROCESSING_REPORT_FILENAME
    from llb.prep.book.validate import validate_book

    try:
        report = validate_book(book_dir, run_dir)
    except (OSError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=_EXIT_REFUSED) from exc
    state = "ok" if report.ok else "failed"
    report_path = book_dir / METADATA_DIRNAME / PROCESSING_REPORT_FILENAME
    typer.echo(f"[book-ocr] validate {state} -> {report_path}")
    for finding in report.findings:
        typer.echo(f"[book-ocr] {finding.file}: {finding.check}: {finding.detail}", err=True)
    if not report.ok:
        raise typer.Exit(code=_EXIT_FAILED)
