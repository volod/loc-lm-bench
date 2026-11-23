"""`llb book-ocr render`: an Obsidian vault from a run's ledger and structure."""

from pathlib import Path

import typer

from llb.cli.book import book_app
from llb.prep.book.constants import SOURCE_POLICY_SYMLINK
from llb.prep.book.render.constants import SECTION_CHAR_CAP


@book_app.command("render")
def render_cmd(
    run_dir: Path = typer.Argument(..., help="a book-ocr run with a document structure"),
    output: Path | None = typer.Option(
        None, "--output", help="vault directory; default book/<slug> inside the run"
    ),
    source_policy: str = typer.Option(
        SOURCE_POLICY_SYMLINK, "--source-policy", help="copy, symlink, or reference"
    ),
    page_fallback: bool = typer.Option(
        False, "--page-fallback", help="also link a rendered image of each PDF page"
    ),
    section_cap: int = typer.Option(
        SECTION_CHAR_CAP, "--section-cap", min=200, help="split a section past this many characters"
    ),
    slug: str | None = typer.Option(
        None, "--slug", help="ASCII vault name; default is the PDF name"
    ),
) -> None:
    """Render the reading vault. Prints the vault path and no recovered text."""
    from llb.prep.book.render.build import RenderOptions, build_vault

    try:
        dest = build_vault(
            run_dir,
            RenderOptions(
                output=output,
                source_policy=source_policy,
                page_fallback=page_fallback,
                section_cap=section_cap,
                slug=slug,
            ),
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"[book-ocr] vault -> {dest}")
    _validate(dest)


def _validate(dest: Path) -> None:
    from llb.prep.book.validate import validate_book

    try:
        report = validate_book(dest)
    except (OSError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    state = "ok" if report.ok else "failed"
    typer.echo(f"[book-ocr] validate {state}")
    for finding in report.findings:
        typer.echo(f"[book-ocr] {finding.file}: {finding.check}: {finding.detail}", err=True)
    if not report.ok:
        raise typer.Exit(code=1)
