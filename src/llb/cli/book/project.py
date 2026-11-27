"""`llb book-ocr project`: a retrieval corpus from a run's ledger."""

from pathlib import Path

import typer

from llb.cli.book import book_app
from llb.prep.book.render.constants import SECTION_CHAR_CAP


@book_app.command("project")
def project_cmd(
    run_dir: Path = typer.Argument(..., help="a book-ocr run with a document structure"),
    out: Path = typer.Option(..., "--out", help="corpus directory to write"),
    licence: str = typer.Option(
        ...,
        "--licence",
        help="redistribution class the operator declares: redistributable or local-only",
    ),
    language: str | None = typer.Option(
        None, "--language", help="document language; detected from the text when omitted"
    ),
    section_cap: int = typer.Option(
        SECTION_CHAR_CAP,
        "--section-cap",
        min=200,
        help="split a section past this many characters",
    ),
    slug: str | None = typer.Option(
        None, "--slug", help="ASCII book name used in vault file paths; default is the PDF name"
    ),
    reference_grammar: Path | None = typer.Option(
        None,
        "--reference-grammar",
        help="YAML grammar that replaces the handbook cues; the vault uses the same file",
    ),
) -> None:
    """Project the ledger for retrieval. Prints counts and no recovered text."""
    from llb.prep.book.project.build import project_corpus

    try:
        result = project_corpus(
            run_dir,
            out,
            licence,
            language=language,
            section_cap=section_cap,
            slug=slug,
            reference_grammar=reference_grammar,
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"[book-ocr] project {len(result.documents)} document(s), "
        f"{result.revisions} revision(s) -> {result.out_dir}"
    )
