"""`llb book-ocr engines`: which ledger engines can run on this host."""

import typer

from llb.cli.book import book_app


@book_app.command("engines")
def engines_cmd() -> None:
    """List each ledger engine, the stages it serves, and whether it can run here."""
    from llb.prep.book.engines.registry import describe_engines

    for line in describe_engines():
        typer.echo(line)
