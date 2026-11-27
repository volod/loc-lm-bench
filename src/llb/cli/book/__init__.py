"""Register the `llb book-ocr` recovery, pilot, review, and export commands."""

import typer

from llb.cli.app import app

book_app = typer.Typer(
    add_completion=False,
    rich_markup_mode=None,
    no_args_is_help=True,
    help="Scanned-book recovery: a resumable per-page, per-stage run over a scanned PDF.",
)
app.add_typer(book_app, name="book-ocr")

import llb.cli.book.compare  # noqa: E402,F401
import llb.cli.book.engines  # noqa: E402,F401
import llb.cli.book.pilot  # noqa: E402,F401
import llb.cli.book.pilot_review  # noqa: E402,F401
import llb.cli.book.pilot_run  # noqa: E402,F401
import llb.cli.book.project  # noqa: E402,F401
import llb.cli.book.render  # noqa: E402,F401
import llb.cli.book.review_queue  # noqa: E402,F401
import llb.cli.book.run  # noqa: E402,F401
import llb.cli.book.score  # noqa: E402,F401
import llb.cli.book.status  # noqa: E402,F401
import llb.cli.book.structure  # noqa: E402,F401
import llb.cli.book.survey  # noqa: E402,F401
import llb.cli.book.validate  # noqa: E402,F401
