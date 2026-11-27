"""Prepare, review, and publish a frozen pilot reference."""

from pathlib import Path

import typer

from llb.cli.book import book_app

review_app = typer.Typer(no_args_is_help=True, help="Human transcription of a frozen book pilot.")
book_app.add_typer(review_app, name="pilot-review")


@review_app.command("prepare")
def prepare_cmd(
    source: Path = typer.Option(..., "--source", help="original scanned PDF"),
    pilot: Path = typer.Option(..., "--pilot", help="frozen pilot_pages.json"),
    primary: Path = typer.Option(..., "--primary", help="layout-rich recognition run"),
    alternate: Path = typer.Option(..., "--alternate", help="second recognition engine run"),
    alternate_override: Path | None = typer.Option(
        None, "--alternate-override", help="same second engine with a better variant on a subset"
    ),
    output: Path | None = typer.Option(None, "--output", help="review bundle directory"),
) -> None:
    """Create a blind review bundle with a scan and an immutable pre-draft for each page."""
    from llb.prep.book.pilot_review.prepare import DraftInput, prepare

    target = output if output is not None else primary / "pilot" / "reference-review"
    try:
        prepare(DraftInput(source, pilot, primary, alternate, alternate_override), target)
    except (OSError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"[book-ocr] pilot review -> {target}")


@review_app.command("open")
def open_cmd(
    bundle: Path = typer.Argument(..., help="pilot review bundle"),
    port: int = typer.Option(
        8765, "--port", min=0, max=65535, help="local port (0 picks a free port)"
    ),
) -> None:
    """Open the local worksheet server for a human reviewer."""
    from llb.prep.book.pilot_review.server import serve
    from llb.prep.book.pilot_review.session import Review

    try:
        serve(Review(bundle), port)
    except KeyboardInterrupt:
        return
    except (OSError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc


@review_app.command("status")
def status_cmd(bundle: Path = typer.Argument(..., help="pilot review bundle")) -> None:
    """Show latest saved decisions without opening the worksheet."""
    from llb.prep.book.pilot_review.session import Review

    try:
        rows = Review(bundle).status()
    except (OSError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    for row in rows:
        label = "accepted" if row["accepted"] else "pending"
        typer.echo(f"{row['id']} {row['page_class']}: {label} (revision {row['version']})")
    count = sum(row["accepted"] for row in rows)
    typer.echo(f"[book-ocr] {count}/{len(rows)} pilot pages accepted")


@review_app.command("finalize")
def finalize_cmd(bundle: Path = typer.Argument(..., help="pilot review bundle")) -> None:
    """Write the scoreable reference and its checksum and reviewer record."""
    from llb.prep.book.pilot_review.session import Review

    try:
        target = Review(bundle).finalize()
    except (OSError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"[book-ocr] reference -> {target}")
