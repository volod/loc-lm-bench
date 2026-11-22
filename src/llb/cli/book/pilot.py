"""`llb book-ocr pilot-select`: a seeded pilot from a run's class stage."""

from collections.abc import Callable
from pathlib import Path

import typer

from llb.cli.book import book_app
from llb.core.contracts.data_prep.book import BookOcrRunConfig
from llb.core.contracts.data_prep.book_classes import (
    PAGE_CLASS_IDS,
    PAGE_CLASS_LABELS,
    PageClassRecord,
)
from llb.prep.book.classes.constants import PILOT_DIRNAME, SELECTION_FILENAME


@book_app.command("pilot-select")
def pilot_select_cmd(
    run_dir: Path = typer.Argument(..., help="a book-ocr run whose classes stage has finished"),
    size: int = typer.Option(25, "--size", min=1, help="how many pages to select"),
    seed: int = typer.Option(0, "--seed", min=0, help="selection seed"),
    out: Path | None = typer.Option(None, "--out", help="also write the selection to this path"),
) -> None:
    """Write a reproducible pilot. Each page records the class it represents and the signals that placed it."""
    from llb.prep.book.classes.records import write_selection
    from llb.prep.book.classes.select import select_pilot
    from llb.prep.book.config import read_config
    from llb.prep.book.ranges import page_id

    try:
        config = read_config(run_dir)
        records = _finished(run_dir, config, page_id)
        selection = select_pilot(records, source_sha256=config.source.sha256, seed=seed, size=size)
    except (OSError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    destination = run_dir / PILOT_DIRNAME / SELECTION_FILENAME
    write_selection(destination, selection)
    if out is not None:
        write_selection(out, selection)
    typer.echo(
        f"[book-ocr] pilot {len(selection.pages)} pages, seed {selection.seed} -> {destination}"
    )
    for name in PAGE_CLASS_IDS:
        label = PAGE_CLASS_LABELS[name]
        if label in selection.empty_classes:
            typer.echo(f"[book-ocr]   {label}: no qualifying page")
        elif label in selection.unfilled_classes:
            typer.echo(f"[book-ocr]   {label}: not filled")
        else:
            typer.echo(f"[book-ocr]   {label}: {selection.coverage[name]}")


def _finished(
    run_dir: Path, config: BookOcrRunConfig, page_id: Callable[[int, int], str]
) -> list[PageClassRecord]:
    from llb.prep.book.classes.constants import CLASSES_FILENAME
    from llb.prep.book.classes.records import read_classes
    from llb.prep.book.constants import PAGES_DIRNAME

    records = []
    missing = []
    for pdf_page in config.pages:
        pid = page_id(pdf_page, config.source.page_count)
        path = run_dir / PAGES_DIRNAME / pid / CLASSES_FILENAME
        if not path.is_file():
            missing.append(pid)
            continue
        records.append(read_classes(path))
    if missing:
        shown = ", ".join(missing[:8])
        extra = f" (+{len(missing) - 8})" if len(missing) > 8 else ""
        raise ValueError(f"classes stage has not finished for {shown}{extra}")
    return records
