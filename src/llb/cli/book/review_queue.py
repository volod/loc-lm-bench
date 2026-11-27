"""`llb book-ocr review-queue`: markers, crops, the quality manifest, and the review queue."""

from collections import Counter
from pathlib import Path

import typer

from llb.cli.book import book_app
from llb.core.contracts.data_prep.book_review import COVERAGE_STATUSES
from llb.prep.book.review.constants import MARKER_KIND_ORDER
from llb.prep.book.review.io import manifest_path, queue_path


@book_app.command("review-queue")
def review_queue_cmd(
    run_dir: Path = typer.Argument(
        ...,
        help="a finished book-ocr run; a terminal page failure is included",
    ),
) -> None:
    """Write the queue for a finished run. An unfinished page is refused."""
    from llb.prep.book.review.build import build_review_queue

    try:
        queue, manifest = build_review_queue(run_dir)
    except (OSError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    crops = sum(1 for entry in queue.entries if entry.crop)
    typer.echo(f"[book-ocr] review queue {len(queue.entries)} entries, {crops} crops")
    counts: dict[str, int] = {}
    for entry in queue.entries:
        counts[entry.kind] = counts.get(entry.kind, 0) + 1
    for kind in MARKER_KIND_ORDER:
        if kind in counts:
            typer.echo(f"[book-ocr]   {kind}: {counts[kind]}")
    coverage = Counter(page.coverage for page in manifest.pages)
    for name in COVERAGE_STATUSES:
        if coverage[name]:
            typer.echo(f"[book-ocr]   coverage {name}: {coverage[name]}")
    typer.echo(f"[book-ocr] queue -> {queue_path(run_dir)}")
    typer.echo(f"[book-ocr] manifest -> {manifest_path(run_dir)}")
