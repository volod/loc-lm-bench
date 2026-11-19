"""`llb book-ocr status`: per-page, per-stage state of a run, read from its stage records."""

import json
from pathlib import Path

import typer

from llb.cli.book import book_app


def _cell(state: object) -> str:
    status = getattr(state, "status")
    if getattr(state, "reused_from"):
        return f"{status}(reused)"
    if status == "failed":
        return f"failed({getattr(state, 'attempts')}: {getattr(state, 'error')})"
    return str(status)


@book_app.command("status")
def status_cmd(
    run_dir: Path = typer.Argument(..., help="a book-ocr run directory"),
    pages_only: str = typer.Option(
        "all", "--show", help="which pages to list: all | open (any stage not done) | none"
    ),
    as_json: bool = typer.Option(False, "--json", help="print the refreshed page manifest as JSON"),
) -> None:
    """Report every page's stage states and the per-stage totals; writes nothing."""
    from llb.prep.book.config import read_config
    from llb.prep.book.ledger import refresh_manifest
    from llb.prep.book.stages import default_registry

    try:
        config = read_config(run_dir)
        manifest = refresh_manifest(run_dir, default_registry(), config)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    if as_json:
        typer.echo(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return
    if pages_only != "none":
        for page in manifest.pages:
            if pages_only == "open" and all(s.status == "done" for s in page.stages):
                continue
            cells = " ".join(f"{s.stage}={_cell(s)}" for s in page.stages)
            typer.echo(f"{page.page_id} pdf-page={page.pdf_page} batch={page.batch} {cells}")
    typer.echo(
        f"[book-ocr] run {manifest.run_id}: {manifest.n_pages} page(s), "
        f"source sha256 {manifest.source_sha256[:16]}"
    )
    for name, summary in manifest.summary.items():
        typer.echo(f"[book-ocr]   {name}: " + ", ".join(f"{k} {v}" for k, v in summary.items()))
