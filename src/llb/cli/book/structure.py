"""`llb book-ocr structure`: printed pages, hierarchy, and the numbering inventory."""

from pathlib import Path

import typer

from llb.cli.book import book_app
from llb.core.contracts.data_prep.book_structure import StructureNode
from llb.prep.book.structure.io import structure_path


@book_app.command("structure")
def structure_cmd(
    run_dir: Path = typer.Argument(
        ...,
        help="a book-ocr run; an unfinished ledger is refused and a terminal failure is included",
    ),
) -> None:
    """Infer printed pages, the hierarchy, and figure, table, and equation numbers."""
    from llb.prep.book.structure.build import build_structure

    try:
        structure = build_structure(run_dir)
    except (OSError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    observed = sum(page.status == "observed" for page in structure.pages)
    gaps = sum(page.status == "gap" for page in structure.pages)
    typer.echo(
        f"[book-ocr] structure {len(structure.pages)} pages, {observed} numbered, {gaps} gaps"
    )
    typer.echo(
        f"[book-ocr] hierarchy {_count(structure.hierarchy)} nodes, "
        f"fallback={str(structure.hierarchy.fallback).lower()}"
    )
    typer.echo(
        f"[book-ocr] inventory {len(structure.inventory)}, conflicts {len(structure.conflicts)}"
    )
    typer.echo(f"[book-ocr] structure -> {structure_path(run_dir)}")


def _count(node: StructureNode) -> int:
    return 1 + sum(_count(child) for child in node.children)
