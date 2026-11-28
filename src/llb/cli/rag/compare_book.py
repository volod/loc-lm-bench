"""Compare vector retrieval over a book projection and the existing PDF lane."""

from pathlib import Path

import typer

from llb.cli.app import app
from llb.cli.helpers import cli_error, load_config
from llb.core.paths import resolve_project_path
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED


@app.command("compare-book-retrieval")
def compare_book_retrieval(
    projection: Path = typer.Option(..., help="projected book corpus with PDF citations"),
    baseline: Path = typer.Option(..., help="pdf-to-markdown output for the same single PDF"),
    goldset: Path = typer.Option(..., help="verified gold JSONL grounded in the projection"),
    mde: float = typer.Option(..., min=0.0, max=1.0, help="declared material metric delta"),
    config: Path | None = typer.Option(None, help="shared vector index and retrieval settings"),
    k: int | None = typer.Option(None, min=1, help="retrieval cutoff (defaults to config top_k)"),
    resamples: int = typer.Option(DEFAULT_RESAMPLES, min=1, help="paired bootstrap replicates"),
    confidence: float = typer.Option(DEFAULT_CONFIDENCE, min=0.501, max=0.999),
    seed: int = typer.Option(DEFAULT_SEED, help="paired bootstrap seed"),
) -> None:
    """Report paired gold span and source-PDF page recall and MRR."""
    from llb.rag.book_comparison.metrics import ReadingSettings
    from llb.rag.book_comparison.run import BookRetrievalRun, run_book_comparison

    if not 0 < mde <= 1:
        cli_error("mde must be greater than zero and at most one")
    cfg = load_config(resolve_project_path(config) if config is not None else None)
    inputs = BookRetrievalRun(
        resolve_project_path(projection),
        resolve_project_path(baseline),
        resolve_project_path(goldset),
    )
    settings = ReadingSettings(k or cfg.top_k, mde, confidence, resamples, seed)
    try:
        out_dir, slices = run_book_comparison(inputs, cfg, settings)
    except (OSError, ValueError) as exc:
        cli_error(str(exc))
    typer.echo(f"[compare-book-retrieval] k={settings.k} mde={mde:g}")
    for label, reading in slices.items():
        typer.echo(f"[compare-book-retrieval] {label} n={reading['n']} {reading['verdict']}")
    typer.echo(f"[compare-book-retrieval] report -> {out_dir / 'report.json'}")
