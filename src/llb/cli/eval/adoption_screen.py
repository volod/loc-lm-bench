"""Recorded-sweep screen command for embedder adoption."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


@app.command("compare-adoption-screen")
def compare_adoption_screen_cmd(
    reports: list[Path] = typer.Argument(
        ..., help="finished sweep comparison.json files (or their directories) to study"
    ),
    focus_cell: Optional[str] = typer.Option(
        None, "--focus-cell", help="cell the screen must reproduce (default: k10+rerank)"
    ),
    sizes: Optional[str] = typer.Option(
        None,
        "--sizes",
        help="comma-separated screen item counts to measure (default: 10,15,20,25,30,35)",
    ),
    draws: Optional[int] = typer.Option(None, "--draws", help="subsamples drawn per size"),
    target: Optional[float] = typer.Option(
        None,
        "--target",
        min=0.5,
        max=1.0,
        help="agreement a size must reach to count as a usable screen (default 0.90)",
    ),
    seed: Optional[int] = typer.Option(None, help="seed for the subsample draws"),
    out_dir: Optional[Path] = typer.Option(
        None, help="artifact dir (default: beside the FIRST report, in a `screen/` sibling)"
    ),
) -> None:
    """How cheaply can the reranker question be decided for ONE model?

    The reranker question lives in a single cell, so scoring only that cell is an exact 4x saving
    over the full grid. Cutting the ITEM set is not free: a smaller ledger can either lose or invent
    a calibrated separation. This resamples the recorded per-item deltas at a range of item counts
    and reports, per model, how
    often a screen that size reproduces the full-set reading -- ending in the honest floor on what
    a per-model answer costs. Reads finished sweeps only; no backend or GPU.
    """
    from llb.eval.embedder_adoption.screen_report import format_screen_summary
    from llb.eval.embedder_adoption.comparison_run import run_screen_study_over_paths

    options: dict[str, object] = {}
    if sizes:
        try:
            options["sizes"] = [int(token) for token in sizes.split(",") if token.strip()]
        except ValueError:
            typer.echo("[error] --sizes must be comma-separated integers", err=True)
            raise typer.Exit(code=2) from None
    for name, value in (("draws", draws), ("target", target), ("seed", seed)):
        if value is not None:
            options[name] = value
    target_dir = out_dir if out_dir is not None else reports[0].resolve().parent / "screen"
    try:
        run = run_screen_study_over_paths(
            reports, out_dir=target_dir, focus_cell=focus_cell, **options
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(format_screen_summary(run.report))
    typer.echo(f"[compare-adoption-screen] report -> {run.paths['report']}")
