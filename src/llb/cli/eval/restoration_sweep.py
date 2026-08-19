"""CLI for the restoration constraint threshold sweep."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.eval.query_robustness_variants import parse_variant_classes
from llb.eval.restoration_sweep import SWEEP_VARIANT_CLASSES, policy_grid
from llb.rag.query_prep.restore_policy import (
    AMBIGUOUS_TOKEN_MAX_CHARS,
    RESTORATION_RANK_ORDERS,
    SURFACE_MAX_DISTANCE,
)

_DEFAULT_SURFACE_VALUES = "0,1"
_DEFAULT_CUTOFF_VALUES = "3,4,5"


def _integers(spec: str, option: str) -> list[int]:
    try:
        return [int(part) for part in spec.split(",") if part.strip()]
    except ValueError:
        raise typer.BadParameter(f"{option} takes comma-separated integers, got {spec!r}") from None


def _ranks(spec: str) -> list[str]:
    values = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [value for value in values if value not in RESTORATION_RANK_ORDERS]
    if unknown:
        raise typer.BadParameter(
            f"unknown rank order(s) {unknown}; choose from {list(RESTORATION_RANK_ORDERS)}"
        )
    return values


@app.command("sweep-restoration-constraints")
def sweep_restoration_constraints_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="verified gold set JSONL"),
    corpus_root: Optional[Path] = typer.Option(None, help="matching indexed corpus"),
    split: str = typer.Option("final", help="verified gold split to probe"),
    limit: Optional[int] = typer.Option(None, help="optional item cap"),
    seed: Optional[int] = typer.Option(None, help="deterministic noise seed"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="retrieved chunks per query"),
    typo_rate: float = typer.Option(0.08, help="share of eligible characters replaced"),
    variant_classes: Optional[str] = typer.Option(
        None,
        "--variant-classes",
        help=f"comma-separated noise classes; default {','.join(SWEEP_VARIANT_CLASSES)}",
    ),
    surface_distances: str = typer.Option(
        _DEFAULT_SURFACE_VALUES,
        help=f"swept surface-compatibility budgets (default constant: {SURFACE_MAX_DISTANCE})",
    ),
    short_cutoffs: str = typer.Option(
        _DEFAULT_CUTOFF_VALUES,
        help=f"swept short-token cutoffs (default constant: {AMBIGUOUS_TOKEN_MAX_CHARS})",
    ),
    rank_orders: str = typer.Option(
        ",".join(RESTORATION_RANK_ORDERS),
        help="swept tie-break orders: morphology | context",
    ),
    dense_case: bool = typer.Option(
        False,
        "--dense-case",
        help="route the raw question's capitalization to the case-sensitive dense encoder "
        "(the measured recommendation wherever the normalize step is on)",
    ),
    full_grid: bool = typer.Option(
        False,
        "--full-grid",
        help="measure the whole product instead of one factor at a time (reads interactions; "
        "only one-factor settings carry a per-constant verdict)",
    ),
) -> None:
    """Sweep the typo step's restoration constants; report recall + edit precision per setting."""
    from llb.eval.restoration_sweep_run import run_and_publish_sweep

    try:
        classes = (
            parse_variant_classes(variant_classes) if variant_classes else SWEEP_VARIANT_CLASSES
        )
        policies = policy_grid(
            _integers(surface_distances, "--surface-distances"),
            _integers(short_cutoffs, "--short-cutoffs"),
            _ranks(rank_orders),
            full=full_grid,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    cfg = load_config(
        config,
        goldset_path=goldset,
        corpus_root=corpus_root,
        seed=seed,
        top_k=top_k,
    )
    try:
        run = run_and_publish_sweep(
            cfg,
            split=split,
            limit=limit,
            typo_rate=typo_rate,
            variant_classes=classes,
            policies=policies,
            dense_case=dense_case,
            progress=typer.echo,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    for verdict in run.verdicts:
        typer.echo(f"[restoration-sweep] {verdict.constant}: {verdict.verdict}")
    typer.echo(f"[restoration-sweep] report -> {run.paths['report']}")
    typer.echo(f"[restoration-sweep] edit audit -> {run.paths['edit_audit']}")
