"""Support for `research-conflict-nulls`: option validation and the console summary.

Split from the command module so the command file stays the flag declaration plus the wiring that
opens the stores and runs the matrix.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from llb.conflicts.constants import (
    RESEARCH_GENERATION_FOURTH,
    RESEARCH_GENERATION_INITIAL,
    RESEARCH_GENERATION_THIRD,
    RESEARCH_GENERATIONS,
)

if TYPE_CHECKING:
    from llb.conflicts.store_access import StoreView
    from llb.core.contracts.common import JsonObject


def validate_generation(
    generation: str,
    reference_corpus: Optional[Path],
    reference_store: Optional[Path],
    domain_reference_corpus: Optional[Path],
    domain_reference_store: Optional[Path],
    conflict_model: Optional[str],
) -> None:
    """Reject option combinations no generation can run, before any store is opened."""
    if generation not in RESEARCH_GENERATIONS:
        raise typer.BadParameter(
            f"unknown generation {generation!r}; choose one of {', '.join(RESEARCH_GENERATIONS)}"
        )
    if (reference_corpus is None) != (reference_store is None):
        raise typer.BadParameter("reference corpus and store must be supplied together")
    if generation != RESEARCH_GENERATION_FOURTH and reference_corpus is None:
        raise typer.BadParameter(f"--generation {generation} requires a reference corpus and store")
    if (domain_reference_corpus is None) != (domain_reference_store is None):
        raise typer.BadParameter("domain reference corpus and store must be supplied together")
    if (
        generation not in (RESEARCH_GENERATION_INITIAL, RESEARCH_GENERATION_FOURTH)
        and domain_reference_corpus is None
    ):
        raise typer.BadParameter(
            f"--generation {generation} requires a domain reference corpus and store"
        )
    if generation == RESEARCH_GENERATION_THIRD and not conflict_model:
        raise typer.BadParameter(
            "--generation third needs --conflict-model: the claim-tier precision and control-role "
            "lanes are model-adjudicated"
        )
    if generation == RESEARCH_GENERATION_FOURTH and not conflict_model:
        raise typer.BadParameter(
            "--generation fourth needs --conflict-model: the control bank is generated and "
            "verified by the local model before any threshold is fitted"
        )


def shared_embedding_model(views: list[Optional["StoreView"]]) -> str:
    """Return the one encoder every supplied store was built with (None entries are skipped)."""
    models = {view.embedding_model for view in views if view is not None}
    if len(models) != 1:
        raise typer.BadParameter(
            "all research stores must use the same embedding model; found "
            + ", ".join(sorted(models))
        )
    return next(iter(models))


def echo_summary(generation: str, summary: "JsonObject", paths: dict[str, Path]) -> None:
    typer.echo(f"[conflict-null] generation={generation} verdict={summary['verdict']}")
    for method in summary["methods"]:
        typer.echo(
            f"[conflict-null] method={method['method']} accepted={method['gates']['accepted']}"
        )
    precision = summary.get("claim_precision")
    if isinstance(precision, dict):
        typer.echo(
            f"[conflict-null] method={precision['method']} "
            f"accepted={precision['gates']['accepted']}"
        )
    typer.echo(f"[conflict-null] report: {paths['report']}")
    if "control_traces" in paths:
        typer.echo(f"[conflict-null] control traces: {paths['control_traces']}")
