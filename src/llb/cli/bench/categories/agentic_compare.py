"""Persisted harness and context-policy comparison commands."""

from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config


@app.command("bench-agentic-context-compare")
def bench_agentic_context_compare_cmd(
    model: str = typer.Option(..., help="the candidate model to compare across context policies"),
) -> None:
    """Rank one model's persisted agent context-policy runs."""
    from llb.board.agentic_context import agentic_context_comparison

    cfg = load_config(None)
    rows, table, policies = agentic_context_comparison(cfg.data_dir, model)
    if not rows:
        typer.echo(
            f"[bench-agentic-context-compare] no context-policy runs for model '{model}' under "
            f"{cfg.data_dir}; run `llb bench-agentic-context --model {model} ...` first"
        )
        raise typer.Exit(code=2)
    typer.echo(
        f"[bench-agentic-context-compare] model={model} policies={', '.join(sorted(set(policies)))}"
    )
    typer.echo(table)


@app.command("bench-agentic-compare")
def bench_agentic_compare_cmd(
    model: str = typer.Option(..., help="the candidate model to compare across harnesses"),
    context_policy: Optional[str] = typer.Option(
        None,
        help="hold this context policy fixed when ranking harnesses; default = the policy with "
        "the most harness coverage (newest on a tie)",
    ),
) -> None:
    """Rank one model's agentic runs across harnesses under one context policy."""
    from llb.board.harnesses import harness_comparison

    cfg = load_config(None)
    rows, table, harnesses = harness_comparison(cfg.data_dir, model, context_policy=context_policy)
    if not rows:
        typer.echo(
            f"[bench-agentic-compare] no agentic runs for model '{model}' under {cfg.data_dir}; "
            "run `llb bench-agentic --harness loop|langgraph|crewai ...` first"
        )
        raise typer.Exit(code=2)
    typer.echo(
        f"[bench-agentic-compare] model={model} harnesses={', '.join(sorted(set(harnesses)))}"
    )
    typer.echo(table)
