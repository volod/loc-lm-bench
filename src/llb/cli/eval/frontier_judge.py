"""Frontier-judge authorization: agreement + cost evidence per provider."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app

DEFAULT_TRUST_THRESHOLD = 0.6


@app.command("frontier-judge-agreement")
def frontier_judge_agreement_cmd(
    worksheet: Path = typer.Option(..., help="filled calibration worksheet CSV"),
    models: str = typer.Option(
        ..., help="comma-separated litellm judge ids, e.g. 'anthropic/<m>,openai/<m>'"
    ),
    goldset: Optional[Path] = typer.Option(
        None, help="gold set JSONL supplying grounding contexts (strongly recommended)"
    ),
    corpus_root: Optional[Path] = typer.Option(
        None, help="corpus root for span windows (default: corpus/ beside the gold set)"
    ),
    scorer_egress_consent: bool = typer.Option(
        False,
        "--scorer-egress-consent",
        help="REQUIRED: consent to send worksheet answers to the named providers",
    ),
    frontier_max_usd: Optional[float] = typer.Option(
        None, help="hard per-provider spend cap in USD"
    ),
    frontier_max_calls: Optional[int] = typer.Option(None, help="hard per-provider call cap"),
    threshold: float = typer.Option(
        DEFAULT_TRUST_THRESHOLD, help="rho at or above which a judge is recommended as trusted"
    ),
    limit: Optional[int] = typer.Option(None, help="cap judged items (cheap dry run)"),
    out_dir: Optional[Path] = typer.Option(
        None, help="output dir (default: $DATA_DIR/frontier-judge/<timestamp>)"
    ),
) -> None:
    """Frontier judge authorization: score a filled calibration worksheet with each frontier
    provider, report Spearman rho against BOTH the human rating and the local judge rating,
    and price the run per item with the cap the measured cost implies. Sends worksheet answers
    to external providers and spends real money, so it requires --scorer-egress-consent plus a
    cap; the accept/reject decision per provider stays the operator's."""
    from llb.scoring.frontier_agreement import run_frontier_agreement
    from llb.scoring.policy.errors import ScorerPolicyError

    if not scorer_egress_consent:
        typer.echo(
            "[frontier-judge] refusing to send answers to a frontier provider without "
            "--scorer-egress-consent",
            err=True,
        )
        raise typer.Exit(code=2)
    if frontier_max_usd is None and frontier_max_calls is None:
        typer.echo(
            "[frontier-judge] set --frontier-max-usd and/or --frontier-max-calls before spending",
            err=True,
        )
        raise typer.Exit(code=2)

    model_ids = [part.strip() for part in models.split(",") if part.strip()]
    if not model_ids:
        typer.echo("[frontier-judge] --models is empty", err=True)
        raise typer.Exit(code=2)

    try:
        payload, out_path = run_frontier_agreement(
            worksheet,
            model_ids,
            goldset=goldset,
            corpus_root=corpus_root,
            out_dir=out_dir,
            max_usd=frontier_max_usd,
            max_calls=frontier_max_calls,
            threshold=threshold,
            limit=limit,
        )
    except ScorerPolicyError as exc:
        typer.echo(f"[frontier-judge] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    for provider in payload["providers"]:
        headline = provider["vs_human"]["mean"]
        rho = "n/a" if headline is None else f"{headline['rho']:.3f}"
        typer.echo(
            f"[frontier-judge] {provider['model']} rho_vs_human={rho} "
            f"cost=${provider['cost']['cost_usd']:.6f} -> {provider['recommendation']}"
        )
    for failure in payload["failures"]:
        typer.echo(f"[frontier-judge] {failure['model']} INCOMPLETE: {failure['reason']}", err=True)
    typer.echo(f"[frontier-judge] items={payload['n_items']} -> {out_path}")
    if payload["failures"]:
        raise typer.Exit(code=1)
