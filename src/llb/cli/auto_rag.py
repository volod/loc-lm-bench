"""CLI for the autonomous corpus-to-recommendation pipeline."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, cast

import typer

from llb.cli.app import app


@app.command("auto-rag")
def auto_rag_cmd(
    corpus: Path = typer.Option(..., help="mixed txt/md/pdf corpus directory"),
    run_id: Optional[str] = typer.Option(
        None, help="stable artifact id; reuse the same id to resume after interruption"
    ),
    draft_model: str = typer.Option(..., help="local Ollama model for ontology drafting"),
    candidates: Path = typer.Option(
        Path("samples/configs/models_uk.yaml"), help="candidate-model YAML manifest"
    ),
    candidate_models: str = typer.Option(
        "", help="optional comma-separated candidate names from the manifest"
    ),
    scorer_policy: str = typer.Option("auto", help="gate policy: auto | human | local | frontier"),
    judge_model: Optional[str] = typer.Option(
        None, help="gate judge model (default: the local draft model)"
    ),
    judge_base_url: Optional[str] = typer.Option(None, help="OpenAI-compatible judge base URL"),
    egress_consent: bool = typer.Option(False, help="frontier judge: explicit egress consent"),
    max_usd: Optional[float] = typer.Option(None, min=0.000001, help="frontier spend cap"),
    max_calls: Optional[int] = typer.Option(None, min=1, help="frontier call cap"),
    max_items: int = typer.Option(60, min=3, help="maximum ontology-drafted QA items"),
    doc_limit: Optional[int] = typer.Option(None, min=1, help="bounded corpus document count"),
    trials: int = typer.Option(20, min=1, help="Optuna trials per joint-search finalist"),
    screen_limit: int = typer.Option(8, min=1, help="tuning cases in the first halving round"),
    min_finalists: int = typer.Option(2, min=1, help="models retained for deep tuning"),
    eval_limit: Optional[int] = typer.Option(None, min=1, help="bounded cases per model eval"),
    seed: int = typer.Option(13, help="deterministic split/search seed"),
    max_model_len: int = typer.Option(8192, min=1024, help="model/prompt context cap"),
    parity_check: bool = typer.Option(
        False, help="repeat the final manual-chain eval and require equal quality"
    ),
) -> None:
    """Ingest a corpus, verify/tune/evaluate RAG, and emit one recommendation bundle."""
    from llb.auto_rag import AutoRagPaused, AutoRagSettings, run_auto_rag
    from llb.auto_rag.models import GatePolicy
    from llb.core.paths import resolve_data_dir, resolve_project_path
    from llb.scoring.policy import SCORER_LANES

    policies = ("auto", *SCORER_LANES)
    if scorer_policy not in policies:
        raise typer.BadParameter(f"scorer-policy must be one of {policies}")
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    settings = AutoRagSettings(
        corpus=resolve_project_path(corpus),
        data_dir=resolve_data_dir(),
        run_id=rid,
        draft_model=draft_model,
        candidates=resolve_project_path(candidates),
        candidate_models=tuple(
            value.strip() for value in candidate_models.split(",") if value.strip()
        ),
        gate_policy=cast(GatePolicy, scorer_policy),
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        egress_consent=egress_consent,
        max_usd=max_usd,
        max_calls=max_calls,
        max_items=max_items,
        doc_limit=doc_limit,
        trials=trials,
        screen_limit=screen_limit,
        min_finalists=min_finalists,
        eval_limit=eval_limit,
        seed=seed,
        max_model_len=max_model_len,
        parity_check=parity_check,
    )
    try:
        status = run_auto_rag(settings)
    except AutoRagPaused as exc:
        typer.echo(f"[auto-rag] paused: {exc}")
        raise typer.Exit(code=3) from None
    typer.echo(
        f"[auto-rag] {'resumed and ' if status.resumed else ''}completed -> {status.run_dir}"
    )
    typer.echo(f"[auto-rag] recommendation: {status.recommendation}")
    typer.echo(f"[auto-rag] report: {status.report}")
