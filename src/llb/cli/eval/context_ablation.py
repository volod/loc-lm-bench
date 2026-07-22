"""RAG-versus-long-context ablation across context lanes (`compare-context-strategies`)."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED


@app.command("compare-context-strategies")
def compare_context_strategies_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    model: Optional[str] = typer.Option(None, help="model name (Ollama tag or HF repo id)"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    corpus: Optional[Path] = typer.Option(
        None, help="corpus root the gold spans point into (the long_context lane reads it)"
    ),
    split: str = typer.Option(
        "final",
        help="gold split(s) to evaluate; a comma-separated list scores one run bundle per split "
        "and pools them into ONE compared item set",
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of eval items"),
    lanes: Optional[str] = typer.Option(
        None,
        help="comma-separated context lanes to score (default: closed_book,rag,long_context). "
        "closed_book is always the baseline every derived number is stated against",
    ),
    include_drafted: bool = typer.Option(
        False,
        "--include-drafted",
        help="score a DRAFTED ledger whose items no reviewer has accepted. Every artifact records "
        "`grounding: drafted`; never use it for a leaderboard run",
    ),
    resamples: int = typer.Option(DEFAULT_RESAMPLES, min=0, help="bootstrap resamples"),
    confidence: float = typer.Option(DEFAULT_CONFIDENCE, min=0.5, max=0.999, help="CI level"),
    seed: int = typer.Option(DEFAULT_SEED, help="bootstrap resampling seed"),
    out_dir: Optional[Path] = typer.Option(
        None, help="artifact dir (default: $DATA_DIR/context-ablation/<timestamp>/)"
    ),
) -> None:
    """Score one item set under three context lanes and report whether RAG pays for itself.

    `closed_book` sends no context, so its score is what the model already knows; `rag` is the run
    configuration as-is; `long_context` lays the item's whole source document into the prompt and
    SKIPS (never truncates) an item whose document exceeds the model's usable window. The report
    states retrieval uplift (`rag - closed_book`), the long-context delta (`long_context - rag`),
    and the per-item contamination flag.

    Each lane persists an ordinary run bundle under `$DATA_DIR/run-eval/`; only the comparison is
    new, and the lanes are diagnostics -- `rag` stays the leaderboard row.
    """
    from llb.eval.context_ablation import parse_lanes, run_context_ablation

    cfg = load_config(
        config, model=model, backend=backend, goldset_path=goldset, corpus_root=corpus
    )
    try:
        selection = parse_lanes(lanes) if lanes else None
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    if include_drafted:
        typer.echo(
            "[compare-context-strategies] scoring a DRAFTED ledger: no reviewer has accepted "
            "these items, so the objective is diagnostic, not a leaderboard result"
        )
    splits = [name.strip() for name in split.split(",") if name.strip()]
    if not splits:
        typer.echo("[error] name at least one gold split", err=True)
        raise typer.Exit(code=2)
    try:
        run = run_context_ablation(
            cfg,
            selection,
            splits=splits,
            limit=limit,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
            out_dir=out_dir,
            verified_only=not include_drafted,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    verdict = run.report["verdict"]
    typer.echo(
        f"[compare-context-strategies] {verdict['decision']}: {verdict['reason']}"
        if verdict["reason"]
        else f"[compare-context-strategies] {verdict['decision']}"
    )
    typer.echo(f"[compare-context-strategies] report -> {run.paths['report']}")
