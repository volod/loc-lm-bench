"""RAG-versus-long-context ablation across context lanes (`compare-context-strategies`)."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.eval.context_ablation.models import RETRIEVED_DOCUMENT_NOT_MEASURED
from llb.eval.context_ablation.power import DEFAULT_TARGET_POWER
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED


@app.command("compare-context-strategies")
def compare_context_strategies_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    model: Optional[str] = typer.Option(None, help="model name (Ollama tag or HF repo id)"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    corpus: Optional[Path] = typer.Option(
        None,
        help="corpus root the gold spans point into; both document lanes read whole documents "
        "from it",
    ),
    split: str = typer.Option(
        "final",
        help="gold split(s) to evaluate; a comma-separated list scores one run bundle per split "
        "and pools them into ONE compared item set",
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of eval items"),
    retrieved_document_top_n: Optional[int] = typer.Option(
        None,
        min=1,
        help="how many DISTINCT retrieved documents the retrieved_document lane lays in, walking "
        "the ranked chunk list best-first (default 1). Set it to --top-k to widen the SAME "
        "retrieved set from chunks to documents instead of also narrowing the depth",
    ),
    lanes: Optional[str] = typer.Option(
        None,
        help="comma-separated context lanes to score (default: closed_book,rag,"
        "retrieved_document,long_context). closed_book is always the baseline every derived "
        "number is stated against",
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
    power_reference: Optional[Path] = typer.Option(
        None,
        help="earlier comparison.json whose paired long-context variance prices the new item count",
    ),
    minimum_detectable_delta: Optional[float] = typer.Option(
        None,
        min=0.001,
        help="smallest material objective delta the predeclared run is powered to distinguish",
    ),
    target_power: float = typer.Option(
        DEFAULT_TARGET_POWER,
        min=0.501,
        max=0.999,
        help="target power for the paired normal-approximation item count",
    ),
) -> None:
    """Score one item set under every context lane and report whether RAG pays for itself.

    `closed_book` sends no context, so its score is what the model already knows; `rag` is the run
    configuration as-is; `retrieved_document` retrieves as configured and then sends the whole
    document the top-ranked chunk came from; `long_context` lays the item's whole GOLD document
    in. Both document lanes SKIP (never truncate) an item whose documents exceed the model's
    usable window. The report states retrieval uplift (`rag - closed_book`), the oracle
    long-context delta (`long_context - rag`), the split of that gap into a capturable part
    (`retrieved_document - rag`) and a gold-label part (`long_context - retrieved_document`), and
    the per-item contamination flag.

    Each lane persists an ordinary run bundle under `$DATA_DIR/run-eval/`; only the comparison is
    new. `closed_book` and `long_context` are diagnostics and `rag` stays the leaderboard row;
    `retrieved_document` is the one lane an operator could ship, so it gets its own adopt-or-reject
    verdict.
    """
    from llb.eval.context_ablation import parse_lanes, run_context_ablation

    cfg = load_config(
        config,
        model=model,
        backend=backend,
        goldset_path=goldset,
        corpus_root=corpus,
        retrieved_document_top_n=retrieved_document_top_n,
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
            power_reference=power_reference,
            minimum_detectable_delta=minimum_detectable_delta,
            target_power=target_power,
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
    adoption = verdict.get("retrieved_document")
    if adoption is not None and adoption["decision"] != RETRIEVED_DOCUMENT_NOT_MEASURED:
        typer.echo(f"[compare-context-strategies] {adoption['decision']}: {adoption['reason']}")
    typer.echo(f"[compare-context-strategies] report -> {run.paths['report']}")
