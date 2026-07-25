"""Embedder adoption-bar sweep (`compare-embedder-adoption`): does a rank gain reach the answer?"""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED

DEFAULT_TOP_KS = "10,3"
DEFAULT_RERANKERS = "off,on"


@app.command("compare-embedder-adoption")
def compare_embedder_adoption_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    model: Optional[str] = typer.Option(None, help="model name (Ollama tag or HF repo id)"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    corpus: Optional[Path] = typer.Option(
        None,
        help="corpus the two stores were built from; both encoders index the SAME corpus, so one "
        "root validates both stores (defaults to the config's corpus_root)",
    ),
    split: str = typer.Option(
        "final",
        help="gold split(s) to evaluate; a comma-separated list scores one run bundle per split "
        "and pools them into ONE compared item set",
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of eval items"),
    baseline_embedder: str = typer.Option(
        ...,
        "--baseline-embedder",
        help="the INCUMBENT encoder; the sweep asks whether to replace this row",
    ),
    baseline_data_dir: Path = typer.Option(
        ...,
        "--baseline-data-dir",
        help="data root whose llb/rag store was built with --baseline-embedder",
    ),
    candidate_embedder: str = typer.Option(
        ..., "--candidate-embedder", help="the encoder under test"
    ),
    candidate_data_dir: Path = typer.Option(
        ...,
        "--candidate-data-dir",
        help="data root whose llb/rag store was built with --candidate-embedder",
    ),
    top_ks: str = typer.Option(
        DEFAULT_TOP_KS,
        "--top-ks",
        help="comma-separated retrieval budgets to sweep (the shipped one plus a small one)",
    ),
    rerankers: str = typer.Option(
        DEFAULT_RERANKERS,
        "--rerankers",
        help="comma-separated reranker settings: off | on (the pinned cross-encoder) | an "
        "explicit cross-encoder id",
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
        None, help="artifact dir (default: $DATA_DIR/embedder-adoption-bar/<timestamp>/)"
    ),
) -> None:
    """Score two encoders END TO END across a `top_k` x reranker grid and decide the adoption bar.

    `compare-embeddings` adopts on recall@k alone, so an encoder that only ranks the same evidence
    EARLIER (an MRR gain with a recall delta spanning zero) is discarded by construction. Whether
    that is right depends on the scored configuration: rank binds at a small `top_k` and under a
    cross-encoder that can only re-sort what it is handed, and is nearly free at a generous one.

    This sweep runs the standard `run-eval` for each (cell, encoder) pair over the IDENTICAL item
    set and reports the answer-side delta per cell, ending in an explicit keep-or-extend call on
    the bake-off's adoption bar. Each pair persists an ordinary run bundle under its encoder's own
    `$DATA_DIR/run-eval/`; only the comparison is new.
    """
    from llb.eval.embedder_adoption import (
        EmbedderLane,
        build_cells,
        format_summary,
        parse_rerankers,
        parse_top_ks,
        run_adoption_bar_sweep,
    )

    overrides: dict[str, object] = {"model": model, "backend": backend, "goldset_path": goldset}
    if corpus is not None:
        overrides["corpus_root"] = corpus
    cfg = load_config(config, **overrides)
    try:
        cells = build_cells(parse_top_ks(top_ks), parse_rerankers(rerankers))
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    splits = [name.strip() for name in split.split(",") if name.strip()]
    if not splits:
        typer.echo("[error] name at least one gold split", err=True)
        raise typer.Exit(code=2)
    if include_drafted:
        typer.echo(
            "[compare-embedder-adoption] scoring a DRAFTED ledger: no reviewer has accepted these "
            "items, so the objective is diagnostic, not a leaderboard result"
        )
    lanes = [
        EmbedderLane(baseline_embedder, baseline_data_dir),
        EmbedderLane(candidate_embedder, candidate_data_dir),
    ]
    try:
        run = run_adoption_bar_sweep(
            cfg,
            cells,
            lanes,
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
    typer.echo(format_summary(run.report))
    typer.echo(f"[compare-embedder-adoption] report -> {run.paths['report']}")
