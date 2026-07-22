"""Held-out threshold calibration for the sidecar-free graph-fusion router."""

import json
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED

DEFAULT_LONG_WORD_GRID = "10,12,14,16,18,20"
DEFAULT_ENTITY_GRID = "0,1,2"
DEFAULT_GRAPH_STRATEGY = "global_community"
DEFAULT_GRAPH_WEIGHT = 0.3
DEFAULT_CANDIDATES = 50
DEFAULT_SPAN_IDENTITY = "overlap"


@app.command("calibrate-fusion-routing")
def calibrate_fusion_routing_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    k: int = typer.Option(10, min=1, help="retrieval cutoff"),
    tuning_split: str = typer.Option("tuning", help="split used to select thresholds"),
    final_split: str = typer.Option("final", help="held-out split scored after selection"),
    long_question_words: str = typer.Option(
        DEFAULT_LONG_WORD_GRID, help="comma-separated long-question thresholds"
    ),
    min_linked_entities: str = typer.Option(
        DEFAULT_ENTITY_GRID,
        help="comma-separated linked-entity thresholds; zero makes length sufficient",
    ),
    graph_strategy: str = typer.Option(DEFAULT_GRAPH_STRATEGY, help="graph strategy"),
    graph_weight: float = typer.Option(DEFAULT_GRAPH_WEIGHT, min=0.0, max=1.0),
    candidates: int = typer.Option(DEFAULT_CANDIDATES, min=1, help="per-lane candidate depth"),
    span_identity: str = typer.Option(DEFAULT_SPAN_IDENTITY, help="exact | overlap"),
    resamples: int = typer.Option(DEFAULT_RESAMPLES, min=0, help="bootstrap resamples"),
    confidence: float = typer.Option(DEFAULT_CONFIDENCE, min=0.5, max=0.999),
    seed: int = typer.Option(DEFAULT_SEED, help="bootstrap seed"),
    out_dir: Optional[Path] = typer.Option(None, help="artifact directory"),
) -> None:
    """Tune without sidecar labels, freeze one policy, then score only that policy on final."""
    from llb.cli.rag.fusion_evidence import (
        FUSION_EVIDENCE_METHOD,
        _evidence_items,
        _load_lanes,
    )
    from llb.core.store_generations import generation_timestamp
    from llb.rag.fusion_calibration import (
        calibrate_routing,
        format_report,
        parse_thresholds,
        policy_grid,
    )
    from llb.rag.fusion_spans import resolve_span_identity

    cfg = load_config(config, goldset_path=goldset)
    try:
        policies = policy_grid(
            parse_thresholds(long_question_words),
            parse_thresholds(min_linked_entities, allow_zero=True),
        )
        identity = resolve_span_identity(span_identity)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    tuning_items = _evidence_items(cfg, tuning_split)
    final_items = _evidence_items(cfg, final_split)
    if not tuning_items or not final_items:
        typer.echo("[error] tuning and final selections must both be non-empty", err=True)
        raise typer.Exit(code=2)
    vector, graphs = _load_lanes(cfg, graph_strategy)
    graph = graphs.get(graph_strategy)
    if graph is None:
        typer.echo(f"[error] graph strategy was not loaded: {graph_strategy}", err=True)
        raise typer.Exit(code=2)
    report = calibrate_routing(
        vector,
        graph,
        tuning_items,
        final_items,
        policies,
        k=k,
        graph_strategy=graph_strategy,
        graph_weight=graph_weight,
        candidates=candidates,
        span_identity=identity,
        tuning_split=tuning_split,
        final_split=final_split,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    default = (
        cfg.data_dir / FUSION_EVIDENCE_METHOD / f"{generation_timestamp()}-routing-calibration"
    )
    target = Path(out_dir) if out_dir is not None else default
    target.mkdir(parents=True, exist_ok=True)
    (target / "calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "report.md").write_text(format_report(report), encoding="utf-8")
    (target / "run_config.json").write_text(
        json.dumps(cfg.fingerprint(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(f"[calibrate-fusion-routing] {report['decision']}: {report['reason']}")
    typer.echo(f"[calibrate-fusion-routing] report -> {target / 'report.md'}")
