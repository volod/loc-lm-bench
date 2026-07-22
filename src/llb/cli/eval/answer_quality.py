"""Multi-hop answer-quality comparison across retrieval lanes (`compare-answer-quality`)."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED

DEFAULT_LANES = "vector,fused/global_community@0.30"


@app.command("compare-answer-quality")
def compare_answer_quality_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    model: Optional[str] = typer.Option(None, help="model name (Ollama tag or HF repo id)"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    split: str = typer.Option(
        "final",
        help="gold split(s) to evaluate; a comma-separated list scores one run bundle per split "
        "and pools them into ONE compared item set",
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of eval items"),
    lanes: str = typer.Option(
        DEFAULT_LANES,
        help="comma-separated retrieval lanes to score; the FIRST is the baseline. Labels are "
        "compare-graph-fusion row labels: vector | graph/<strategy> | "
        "fused/<strategy>@<weight>[/d<depth>] | routed/<strategy>@<weight>[/d<depth>]",
    ),
    from_comparison: Optional[Path] = typer.Option(
        None,
        "--from-comparison",
        help="a compare-graph-fusion comparison.json; scores its baseline plus the fused row its "
        "verdict named best (overrides --lanes)",
    ),
    include_drafted: bool = typer.Option(
        False,
        "--include-drafted",
        help="score a DRAFTED ledger whose items no reviewer has accepted -- the only way to "
        "measure the same set a drafted-grounded retrieval sweep measured. Every artifact records "
        "`grounding: drafted`; never use it for a leaderboard run",
    ),
    focus_slice: Optional[str] = typer.Option(
        None, help="question type the verdict is decided on (default: multi-hop)"
    ),
    resamples: int = typer.Option(DEFAULT_RESAMPLES, min=0, help="bootstrap resamples"),
    confidence: float = typer.Option(DEFAULT_CONFIDENCE, min=0.5, max=0.999, help="CI level"),
    seed: int = typer.Option(DEFAULT_SEED, help="bootstrap resampling seed"),
    out_dir: Optional[Path] = typer.Option(
        None,
        help="artifact dir (default: "
        "$DATA_DIR/graph-vector-fusion-multihop/<timestamp>/answer-quality/)",
    ),
) -> None:
    """Score the multi-hop slice END TO END under two retrieval lanes and compare the answers.

    `compare-graph-fusion` measures whether the retrieved context CARRIES every span a multi-hop
    answer needs; it cannot say whether the model then uses both. This lane runs the standard
    `run-eval` under each retrieval lane over the IDENTICAL item set and reports the objective per
    question-type slice, so a measured coverage gain is either confirmed as an answer-quality gain
    or recorded as a retrieval-only effect.

    Each lane persists an ordinary run bundle under `$DATA_DIR/run-eval/`; only the comparison is
    new.
    """
    from llb.eval.answer_quality import (
        FOCUS_SLICE,
        lane_labels_from_comparison,
        parse_lanes,
        run_answer_quality,
    )

    cfg = load_config(config, model=model, backend=backend, goldset_path=goldset)
    try:
        selection = (
            ",".join(lane_labels_from_comparison(from_comparison))
            if from_comparison is not None
            else lanes
        )
        specs = parse_lanes(selection)
    except (OSError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    if len(specs) < 2:
        typer.echo("[error] name at least two lanes: a baseline and a candidate", err=True)
        raise typer.Exit(code=2)
    if include_drafted:
        typer.echo(
            "[compare-answer-quality] scoring a DRAFTED ledger: no reviewer has accepted these "
            "items, so the objective is diagnostic, not a leaderboard result"
        )
    splits = [name.strip() for name in split.split(",") if name.strip()]
    if not splits:
        typer.echo("[error] name at least one gold split", err=True)
        raise typer.Exit(code=2)
    run = run_answer_quality(
        cfg,
        specs,
        splits=splits,
        limit=limit,
        focus_slice=focus_slice or FOCUS_SLICE,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
        out_dir=out_dir,
        verified_only=not include_drafted,
    )
    verdict = run.report["verdict"]
    typer.echo(
        f"[compare-answer-quality] {verdict['decision']}: {verdict['reason']}"
        if verdict["reason"]
        else f"[compare-answer-quality] {verdict['decision']}"
    )
    typer.echo(f"[compare-answer-quality] report -> {run.paths['report']}")
