"""CLI: `llb joint-search` -- successive-halving model + RAG-config search."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import (
    best_effort_gpu_readers,
    load_config,
    load_models,
    resolver_probes,
)
from llb.optimize.joint_search.constants import (
    DEFAULT_ETA,
    DEFAULT_MIN_FINALISTS,
    DEFAULT_OBJECTIVES,
    DEFAULT_SCREEN_LIMIT,
)


@app.command("joint-search")
def joint_search_cmd(
    candidates: Path = typer.Option(
        Path("samples/configs/models_uk.yaml"),
        "--candidates",
        help="candidate-models YAML manifest",
    ),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL for screen + tune"),
    trials: int = typer.Option(20, min=1, help="stage-1 Optuna trials per finalist"),
    screen_limit: int = typer.Option(
        DEFAULT_SCREEN_LIMIT,
        min=1,
        help="tuning-split case cap for screen round 0 (later rounds multiply by eta)",
    ),
    min_finalists: int = typer.Option(
        DEFAULT_MIN_FINALISTS,
        min=1,
        help="survivors that advance to per-finalist multi-objective tune",
    ),
    eta: int = typer.Option(DEFAULT_ETA, min=2, help="successive-halving reduction factor"),
    objectives: str = typer.Option(
        DEFAULT_OBJECTIVES, help="multi-objective goals for finalist tunes"
    ),
    run_id: Optional[str] = typer.Option(
        None, help="artifact id (default: UTC timestamp); reuse to resume after kill"
    ),
    offline: bool = typer.Option(False, help="resolver: assume declared sources exist"),
    isolate: bool = typer.Option(
        True, help="VRAM-reclaim isolation around screen cells and Optuna trials"
    ),
    max_model_len: int = typer.Option(8192, help="vLLM context cap per cell"),
    seed: int = typer.Option(13, help="Optuna sampler seed"),
    limit: Optional[int] = typer.Option(
        None, help="cap gold cases for screen + finalist tune evals (smoke/evidence)"
    ),
    corpus: Optional[Path] = typer.Option(None, help="corpus directory for RAG store builds"),
) -> None:
    """Screen candidates on the tuning split, halve, deep-tune survivors, final scoreboard.

    Elimination uses the tuning split only; the scoreboard is built from final-split pick
    scores so there is no tuning/final leakage into the recommendation.
    """
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.optimize.joint_search import run_joint_search

    models = load_models(candidates)
    overrides: dict[str, object] = {"goldset_path": goldset}
    if corpus is not None:
        overrides["corpus_root"] = corpus
    cfg = load_config(None, **{k: v for k, v in overrides.items() if v is not None})
    gpus = detect_gpus()
    vram_reader, pid_reader = best_effort_gpu_readers() if isolate else (None, None)
    sid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    typer.echo(
        f"[joint-search] run={sid} candidates={len(models)} screen_limit={screen_limit} "
        f"min_finalists={min_finalists} trials={trials} objectives={objectives}"
    )
    result = run_joint_search(
        cfg,
        models,
        n_trials=trials,
        run_id=sid,
        screen_limit=screen_limit,
        min_finalists=min_finalists,
        eta=eta,
        objectives=objectives,
        vram_mib=max_vram_mb(gpus),
        ram_mib=detect_ram_mb(),
        probes=resolver_probes(offline),
        isolate=isolate,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        seed=seed,
        max_model_len=max_model_len,
        case_limit=limit,
    )
    for skip in result.skipped:
        typer.echo(f"[joint-search] skip {skip['name']}: {skip['reason']}")
    for round_rec in result.ledger.rounds:
        typer.echo(
            f"[joint-search] screen round={round_rec.round_index} "
            f"limit={round_rec.case_limit} split={round_rec.split} "
            f"kept={list(round_rec.kept)} eliminated={list(round_rec.eliminated)}"
        )
    typer.echo(f"[joint-search] finalists: {list(result.ledger.finalists)}")
    for finalist in result.finalists:
        typer.echo(
            f"[joint-search] tuned {finalist.name} picks={list(finalist.finals)} "
            f"study={finalist.study_name}"
        )
    if result.recommended:
        rec = result.recommended
        typer.echo(
            f"[joint-search] recommended: {rec['model']} pick={rec['pick']} "
            f"quality={rec.get('quality')} overrides={rec.get('overrides')}"
        )
    md = result.scoreboard_paths.get("markdown")
    typer.echo(f"[joint-search] artifacts: {result.run_dir}" + (f" scoreboard={md}" if md else ""))
