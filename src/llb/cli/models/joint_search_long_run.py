"""CLI: `llb joint-search-long-run` -- the research-scale roster confirmation."""

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
)
from llb.optimize.joint_search.long_run.plan import (
    DEFAULT_MINIMUM_DETECTABLE_GAIN,
    DEFAULT_STABILITY_AGREEMENT,
    DEFAULT_STABILITY_BLOCKS,
    DEFAULT_TRIAL_BLOCK,
    DEFAULT_TRIAL_BUDGET,
)
from llb.rag.fusion_evidence.power import DEFAULT_TARGET_POWER
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE


@app.command("joint-search-long-run")
def joint_search_long_run_cmd(
    candidates: Path = typer.Option(
        Path("samples/configs/models_uk.yaml"),
        "--candidates",
        help="candidate-models YAML manifest",
    ),
    incumbent: str = typer.Option(
        ..., help="the model the adoption decision is measured against (a candidate `name`)"
    ),
    power_reference: Path = typer.Option(
        ...,
        help="run bundle whose per-case objective scores supply the CANDIDATE side of the variance",
    ),
    power_baseline: Path = typer.Option(
        ..., help="run bundle supplying the BASELINE side of the same paired variance"
    ),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL for screen + tune + final"),
    corpus: Optional[Path] = typer.Option(None, help="corpus directory for RAG store builds"),
    minimum_detectable_gain: float = typer.Option(
        DEFAULT_MINIMUM_DETECTABLE_GAIN,
        min=1e-6,
        help="smallest objective gain worth swapping the default model for (declared, not fitted)",
    ),
    target_power: float = typer.Option(DEFAULT_TARGET_POWER, help="power the screen is sized for"),
    confidence: float = typer.Option(
        DEFAULT_CONFIDENCE, help="two-sided reporting level for every interval and reading"
    ),
    trial_budget: int = typer.Option(
        DEFAULT_TRIAL_BUDGET, min=1, help="hard cap on multi-objective trials per finalist"
    ),
    trial_block: int = typer.Option(
        DEFAULT_TRIAL_BLOCK, min=1, help="trials each finalist advances between ranking reads"
    ),
    stability_blocks: int = typer.Option(
        DEFAULT_STABILITY_BLOCKS, min=1, help="consecutive stable block transitions that stop it"
    ),
    stability_agreement: float = typer.Option(
        DEFAULT_STABILITY_AGREEMENT, help="pairwise rank agreement a transition needs (1.0 = same)"
    ),
    min_finalists: int = typer.Option(DEFAULT_MIN_FINALISTS, min=1, help="survivors to deep-tune"),
    eta: int = typer.Option(DEFAULT_ETA, min=2, help="successive-halving reduction factor"),
    objectives: str = typer.Option(DEFAULT_OBJECTIVES, help="multi-objective goals"),
    run_id: Optional[str] = typer.Option(
        None, help="artifact id (default: UTC timestamp); reuse to resume after kill"
    ),
    offline: bool = typer.Option(False, help="resolver: assume declared sources exist"),
    isolate: bool = typer.Option(True, help="VRAM-reclaim isolation around cells and trials"),
    max_model_len: int = typer.Option(8192, help="vLLM context cap per cell"),
    seed: int = typer.Option(13, help="Optuna sampler seed"),
    public_limit: Optional[int] = typer.Option(
        None, help="cap examples per public lm-eval task (smoke runs)"
    ),
    public_evict: bool = typer.Option(
        True,
        help="unload Ollama's resident models before a vLLM finalist's public screen launches",
    ),
) -> None:
    """Confirm the roster ranking at research scale and state an adopt-or-retain verdict.

    Declares the minimum detectable objective gain and the stopping rule BEFORE measuring, derives
    the tuning-screen size from paired power over the two named reference bundles, spends
    multi-objective trials in blocks until the finalist ranking settles or the budget is gone,
    scores the FULL held-out split, screens both finalists on the public Ukrainian tracks, and
    writes `long_run.{json,md}` beside the ordinary scoreboard.
    """
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.executor.runner_setup import eval_item_count
    from llb.optimize.joint_search.long_run import declare_plan, run_long_run
    from llb.optimize.joint_search.long_run.reference import paired_reference_deltas
    from llb.optimize.tuning_space import TUNING_SPLIT

    models = load_models(candidates)
    overrides = {"goldset_path": goldset, "corpus_root": corpus}
    cfg = load_config(None, **{k: v for k, v in overrides.items() if v is not None})
    if incumbent not in {model["name"] for model in models}:
        typer.echo(f"[error] incumbent {incumbent!r} is not in {candidates}", err=True)
        raise typer.Exit(code=2)
    try:
        deltas = paired_reference_deltas(power_reference, power_baseline)
        plan = declare_plan(
            power_reference,
            deltas,
            minimum_detectable_gain=minimum_detectable_gain,
            target_power=target_power,
            confidence=confidence,
            available_n=eval_item_count(cfg, TUNING_SPLIT),
            trial_budget=trial_budget,
            trial_block=trial_block,
            stability_blocks=stability_blocks,
            stability_agreement=stability_agreement,
            selector={
                "lane": "joint-search-long-run",
                "candidate": str(power_reference),
                "baseline": str(power_baseline),
                "metric": "objective_score",
                "population": "all",
            },
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    sid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    vram_reader, pid_reader = best_effort_gpu_readers() if isolate else (None, None)
    typer.echo(
        f"[long-run] run={sid} candidates={len(models)} incumbent={incumbent} "
        f"screen={plan.screen.applied_n}/{plan.screen.required_n} "
        f"mdg={minimum_detectable_gain:+.3f} budget={trial_budget}x{trial_block}"
    )
    if not plan.screen.satisfied:
        typer.echo(
            f"[long-run] WARNING: the tuning split holds {plan.screen.available_n} items, below "
            f"the {plan.screen.required_n} the declared power asks for"
        )
    gpus = detect_gpus()
    result = run_long_run(
        cfg,
        models,
        plan=plan,
        incumbent=incumbent,
        run_id=sid,
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
        public_limit=public_limit,
        public_evict=public_evict,
    )
    _echo_result(result)


def _echo_result(result: object) -> None:
    """Print the trail, the verdict, and where the artifact landed."""
    from llb.optimize.joint_search.long_run.run import LongRunResult

    assert isinstance(result, LongRunResult)
    for skip in result.search.skipped:
        typer.echo(f"[long-run] skip {skip['name']}: {skip['reason']}")
    typer.echo(f"[long-run] finalists: {list(result.search.ledger.finalists)}")
    for block in result.trail.get("blocks", []):
        typer.echo(
            f"[long-run] block={block['index']} trials/finalist={block['trials_per_finalist']} "
            f"ranking={block['ranking']} agreement={block['agreement']} "
            f"streak={block['stable_streak']}"
        )
    typer.echo(
        f"[long-run] stopped by {result.trail.get('stopped_by')} after "
        f"{result.trail.get('consumed_total')} trials"
    )
    for name, complete in (result.public.get("complete") or {}).items():
        typer.echo(f"[long-run] public screen {name}: {'complete' if complete else 'PARTIAL'}")
    typer.echo(
        f"[long-run] verdict: {result.verdict.decision.upper()} {result.verdict.model or ''}"
    )
    typer.echo(f"[long-run]   {result.verdict.reason}")
    typer.echo(f"[long-run]   {result.verdict.tradeoff}")
    typer.echo(f"[long-run] artifacts: {result.paths['markdown']}")
