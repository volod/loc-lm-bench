"""Model prep, planning, resolution, sweep, and tuning commands."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import (
    best_effort_gpu_readers,
    echo_gpus,
    load_config,
    load_models,
    planning_models,
    resolver_probes,
)
from llb.config import RunConfig
from llb.contracts import ResolvedModel


@app.command("prep-models")
def prep_models_cmd(
    manifest: Path = typer.Option(
        Path("samples/models_uk.yaml"), help="candidate-models YAML manifest"
    ),
    backend: str = typer.Option("all", help="ollama | vllm | all"),
    force: bool = typer.Option(False, help="prepare even if a model looks too big for VRAM"),
    dry_run: bool = typer.Option(False, help="show the plan; pull/cache nothing"),
    cache_dir: Optional[Path] = typer.Option(None, help="HF cache dir for vLLM weights"),
) -> None:
    """Detect the GPU, pull Ollama tags, and cache vLLM (HF) weights once."""
    from llb.backends.prepare import prepare_models

    models = load_models(manifest)
    report = prepare_models(
        models,
        backend_filter=backend,
        force=force,
        dry_run=dry_run,
        cache_dir=cache_dir,
    )

    if report["gpus"]:
        for g in report["gpus"]:
            typer.echo(
                f"[prep-models] GPU {g.index}: {g.name} "
                f"({g.total_mb} MB total, {g.free_mb} MB free, driver {g.driver})"
            )
    else:
        typer.echo("[prep-models] no GPU detected (Ollama runs on CPU; vLLM is skipped)")

    for r in report["results"]:
        typer.echo(
            f"[prep-models] {r['status']:<8} {r['backend']:<6} {r['name']:<22} "
            f"{r['source']}  -- {r['detail']}"
        )
    failed = [r for r in report["results"] if r["status"] == "failed"]
    if failed:
        raise typer.Exit(code=1)


@app.command("list-models")
def list_models_cmd(
    manifest: Path = typer.Option(
        Path("samples/models_uk.yaml"), help="candidate-models YAML manifest"
    ),
    context: Optional[int] = typer.Option(
        None, help="plan at this target context instead of the max the host can hold"
    ),
    vram_reserve: int = typer.Option(1024, help="VRAM MiB held back for CUDA/display"),
    ram_reserve: int = typer.Option(2048, help="RAM MiB held back for the OS"),
    runnable_only: bool = typer.Option(False, help="hide models that cannot run at all"),
) -> None:
    """List which candidate models can run here (GPU+RAM, KV-cache-aware, batch=1)."""
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.backends.planner import VERDICT_NO, format_plan, plan_models

    models = planning_models(manifest)
    gpus = detect_gpus()
    vram_mib = max_vram_mb(gpus)
    ram_mib = detect_ram_mb()

    echo_gpus("list-models")
    typer.echo(f"[list-models] system RAM: {ram_mib} MiB")

    rows = plan_models(
        models,
        vram_mib,
        ram_mib,
        target_ctx=context,
        vram_reserve=vram_reserve,
        ram_reserve=ram_reserve,
    )
    if runnable_only:
        rows = [r for r in rows if r["verdict"] != VERDICT_NO]
    typer.echo(format_plan(rows, max(0, vram_mib - vram_reserve), max(0, ram_mib - ram_reserve)))
    runnable = sum(1 for r in rows if r["verdict"] in ("gpu", "offload"))
    typer.echo(f"[list-models] runnable here: {runnable} of {len(rows)}")
    typer.echo(
        "[list-models] verdict is at ctx_max; ctx_gpu = max context that fits fully "
        "on GPU. gpu = no offload needed; offload = split layers GPU/CPU RAM."
    )


@app.command("resolve-models")
def resolve_models_cmd(
    manifest: Path = typer.Option(
        Path("samples/models_uk.yaml"), help="candidate-models YAML manifest"
    ),
    context: Optional[int] = typer.Option(
        None, help="resolve fit at this target context instead of the max the host can hold"
    ),
    vram_reserve: int = typer.Option(1024, help="VRAM MiB held back for CUDA/display"),
    ram_reserve: int = typer.Option(2048, help="RAM MiB held back for the OS"),
    offline: bool = typer.Option(
        False, help="skip availability probes (assume every declared source exists)"
    ),
) -> None:
    """Pick the backend that can actually serve each model (discovery + vLLM>Ollama priority)."""
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.backends.resolver import format_resolution, resolve_all

    models = planning_models(manifest)
    gpus = detect_gpus()
    vram_mib = max_vram_mb(gpus)
    ram_mib = detect_ram_mb()
    echo_gpus("resolve-models")

    rows = resolve_all(
        models,
        vram_mib,
        ram_mib,
        probes=resolver_probes(offline),
        target_ctx=context,
        vram_reserve=vram_reserve,
        ram_reserve=ram_reserve,
    )
    typer.echo(format_resolution(rows))
    resolved = sum(1 for r in rows if r["chosen_backend"] is not None)
    typer.echo(f"[resolve-models] resolved {resolved} of {len(rows)} to a runnable backend")


def _sweep_cell_overrides(
    resolution: ResolvedModel, telemetry: bool, max_model_len: int
) -> dict[str, Any] | None:
    """Build RunConfig overrides for one resolved model, or None when not runnable."""
    if not resolution["chosen_backend"]:
        return None
    overrides: dict[str, Any] = {
        "model": resolution["chosen_source"],
        "backend": resolution["chosen_backend"],
        "measure_telemetry": telemetry,
        "run_name": f"sweep-{resolution['name']}",
    }
    if resolution["chosen_backend"] == "vllm":
        overrides["max_model_len"] = max_model_len
    elif resolution["chosen_backend"] == "llamacpp":
        from llb.backends.resolver import llamacpp_offload_split

        ngl = llamacpp_offload_split(resolution)
        if ngl is not None:
            overrides["n_gpu_layers"] = ngl
            typer.echo(f"[sweep] {resolution['name']}: llama.cpp offload split -ngl={ngl}")
    return overrides


@app.command("sweep")
def sweep_cmd(
    manifest: Path = typer.Option(
        Path("samples/models_uk.yaml"), help="candidate-models YAML manifest"
    ),
    split: str = typer.Option("final", help="gold split to evaluate each cell on"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL for every cell"),
    sweep_id: Optional[str] = typer.Option(None, help="resume id (default: a UTC timestamp)"),
    max_model_len: int = typer.Option(8192, help="vLLM context cap per cell (KV cache fit)"),
    telemetry: bool = typer.Option(True, help="measure steady-state throughput + peak VRAM"),
    resume: bool = typer.Option(True, help="skip cells already completed under this sweep id"),
    offline: bool = typer.Option(False, help="skip availability probes (assume sources exist)"),
) -> None:
    """Run one isolated cell per runnable model (process-per-cell, VRAM gate, thermal cooldown)."""
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.backends.resolver import resolve_all
    from llb.executor.isolation import run_sweep

    models = load_models(manifest)
    gpus = detect_gpus()
    resolved = resolve_all(
        models, max_vram_mb(gpus), detect_ram_mb(), probes=resolver_probes(offline), target_ctx=None
    )
    base = load_config(None, goldset_path=goldset)
    cells: list[RunConfig] = []
    for r in resolved:
        if not r["chosen_backend"]:
            typer.echo(f"[sweep] skip {r['name']}: {r['note']}")
            continue
        overrides = _sweep_cell_overrides(r, telemetry, max_model_len)
        if overrides is not None:
            cells.append(base.with_overrides(**overrides))

    if not cells:
        typer.echo("[sweep] no runnable models resolved; nothing to do")
        raise typer.Exit(code=1)

    vram_reader, pid_reader = best_effort_gpu_readers()
    sid = sweep_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    typer.echo(f"[sweep] id={sid} cells={len(cells)} split={split} resume={resume}")
    report = run_sweep(
        cells,
        sweep_id=sid,
        split=split,
        data_dir=base.data_dir,
        telemetry=telemetry,
        resume=resume,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
    )
    for cell in report["results"]:
        hot = next((g["temp_c"] for g in cell["gpu"] if g["temp_c"] is not None), None)
        typer.echo(
            f"[sweep] {cell['status']:<7} {cell['backend']:<6} {cell['model']:<34} "
            f"residual={cell['vram_residual_mb']} cooldown={cell['cooldown_s']}s temp={hot}C"
        )
    typer.echo(
        f"[sweep] done id={sid}: {report['completed']} run, {report['skipped']} skipped, "
        f"{report['failed']} failed"
    )
    if report["failed"]:
        raise typer.Exit(code=1)


@app.command("tune")
def tune_cmd(
    model: str = typer.Option(..., help="model name (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm"),
    trials: int = typer.Option(30, min=1, help="stage-1 Optuna trials on the tuning split"),
    study: Optional[str] = typer.Option(None, help="study name (persistent SQLite; resumes)"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL"),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM context cap"),
    manifest: Path = typer.Option(
        Path("samples/models_uk.yaml"), help="manifest to read the model's arch from (VRAM prune)"
    ),
    seed: int = typer.Option(13, help="Optuna sampler seed"),
    isolate: bool = typer.Option(
        False, help="run each trial under the executor VRAM-reclaim + thermal-cooldown gate"
    ),
) -> None:
    """Two-stage tune: search RAG params on the tuning split, score the winner on final."""
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.optimize.tuner import two_stage

    cfg = load_config(
        None, model=model, backend=backend, goldset_path=goldset, max_model_len=max_model_len
    )
    spec = next(
        (m for m in load_models(manifest) if m["source"] == model or m.get("name") == model),
        None,
    )
    gpus = detect_gpus()
    vram_reader, pid_reader = best_effort_gpu_readers() if isolate else (None, None)
    study_name = study or f"tune-{model.replace('/', '_').replace(':', '_')}"
    typer.echo(f"[tune] study={study_name} model={model} backend={backend} trials={trials}")
    out = two_stage(
        cfg,
        n_trials=trials,
        study_name=study_name,
        model_spec=spec,
        vram_mib=max_vram_mb(gpus),
        ram_mib=detect_ram_mb(),
        seed=seed,
        isolate=isolate,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
    )
    t = out.tune
    typer.echo(
        f"[tune] stage-1 best quality={t.best_value:.4f} "
        f"({t.n_complete} complete, {t.n_pruned} pruned of {t.n_trials})"
    )
    typer.echo(
        f"[tune] winning config: strategy={t.best_config.strategy} "
        f"size={t.best_config.chunk_size} overlap={t.best_config.chunk_overlap} "
        f"top_k={t.best_config.top_k} mode={t.best_config.retrieval_mode}"
    )
    typer.echo("[tune] stage-2 (final split) is the leaderboard entry:")
    typer.echo(out.final["table"])
