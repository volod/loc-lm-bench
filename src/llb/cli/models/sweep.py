"""Hard-isolation model sweep + two-stage Optuna tuning commands."""

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
from llb.cli.models.grid import (
    _grid_cells,
    _local_backend_ready,
    _parse_rag_grid,
    _sweep_cell_overrides,
)
from llb.core.config import RunConfig


@app.command("sweep")
def sweep_cmd(
    manifest: Path = typer.Option(
        Path("samples/configs/models_uk.yaml"), help="candidate-models YAML manifest"
    ),
    split: str = typer.Option("final", help="gold split to evaluate each cell on"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL for every cell"),
    sweep_id: Optional[str] = typer.Option(None, help="resume id (default: a UTC timestamp)"),
    max_model_len: int = typer.Option(8192, help="vLLM context cap per cell (KV cache fit)"),
    telemetry: bool = typer.Option(True, help="measure steady-state throughput + peak VRAM"),
    limit: Optional[int] = typer.Option(None, help="cap eval items per sweep cell"),
    resume: bool = typer.Option(True, help="skip cells already completed under this sweep id"),
    offline: bool = typer.Option(False, help="skip availability probes (assume sources exist)"),
    rag_grid: Optional[str] = typer.Option(
        None,
        "--rag-grid",
        help="opt-in retrieval grid, e.g. 'top_k=3,5,8', 'top_k=3,5;fusion_weight=0.4,0.6', or "
        "'rerank_candidates=0,30' -> one cell per (model, grid point); fusion_weight implies "
        "retrieval_mode=hybrid (build the index with --retrieval-mode hybrid first); "
        "rerank_candidates=0 is the reranker-off cell; default keeps the manifest's "
        "single config",
    ),
    reranker: Optional[str] = typer.Option(
        None,
        help="cross-encoder id for positive rerank_candidates grid points "
        "(default BAAI/bge-reranker-v2-m3)",
    ),
) -> None:
    """Run one isolated cell per runnable model (process-per-cell, VRAM gate, thermal cooldown)."""
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.backends.resolver import resolve_all
    from llb.executor.isolation import run_sweep
    from llb.rag.rerank import DEFAULT_RERANKER

    grid = _parse_rag_grid(rag_grid)
    if grid != [{}]:
        typer.echo(f"[sweep] rag-grid {len(grid)} points: {grid}")
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
        ready, reason = _local_backend_ready(r["chosen_backend"], base.data_dir)
        if not ready:
            typer.echo(f"[sweep] skip {r['name']}: {reason}")
            continue
        overrides = _sweep_cell_overrides(r, telemetry, max_model_len)
        if overrides is not None:
            cells.extend(_grid_cells(base, overrides, grid, reranker=reranker or DEFAULT_RERANKER))

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
        limit=limit,
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
        Path("samples/configs/models_uk.yaml"),
        help="manifest to read the model's arch from (VRAM prune)",
    ),
    seed: int = typer.Option(13, help="Optuna sampler seed"),
    isolate: bool = typer.Option(
        False, help="run each trial under the executor VRAM-reclaim + thermal-cooldown gate"
    ),
    extended_chunkers: bool = typer.Option(
        False,
        "--extended-chunkers",
        help="add the page/heading/late chunking strategies to the stage-1 search space "
        "(late re-embeds whole documents per trial; page needs PDF citation sidecars)",
    ),
    tune_reranker: Optional[str] = typer.Option(
        None,
        "--reranker",
        help="add the opt-in rerank axes (reranker on/off + candidate depth) to the stage-1 "
        "search space, using this local cross-encoder id (e.g. BAAI/bge-reranker-v2-m3); "
        "each on-trial reranks per case, so trials get slower",
    ),
) -> None:
    """Two-stage tune: search RAG params on the tuning split, score the winner on final."""
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.optimize.tuner import EXTENDED_STRATEGIES, two_stage

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
        strategies=EXTENDED_STRATEGIES if extended_chunkers else None,
        reranker=tune_reranker,
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
