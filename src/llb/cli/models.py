"""Model prep, planning, resolution, sweep, and tuning commands."""

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

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
from llb.core.config import RunConfig
from llb.core.contracts import ModelSpec, PreparedModel, ResolvedModel


def _expand_quant_variants(specs: list[ModelSpec]) -> list[ModelSpec]:
    """list-models visibility: expand a multi-quant `sources.vllm` list into one plan row per quant.

    So an operator sees the row the resolver would actually pick on a bigger card -- e.g. the fp8
    Mistral quant on a 32 GiB host -- not just the parent quant the planner prices. Each variant
    inherits the parent arch and overrides source/quant; single-source entries pass through. This is
    display-only and does not affect `resolve-models` / `sweep`, which own backend selection.
    """
    from llb.backends.resolver import normalize_source_list

    out: list[ModelSpec] = []
    for spec in specs:
        vllm = (spec.get("sources") or {}).get("vllm")
        if not isinstance(vllm, list) or len(vllm) <= 1:
            out.append(spec)
            continue
        for record in normalize_source_list(vllm):
            merged = cast(ModelSpec, {**spec, "backend": "vllm", **record})
            if record.get("source") != spec.get("source"):
                quant = record.get("quant")
                merged["name"] = f"{spec['name']}-{quant}" if quant else f"{spec['name']}-vllm"
            out.append(merged)
    return out


@app.command("prep-models")
def prep_models_cmd(
    manifest: Path = typer.Option(
        Path("samples/configs/models_uk.yaml"), help="candidate-models YAML manifest"
    ),
    backend: str = typer.Option("all", help="ollama | vllm | all"),
    force: bool = typer.Option(False, help="prepare even if a model looks too big for VRAM"),
    dry_run: bool = typer.Option(False, help="show the plan; pull/cache nothing"),
    cache_dir: Optional[Path] = typer.Option(None, help="HF cache dir for vLLM weights"),
) -> None:
    """Detect the GPU, pull Ollama tags, and cache vLLM (HF) weights once."""
    from llb.backends.prepare import prepare_models

    models = load_models(manifest)

    def progress(row: PreparedModel) -> None:
        typer.echo(
            f"[prep-models] start    {row['backend']:<6} {row['name']:<22} "
            f"{row['source']}  -- {row['action']}: {row['reason']}"
        )

    report = prepare_models(
        models,
        backend_filter=backend,
        force=force,
        dry_run=dry_run,
        cache_dir=cache_dir,
        progress=progress,
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
        raise SystemExit(1)


@app.command("prep-serving-targets")
def prep_serving_targets_cmd(
    tier_json: Path = typer.Option(..., help="generated serving tier.json from gen-serving-config"),
    backend: str = typer.Option("all", help="ollama | vllm | all"),
    force: bool = typer.Option(False, help="prepare even if a target looks too big for VRAM"),
    dry_run: bool = typer.Option(False, help="show the plan; pull/cache nothing"),
    cache_dir: Optional[Path] = typer.Option(None, help="HF cache dir for vLLM weights"),
) -> None:
    """Pull/cache the concrete models referenced by a generated CUDA-tier serving config."""
    from llb.backends.prepare import load_serving_targets, prepare_models

    models = load_serving_targets(tier_json)

    def progress(row: PreparedModel) -> None:
        typer.echo(
            f"[prep-serving-targets] start    {row['backend']:<6} {row['name']:<22} "
            f"{row['source']}  -- {row['action']}: {row['reason']}"
        )

    report = prepare_models(
        models,
        backend_filter=backend,
        force=force,
        dry_run=dry_run,
        cache_dir=cache_dir,
        progress=progress,
    )

    if report["gpus"]:
        for g in report["gpus"]:
            typer.echo(
                f"[prep-serving-targets] GPU {g.index}: {g.name} "
                f"({g.total_mb} MB total, {g.free_mb} MB free, driver {g.driver})"
            )
    else:
        typer.echo("[prep-serving-targets] no GPU detected (Ollama runs on CPU; vLLM is skipped)")

    for r in report["results"]:
        typer.echo(
            f"[prep-serving-targets] {r['status']:<8} {r['backend']:<6} {r['name']:<22} "
            f"{r['source']}  -- {r['detail']}"
        )
    failed = [r for r in report["results"] if r["status"] == "failed"]
    if failed:
        raise SystemExit(1)


@app.command("list-models")
def list_models_cmd(
    manifest: Path = typer.Option(
        Path("samples/configs/models_uk.yaml"), help="candidate-models YAML manifest"
    ),
    context: Optional[int] = typer.Option(
        None, help="plan at this target context instead of the max the host can hold"
    ),
    vram_reserve: int = typer.Option(1024, help="VRAM MiB held back for CUDA/display"),
    ram_reserve: int = typer.Option(2048, help="RAM MiB held back for the OS"),
    runnable_only: bool = typer.Option(False, help="hide models that cannot run at all"),
    trust_config: bool = typer.Option(
        False,
        "--trust-config",
        help="let a cached config.json OVERRIDE curated arch fields (memory planner), not only fill gaps",
    ),
) -> None:
    """List which candidate models can run here (GPU+RAM, KV-cache-aware, batch=1)."""
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.backends.planner import VERDICT_NO, format_plan, plan_models

    models = _expand_quant_variants(planning_models(manifest, trust_config=trust_config))
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


@app.command("preflight-vllm")
def preflight_vllm_cmd(
    force: bool = typer.Option(
        False, "--force", help="re-probe even when a cached verdict is current for this driver"
    ),
    auto_pin: bool = typer.Option(
        False,
        "--auto-pin",
        help="when the bundled flashinfer fails, pip-install + try candidate versions "
        "(LLB_FLASHINFER_CANDIDATES); CHANGES the environment, so it is opt-in",
    ),
) -> None:
    """Probe the vLLM flashinfer sampler and record the verdict (vLLM serving preflight).

    Reuses a cached verdict when it was recorded under the CURRENT GPU driver; a driver change (or
    --force) re-runs the probe WITHOUT a full `build-vllm`. With --auto-pin it also tries to install
    a host-compatible flashinfer when the bundled one fails. Run this after a driver upgrade."""
    from llb.backends.preflight import configured_candidates, run_preflight

    candidates = configured_candidates() if auto_pin else None
    verdict = run_preflight(force=force, candidates=candidates)
    typer.echo(
        f"[preflight-vllm] sampler={verdict['sampler']} driver={verdict.get('driver')} "
        f"flashinfer={verdict.get('flashinfer_version')} "
        f"auto_pinned={verdict.get('auto_pinned')} -- {verdict['detail']}"
    )


@app.command("resolve-models")
def resolve_models_cmd(
    manifest: Path = typer.Option(
        Path("samples/configs/models_uk.yaml"), help="candidate-models YAML manifest"
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


# Query-time --rag-grid axes: value parser + validity predicate per supported key. Only
# query-time knobs belong here (they retrieve against the SAME index, so no re-index per cell).
# `rerank_candidates` (rerank-context-order): 0 == reranker off; a positive depth enables the
# sweep-level `--reranker` cross-encoder with that candidate pool.
_RAG_GRID_AXES: dict[str, tuple[Any, Any]] = {
    "top_k": (int, lambda v: v >= 1),
    "fusion_weight": (float, lambda v: 0.0 <= v <= 1.0),
    "rerank_candidates": (int, lambda v: v >= 0),
}
_RAG_GRID_USAGE = (
    "--rag-grid must look like 'top_k=3,5,8', 'top_k=3,5;fusion_weight=0.4,0.6', "
    "or 'rerank_candidates=0,30' (0 == reranker off)"
)


def _parse_rag_grid(spec: str | None) -> list[dict[str, Any]]:
    """Parse an opt-in RAG-config grid into per-cell override dicts (axes cross-multiplied).

    Returns `[{}]` (keep the manifest's single config) when no grid is given, so the default
    sweep is unchanged. Supported axes (`;`-separated): `top_k` (retrieval depth) and
    `fusion_weight` (hybrid dense/lexical RRF share; the index must be built with
    `build-index --retrieval-mode hybrid`). Index-time knobs (chunk_size/overlap) are out of
    scope because they need rebuilt indexes.
    """
    if not spec:
        return [{}]
    axes: list[tuple[str, list[Any]]] = []
    for part in spec.split(";"):
        key, sep, raw = part.partition("=")
        key = key.strip()
        if key not in _RAG_GRID_AXES or not sep or not raw.strip():
            raise typer.BadParameter(_RAG_GRID_USAGE)
        if any(key == seen for seen, _ in axes):
            raise typer.BadParameter(f"--rag-grid axis '{key}' given twice")
        cast, valid = _RAG_GRID_AXES[key]
        try:
            values = [cast(v) for v in raw.split(",") if v.strip()]
        except ValueError as exc:
            raise typer.BadParameter(
                f"--rag-grid {key} values must be {cast.__name__}s: {raw!r}"
            ) from exc
        values = list(dict.fromkeys(values))  # de-dupe, preserve order
        if not values or not all(valid(v) for v in values):
            raise typer.BadParameter(f"--rag-grid {key} values out of range: {raw!r}")
        axes.append((key, values))
    points = [{}]  # type: list[dict[str, Any]]
    for key, values in axes:
        points = [{**point, key: value} for point in points for value in values]
    return points


_GRID_SUFFIX_PREFIX = {"top_k": "k", "fusion_weight": "w", "rerank_candidates": "r"}


def _grid_cells(
    base: RunConfig,
    overrides: dict[str, Any],
    rag_grid: list[dict[str, Any]],
    reranker: str | None = None,
) -> list[RunConfig]:
    """One revalidated RunConfig per grid point for a resolved model (a single cell when no grid).

    Every grid knob is a `RunConfig` field and therefore part of the cell fingerprint, so
    distinct grid points get distinct resume keys; the `-k<top_k>`/`-w<fusion_weight>`/
    `-r<rerank_candidates>` run-name suffix only makes the sweep log readable. A `fusion_weight`
    point implies `retrieval_mode=hybrid` (the knob is dead outside hybrid fusion). A
    `rerank_candidates` point of 0 turns the reranker OFF; a positive depth turns it on with
    the sweep-level `reranker` cross-encoder id.
    """
    cells: list[RunConfig] = []
    for point in rag_grid:
        cell = dict(overrides)
        suffix = ""
        for key, value in point.items():
            suffix += (
                f"-{_GRID_SUFFIX_PREFIX[key]}{value:g}"
                if isinstance(value, float)
                else (f"-{_GRID_SUFFIX_PREFIX[key]}{value}")
            )
            if key == "rerank_candidates":
                if value == 0:
                    cell["reranker"] = None
                    continue
                cell["reranker"] = reranker
                cell["rerank_candidates"] = value
                continue
            cell[key] = value
        if "fusion_weight" in point:
            cell["retrieval_mode"] = "hybrid"
        if suffix:
            cell["run_name"] = f"{overrides['run_name']}{suffix}"
        cells.append(base.with_overrides(**cell))
    return cells


def _local_backend_ready(backend: str, data_dir: Path) -> tuple[bool, str]:
    """Return whether the local serving binary needed by a resolved backend is installed."""
    if backend == "vllm":
        from llb.backends.vllm import vllm_executable

        if vllm_executable():
            return True, ""
        return False, "vllm executable not found (run make build-vllm)"
    if backend == "llamacpp":
        built = data_dir / "llb" / "llamacpp" / "build" / "bin" / "llama-server"
        if built.exists() or shutil.which("llama-server"):
            return True, ""
        return False, "llama-server not found (run make build-llamacpp)"
    return True, ""


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
