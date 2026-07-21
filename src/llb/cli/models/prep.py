"""Model prep, planning, and resolution commands."""

from pathlib import Path
from typing import Optional, cast

import typer

from llb.cli.app import app
from llb.cli.helpers import echo_gpus, load_models, planning_models, resolver_probes
from llb.core.contracts.models import ModelSpec, PreparedModel


def _expand_quant_variants(specs: list[ModelSpec]) -> list[ModelSpec]:
    """list-models visibility: expand a multi-quant `sources.vllm` list into one plan row per quant.

    So an operator sees the row the resolver would actually pick on a bigger card -- e.g. the fp8
    Mistral quant on a 32 GiB host -- not just the parent quant the planner prices. Each variant
    inherits the parent arch and overrides source/quant; single-source entries pass through. This is
    display-only and does not affect `resolve-models` / `sweep`, which own backend selection.
    """
    from llb.backends.resolver_sources import normalize_source_list

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
    from llb.backends.prepare.run import prepare_models

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
    from llb.backends.prepare.manifest import load_serving_targets
    from llb.backends.prepare.run import prepare_models

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
    from llb.backends.planner.format import format_plan
    from llb.backends.planner.plan import plan_models
    from llb.backends.resolver_feasibility import MIN_SERVING_CTX, backend_fits

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
        rows = [r for r in rows if backend_fits(r["backend"], r)]
    typer.echo(format_plan(rows, max(0, vram_mib - vram_reserve), max(0, ram_mib - ram_reserve)))
    runnable = sum(1 for r in rows if backend_fits(r["backend"], r))
    typer.echo(f"[list-models] runnable here: {runnable} of {len(rows)}")
    typer.echo(
        f"[list-models] runnable is backend-aware at >={MIN_SERVING_CTX} tokens: "
        "vLLM requires ctx_gpu; "
        "Ollama/llama.cpp may use ctx_max with CPU offload. Verdict remains the ctx_max plan."
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
    from llb.backends.resolver import resolve_all
    from llb.backends.resolver_report import format_resolution

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
