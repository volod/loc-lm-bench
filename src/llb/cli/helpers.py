"""Shared CLI helpers (config load, manifest, GPU readers, resolver probes)."""

from pathlib import Path
from typing import Any, Optional

import typer

from llb.config import RunConfig
from llb.contracts import ModelSpec


def load_config(config_path: Optional[Path], **overrides: Any) -> RunConfig:
    try:
        base = RunConfig.load(config_path) if config_path else RunConfig()
        return base.with_overrides(**overrides)
    except ValueError as exc:
        typer.echo(f"[error] invalid run config: {exc}", err=True)
        raise typer.Exit(code=2) from None


def load_models(manifest: Path) -> list[ModelSpec]:
    """Load a models manifest, reporting a YAML/schema error as a clean one-liner."""
    from llb.backends.prepare import load_manifest

    try:
        return load_manifest(manifest)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None


def planning_models(manifest: Path) -> list[ModelSpec]:
    """Manifest models with missing arch fields filled from a cached config.json (M4.1).

    Offline + best-effort: it sharpens the embedding-aware VRAM estimate when weights are
    already cached, and is a no-op otherwise."""
    from llb.backends.planner import enrich_arch

    return [enrich_arch(m) for m in load_models(manifest)]


def best_effort_gpu_readers() -> tuple[Any, Any]:
    """Best-effort (vram_reader, pid_usage_reader) for the VRAM-reclaim + leak-attribution gate.

    Both are None when the [telemetry] extra / a GPU is absent (the gate then no-ops)."""
    try:
        from llb.executor.vram import nvml_process_reader, nvml_reader

        return nvml_reader(), nvml_process_reader()
    except (Exception, SystemExit):
        return None, None


def resolver_probes(offline: bool) -> Any:
    """Availability probes for model resolution; offline mode assumes every source exists."""
    from llb.backends.resolver import ResolverProbes

    if offline:
        return ResolverProbes(
            hf_repo=lambda _s: True, gguf=lambda _s: True, ollama_tag=lambda _s: True
        )
    return ResolverProbes()


def echo_gpus(prefix: str) -> None:
    """Print detected GPUs or a no-GPU fallback line."""
    from llb.backends.hardware import detect_gpus

    gpus = detect_gpus()
    if gpus:
        for g in gpus:
            typer.echo(f"[{prefix}] GPU {g.index}: {g.name} ({g.total_mb} MiB)")
    else:
        typer.echo(f"[{prefix}] no GPU detected -- planning against system RAM only")
