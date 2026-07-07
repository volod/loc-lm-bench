"""GPU tier detection and serving-config generation commands."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


@app.command("detect-gpu-vram")
def detect_gpu_vram_cmd() -> None:
    """Print the supported GPU VRAM tier (12/16/24/32 GiB) for this host."""
    from llb.inference.generate import detect_gpu_tier, format_detect_line

    typer.echo(format_detect_line(detect_gpu_tier()))


@app.command("gen-serving-config")
def gen_serving_config_cmd(
    gpu_gb: Optional[int] = typer.Option(
        None, help="GPU VRAM tier in GiB (12, 16, 24, 32); default: detect from nvidia-smi"
    ),
    manifest: Path = typer.Option(
        Path("samples/config-example/manifest.yaml"),
        help="tier manifest with model + vLLM knobs",
    ),
    output: Optional[Path] = typer.Option(
        None, help="output directory (default: .data/llb/serving/gpu-<tier>gb/)"
    ),
) -> None:
    """Generate serve scripts and run-eval YAML for the largest models on this GPU tier."""
    from llb.inference.generate import generate_serving_configs, resolve_tier
    from llb.core.paths import PROJECT_ROOT

    manifest_path = manifest.resolve()
    out = generate_serving_configs(
        gpu_gb=gpu_gb,
        output_root=output.resolve() if output else None,
        manifest_path=manifest_path,
    )
    info = resolve_tier(gpu_gb)
    rel = out.resolve().relative_to(PROJECT_ROOT.resolve())
    typer.echo(f"[gen-serving-config] tier={info.tier_gb} GiB gpu_mb={info.total_mb} -> {rel}/")
    typer.echo(f"[gen-serving-config] see {rel / 'tier.json'} for serve/run script names")
