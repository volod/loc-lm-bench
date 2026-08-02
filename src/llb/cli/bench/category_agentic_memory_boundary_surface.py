"""CLI orchestration for the cap-fitting compact-memory boundary surface."""

from pathlib import Path
from typing import cast

import typer

from llb.cli.app import app


@app.command("bench-agentic-context-compact-memory-boundary-surface")
def bench_agentic_context_compact_memory_boundary_surface_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_compact_memory_boundary_surface_design.json"),
        "--design",
        help="pinned family, held-fixed contract, and the predeclared cap-fitting guard grid",
    ),
) -> None:
    """Re-qualify the pinned family, then map the cap-fitting cost crossover over depth/guard."""
    from llb.backends.ollama import list_models
    from llb.bench.agentic_memory_boundary_surface import (
        analyze_surface,
        load_surface_design,
        run_surface_grid,
        surface_cap_peaks,
        validate_surface_design,
    )
    from llb.bench.agentic_memory_boundary_surface_report import (
        format_surface_table,
        persist_surface,
    )
    from llb.bench.agentic_memory_transfer import run_control_pilot
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config

    try:
        design = load_surface_design(design_path)
        validate_surface_design(design)
    except ValueError as exc:
        cli_error(str(exc))
    held = cast(dict[str, object], design["held_fixed"])
    model = cast(str, held["model"])
    backend = cast(str, held["backend"])
    if model not in set(list_models()):
        cli_error(f"the pinned boundary-surface model is not installed: {model}")
    for depth, peak in surface_cap_peaks(design).items():
        typer.echo(f"[boundary-surface] depth={depth} deterministic cap peak={peak} chars")

    cfg = load_config(
        None,
        model=model,
        backend=backend,
        max_model_len=int(cast(int, held["max_model_len"])),
        seed=int(cast(int, design["seed"])),
        temperature=0.0,
    )
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()

    def run(complete: LLMComplete) -> tuple[dict[str, object], list[dict[str, object]]]:
        _report, control = run_control_pilot(
            cast(dict[str, object], design["control_recheck"]),
            model=model,
            backend=backend,
            complete=complete,
        )
        control["model_family"] = held["model_family"]
        typer.echo(
            f"[boundary-surface] control completion={cast(float, control['completion']):.3f} "
            f"eligible={str(control['eligible']).lower()}"
        )
        if not control["eligible"]:
            return control, []
        return control, run_surface_grid(
            design,
            model=model,
            backend=backend,
            complete=complete,
            data_dir=cfg.data_dir,
        )

    control_row, grid_rows = drive_with_backend(
        cfg,
        run,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        meter=meter,
    )
    control_row["tokens_per_s"] = meter.tokens_per_s
    analysis = analyze_surface(design, control_row, grid_rows)
    table = format_surface_table(analysis)
    paths = persist_surface(
        design,
        analysis,
        data_dir=cfg.data_dir,
        table=table,
        tokens_per_s=meter.tokens_per_s,
    )
    typer.echo(table)
    typer.echo(f"[boundary-surface] aggregate -> {paths['manifest']}")
    if not control_row["eligible"]:
        raise typer.Exit(code=2)
