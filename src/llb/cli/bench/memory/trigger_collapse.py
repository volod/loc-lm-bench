"""CLI orchestration for the compact trigger-versus-guard collapse study."""

from pathlib import Path
from typing import cast

import typer

from llb.cli.app import app


@app.command("bench-agentic-context-compact-trigger-collapse")
def bench_agentic_context_compact_trigger_collapse_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_compact_trigger_guard_collapse_design.json"),
        "--design",
        help="pinned family, equal-trigger families, and the equal-guard contrast",
    ),
) -> None:
    """Re-qualify the pinned family, then test whether share and guard act only through the trigger."""
    from llb.backends.ollama import list_models
    from llb.bench.memory.trigger_collapse.run import analyze_collapse, run_collapse_families
    from llb.bench.memory.trigger_collapse.design import (
        collapse_cap_peaks,
        load_collapse_design,
        validate_collapse_design,
    )
    from llb.bench.memory.trigger_collapse.report import (
        format_collapse_table,
        persist_collapse,
    )
    from llb.bench.memory.transfer.run import run_control_pilot
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config

    try:
        design = load_collapse_design(design_path)
        validate_collapse_design(design)
    except ValueError as exc:
        cli_error(str(exc))
    held = cast(dict[str, object], design["held_fixed"])
    model = cast(str, held["model"])
    backend = cast(str, held["backend"])
    if model not in set(list_models()):
        cli_error(f"the pinned trigger-collapse model is not installed: {model}")
    for depth, peak in collapse_cap_peaks(design).items():
        typer.echo(f"[trigger-collapse] depth={depth} deterministic cap peak={peak} chars")

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
            f"[trigger-collapse] control completion={cast(float, control['completion']):.3f} "
            f"eligible={str(control['eligible']).lower()}"
        )
        if not control["eligible"]:
            return control, []
        return control, run_collapse_families(
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
    analysis = analyze_collapse(design, control_row, grid_rows)
    table = format_collapse_table(analysis)
    paths = persist_collapse(
        design,
        analysis,
        data_dir=cfg.data_dir,
        table=table,
        tokens_per_s=meter.tokens_per_s,
    )
    typer.echo(table)
    typer.echo(f"[trigger-collapse] aggregate -> {paths['manifest']}")
    # A non-collapse verdict is a measured outcome; only an ineligible pinned family is a failure.
    if not control_row["eligible"]:
        raise typer.Exit(code=2)
