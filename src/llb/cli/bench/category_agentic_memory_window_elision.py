"""CLI orchestration for completion under unavoidable window-bound summary elision."""

from pathlib import Path
from typing import cast

import typer

from llb.cli.app import app


@app.command("bench-agentic-context-compact-window-elision")
def bench_agentic_context_compact_window_elision_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_compact_window_elision_design.json"),
        "--design",
        help="trigger-matched transcript-fitting and window-elided cells",
    ),
) -> None:
    """Measure completion when a folded transcript cannot fit its summarize call."""
    from llb.backends.ollama import list_models
    from llb.bench.agentic_memory_window_elision import (
        WindowElisionRun,
        run_window_elision,
    )
    from llb.bench.agentic_memory_window_elision_design import (
        elision_cells,
        load_window_elision_design,
        probe_elision_cell,
        validate_window_elision_design,
    )
    from llb.bench.agentic_memory_window_elision_report import (
        format_window_elision_table,
        persist_window_elision_run,
    )
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config

    try:
        design = load_window_elision_design(design_path)
        validate_window_elision_design(design)
    except ValueError as exc:
        cli_error(str(exc))
    held = cast(dict[str, object], design["held_fixed"])
    model = cast(str, held["model"])
    backend = cast(str, held["backend"])
    if backend == "ollama" and model not in set(list_models()):
        cli_error(f"the pinned window-elision model is not installed: {model}")
    for cell in elision_cells(design):
        probe = probe_elision_cell(cell, held)
        typer.echo(
            f"[window-elision] cell={cell['cell_id']} role={cell['role']} "
            f"trigger={probe['compaction_trigger_chars']} input={probe['summary_input_chars']} "
            f"elided={probe['summary_input_elided_chars']}"
        )
    cfg = load_config(
        None,
        model=model,
        backend=backend,
        max_model_len=int(cast(int, held["max_model_len"])),
        seed=int(cast(int, design["seed"])),
        temperature=0.0,
    )
    meter = ThroughputMeter()
    vram_reader, pid_reader = best_effort_gpu_readers()

    def run(complete: LLMComplete) -> WindowElisionRun:
        return run_window_elision(
            design,
            model=model,
            backend=backend,
            complete=complete,
        )

    result = drive_with_backend(
        cfg,
        run,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        meter=meter,
    )
    table = format_window_elision_table(result.analysis)
    paths = persist_window_elision_run(
        design,
        result,
        data_dir=cfg.data_dir,
        table=table,
        tokens_per_s=meter.tokens_per_s,
    )
    typer.echo(table)
    typer.echo(f"[window-elision] aggregate -> {paths['manifest']}")
    if not result.analysis["comparison_eligible"]:
        raise typer.Exit(code=2)
