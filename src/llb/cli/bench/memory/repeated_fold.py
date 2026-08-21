"""CLI orchestration for repeated-fold compact-memory completion."""

from pathlib import Path
from typing import cast

import typer

from llb.cli.app import app


@app.command("bench-agentic-context-compact-repeated-fold")
def bench_agentic_context_compact_repeated_fold_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_compact_repeated_fold_completion_design.json"),
        "--design",
        help="one-fold control plus the committed repeatedly folding cells",
    ),
) -> None:
    """Measure completion by fold count and attribute survival with a marker ablation."""
    from llb.backends.ollama import list_models
    from llb.bench.memory.repeated_fold.completion import (
        RepeatedFoldRun,
        run_repeated_fold_completion,
    )
    from llb.bench.memory.repeated_fold.design import (
        completion_cells,
        load_repeated_fold_design,
        probe_completion_cell,
        validate_repeated_fold_design,
    )
    from llb.bench.memory.repeated_fold.report import (
        format_repeated_fold_table,
        persist_repeated_fold_run,
    )
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config

    try:
        design = load_repeated_fold_design(design_path)
        validate_repeated_fold_design(design)
    except ValueError as exc:
        cli_error(str(exc))
    held = cast(dict[str, object], design["held_fixed"])
    model = cast(str, held["model"])
    backend = cast(str, held["backend"])
    if backend == "ollama" and model not in set(list_models()):
        cli_error(f"the pinned repeated-fold model is not installed: {model}")
    for cell in completion_cells(design):
        probe = probe_completion_cell(cell, held)
        typer.echo(
            f"[repeated-fold] cell={cell['cell_id']} guard={cell['max_prompt_chars']} "
            f"oracle-folds={probe['oracle_folds']} cap-fitting={str(probe['cap_fitting']).lower()}"
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

    def run(complete: LLMComplete) -> RepeatedFoldRun:
        return run_repeated_fold_completion(
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
    table = format_repeated_fold_table(result.analysis)
    paths = persist_repeated_fold_run(
        design,
        result,
        data_dir=cfg.data_dir,
        table=table,
        tokens_per_s=meter.tokens_per_s,
    )
    typer.echo(table)
    typer.echo(f"[repeated-fold] aggregate -> {paths['manifest']}")
    if not result.analysis["control_eligible"]:
        raise typer.Exit(code=2)
