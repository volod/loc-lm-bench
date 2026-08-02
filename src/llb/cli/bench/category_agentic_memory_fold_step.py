"""CLI orchestration for the compact fold-step crossover study."""

from pathlib import Path
from typing import cast

import typer

from llb.cli.app import app


@app.command("bench-agentic-context-compact-fold-step")
def bench_agentic_context_compact_fold_step_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_compact_fold_step_crossover_design.json"),
        "--design",
        help="pinned family plus one guard ladder per depth, placed one fold step apart",
    ),
) -> None:
    """Re-qualify the pinned family, then test whether the cost side flips at a fold-step change."""
    from llb.backends.ollama import list_models
    from llb.bench.agentic_memory_fold_step import analyze_fold_steps, run_fold_step_ladders
    from llb.bench.agentic_memory_fold_step_design import (
        fold_step_prompt_sequences,
        load_fold_step_design,
        validate_fold_step_design,
    )
    from llb.bench.agentic_memory_fold_step_report import (
        format_fold_step_table,
        persist_fold_steps,
    )
    from llb.bench.agentic_memory_transfer import run_control_pilot
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config

    try:
        design = load_fold_step_design(design_path)
        validate_fold_step_design(design)
    except ValueError as exc:
        cli_error(str(exc))
    held = cast(dict[str, object], design["held_fixed"])
    model = cast(str, held["model"])
    backend = cast(str, held["backend"])
    if model not in set(list_models()):
        cli_error(f"the pinned fold-step model is not installed: {model}")
    for depth, sequence in fold_step_prompt_sequences(design).items():
        typer.echo(f"[fold-step] depth={depth} deterministic cap prompt sequence={sequence}")

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
            f"[fold-step] control completion={cast(float, control['completion']):.3f} "
            f"eligible={str(control['eligible']).lower()}"
        )
        if not control["eligible"]:
            return control, []
        return control, run_fold_step_ladders(
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
    analysis = analyze_fold_steps(design, control_row, grid_rows)
    table = format_fold_step_table(analysis)
    paths = persist_fold_steps(
        design,
        analysis,
        data_dir=cfg.data_dir,
        table=table,
        tokens_per_s=meter.tokens_per_s,
    )
    typer.echo(table)
    typer.echo(f"[fold-step] aggregate -> {paths['manifest']}")
    # An unconfirmed boundary is a measured outcome; only an ineligible pinned family is a failure.
    if not control_row["eligible"]:
        raise typer.Exit(code=2)
