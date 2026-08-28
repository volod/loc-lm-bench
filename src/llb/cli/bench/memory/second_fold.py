"""CLI orchestration for the second-fold trigger restatement."""

from pathlib import Path
from typing import cast

import typer

from llb.cli.app import app


@app.command("bench-agentic-context-compact-second-fold")
def bench_agentic_context_compact_second_fold_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_compact_second_fold_trigger_design.json"),
        "--design",
        help="equal-trigger family over the committed two-fold geometry, plus its contrast",
    ),
) -> None:
    """Re-qualify the pinned family, then test the trigger rule where episodes fold repeatedly."""
    from llb.backends.ollama import list_models
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.bench.context_policy.report import PolicyReport
    from llb.bench.memory.second_fold.design import validate_second_fold_design
    from llb.bench.memory.second_fold.geometry import (
        load_second_fold_design,
        probe_second_fold_cell,
        second_fold_cells,
    )
    from llb.bench.memory.second_fold.report import (
        format_second_fold_table,
        persist_second_fold,
    )
    from llb.bench.memory.second_fold.run import analyze_second_fold, run_second_fold_cells
    from llb.bench.memory.transfer.run import run_control_pilot
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config

    try:
        design = load_second_fold_design(design_path)
        validate_second_fold_design(design)
    except ValueError as exc:
        cli_error(str(exc))
    held = cast(dict[str, object], design["held_fixed"])
    model = cast(str, held["model"])
    backend = cast(str, held["backend"])
    if backend == "ollama" and model not in set(list_models()):
        cli_error(f"the pinned second-fold model is not installed: {model}")
    for cell in second_fold_cells(design):
        probe = probe_second_fold_cell(cell, held)
        typer.echo(
            f"[second-fold] cell={cell['cell_id']} guard={cell['max_prompt_chars']} "
            f"trigger={probe['compaction_trigger_chars']} folds={probe['oracle_folds']} "
            f"fold-inputs={probe['oracle_fold_input_chars']} "
            f"cap-peak={probe['cap_peak_prompt_chars']}"
        )

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

    def run(
        complete: LLMComplete,
    ) -> tuple[dict[str, object], list[dict[str, object]], dict[str, PolicyReport]]:
        _report, control = run_control_pilot(
            cast(dict[str, object], design["control_recheck"]),
            model=model,
            backend=backend,
            complete=complete,
        )
        control["model_family"] = held["model_family"]
        typer.echo(
            f"[second-fold] control completion={cast(float, control['completion']):.3f} "
            f"eligible={str(control['eligible']).lower()}"
        )
        if not control["eligible"]:
            return control, [], {}
        rows, reports = run_second_fold_cells(
            design, model=model, backend=backend, complete=complete
        )
        return control, rows, reports

    control_row, cell_rows, reports = drive_with_backend(
        cfg,
        run,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        meter=meter,
    )
    control_row["tokens_per_s"] = meter.tokens_per_s
    analysis = analyze_second_fold(design, control_row, cell_rows)
    table = format_second_fold_table(analysis)
    paths = persist_second_fold(
        design,
        analysis,
        reports,
        data_dir=cfg.data_dir,
        table=table,
        tokens_per_s=meter.tokens_per_s,
    )
    typer.echo(table)
    typer.echo(f"[second-fold] aggregate -> {paths['manifest']}")
    # Either reading is a measured outcome; only an ineligible pinned family is a failure.
    if not control_row["eligible"]:
        raise typer.Exit(code=2)
