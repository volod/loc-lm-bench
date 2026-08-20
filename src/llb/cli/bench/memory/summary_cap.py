"""CLI orchestration for the compact summarize-input-cap study."""

from pathlib import Path
from typing import cast

import typer

from llb.cli.app import app


@app.command("bench-agentic-context-compact-summary-input-cap")
def bench_agentic_context_compact_summary_input_cap_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_compact_summary_input_cap_design.json"),
        "--design",
        help="pinned family plus one fold-step ladder run under each summarize-input cap",
    ),
) -> None:
    """Re-qualify the pinned family, then price the summarize call's input cap on one ladder."""
    from llb.backends.ollama import list_models
    from llb.bench.memory.summary_cap.run import analyze_summary_cap, run_summary_cap_arms
    from llb.bench.memory.summary_cap.design import (
        arm_fold_input_probes,
        load_summary_cap_design,
        summary_cap_prompt_sequence,
        validate_summary_cap_design,
    )
    from llb.bench.memory.summary_cap.report import (
        format_summary_cap_table,
        persist_summary_cap,
    )
    from llb.bench.memory.transfer.run import run_control_pilot
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config

    try:
        design = load_summary_cap_design(design_path)
        validate_summary_cap_design(design)
    except ValueError as exc:
        cli_error(str(exc))
    held = cast(dict[str, object], design["held_fixed"])
    model = cast(str, held["model"])
    backend = cast(str, held["backend"])
    if model not in set(list_models()):
        cli_error(f"the pinned summarize-input-cap model is not installed: {model}")
    typer.echo(f"[summary-cap] cap prompt sequence={summary_cap_prompt_sequence(design)}")
    for probe in arm_fold_input_probes(design):
        typer.echo(
            f"[summary-cap] probe arm={probe['arm_id']} cell={probe['cell_id']} "
            f"offered={probe['summary_input_chars']} elided={probe['summary_input_elided_chars']}"
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

    def run(complete: LLMComplete) -> tuple[dict[str, object], list[dict[str, object]]]:
        _report, control = run_control_pilot(
            cast(dict[str, object], design["control_recheck"]),
            model=model,
            backend=backend,
            complete=complete,
        )
        control["model_family"] = held["model_family"]
        typer.echo(
            f"[summary-cap] control completion={cast(float, control['completion']):.3f} "
            f"eligible={str(control['eligible']).lower()}"
        )
        if not control["eligible"]:
            return control, []
        return control, run_summary_cap_arms(
            design,
            model=model,
            backend=backend,
            complete=complete,
            data_dir=cfg.data_dir,
        )

    control_row, arm_rows = drive_with_backend(
        cfg,
        run,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        meter=meter,
    )
    control_row["tokens_per_s"] = meter.tokens_per_s
    analysis = analyze_summary_cap(design, control_row, arm_rows)
    table = format_summary_cap_table(analysis)
    paths = persist_summary_cap(
        design,
        analysis,
        data_dir=cfg.data_dir,
        table=table,
        tokens_per_s=meter.tokens_per_s,
    )
    typer.echo(table)
    typer.echo(f"[summary-cap] aggregate -> {paths['manifest']}")
    # An unconfirmed reading is a measured outcome; only an ineligible pinned family is a failure.
    if not control_row["eligible"]:
        raise typer.Exit(code=2)
