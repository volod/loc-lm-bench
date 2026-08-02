"""CLI orchestration for compact-memory cross-model transfer."""

from pathlib import Path
from typing import cast

import typer

from llb.cli.app import app


@app.command("bench-agentic-context-compact-memory-transfer")
def bench_agentic_context_compact_memory_transfer_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_compact_memory_transfer_design.json"),
        "--design",
        help="prospective candidate, control-gate, and depth/trigger contract",
    ),
) -> None:
    """Select the first eligible non-Qwen model and run the compact-memory matrix."""
    from llb.backends.ollama import list_models
    from llb.bench.agentic_memory_transfer import (
        analyze_transfer,
        load_transfer_design,
        run_control_pilot,
        run_transfer_matrix,
        validate_transfer_design,
    )
    from llb.bench.agentic_memory_transfer_report import (
        format_transfer_table,
        persist_transfer,
    )
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config

    try:
        design = load_transfer_design(design_path)
        validate_transfer_design(design)
    except ValueError as exc:
        cli_error(str(exc))
    candidates = cast(list[dict[str, object]], design["candidate_roster"])
    available = set(list_models())
    missing = [str(row["model"]) for row in candidates if row["model"] not in available]
    if missing:
        cli_error(f"compact-memory transfer roster models are not installed: {missing}")

    matrix = cast(dict[str, object], design["matrix"])
    max_model_len = int(cast(int, matrix["max_model_len"]))
    vram_reader, pid_reader = best_effort_gpu_readers()
    pilot_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    selected_meter: ThroughputMeter | None = None
    data_dir: Path | None = None

    for candidate in candidates:
        family = cast(str, candidate["model_family"])
        model = cast(str, candidate["model"])
        backend = cast(str, candidate["backend"])
        typer.echo(f"[compact-memory-transfer] control family={family} model={model}")
        cfg = load_config(
            None,
            model=model,
            backend=backend,
            max_model_len=max_model_len,
            seed=int(cast(int, design["seed"])),
            temperature=0.0,
        )
        data_dir = cfg.data_dir
        meter = ThroughputMeter()

        def run(complete: LLMComplete) -> tuple[dict[str, object], list[dict[str, object]]]:
            _report, pilot = run_control_pilot(
                design,
                model=model,
                backend=backend,
                complete=complete,
            )
            pilot["model_family"] = family
            if not pilot["eligible"]:
                return pilot, []
            typer.echo(f"[compact-memory-transfer] eligible family={family}; running matrix")
            rows = run_transfer_matrix(
                design,
                model=model,
                backend=backend,
                complete=complete,
                data_dir=cfg.data_dir,
            )
            return pilot, rows

        pilot, candidate_matrix = drive_with_backend(
            cfg,
            run,
            vram_reader=vram_reader,
            pid_usage_reader=pid_reader,
            meter=meter,
        )
        pilot["tokens_per_s"] = meter.tokens_per_s
        pilot_rows.append(pilot)
        typer.echo(
            f"[compact-memory-transfer] control completion="
            f"{cast(float, pilot['completion']):.3f} eligible={str(pilot['eligible']).lower()} "
            f"throughput={meter.tokens_per_s:.1f} tok/s"
        )
        if candidate_matrix:
            matrix_rows = candidate_matrix
            selected_meter = meter
            break

    analysis = analyze_transfer(design, pilot_rows, matrix_rows)
    table = format_transfer_table(analysis)
    if data_dir is None:
        cli_error("compact-memory transfer roster is empty")
    paths = persist_transfer(
        design,
        analysis,
        data_dir=data_dir,
        table=table,
        tokens_per_s=selected_meter.tokens_per_s if selected_meter is not None else 0.0,
    )
    typer.echo(table)
    typer.echo(f"[compact-memory-transfer] aggregate -> {paths['manifest']}")
    if analysis["selected_candidate"] is None:
        raise typer.Exit(code=2)
