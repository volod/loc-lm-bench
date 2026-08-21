"""CLI orchestration for two-family middle-critical window-elision transfer."""

from pathlib import Path
from typing import Any, cast

import typer

from llb.cli.app import app


@app.command("bench-agentic-context-compact-window-elision-transfer")
def bench_agentic_context_compact_window_elision_transfer_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_compact_window_elision_transfer_design.json"),
        "--design",
        help="two-family head/middle/tail elision transfer and conditional prototype",
    ),
) -> None:
    """Qualify two families and price task-critical evidence in each trim stratum."""
    from llb.backends.ollama import list_models
    from llb.bench.memory.window_elision.transfer import (
        analyze_transfer_runs,
    )
    from llb.bench.memory.window_elision.transfer_design import (
        load_window_elision_transfer_design,
        validate_window_elision_transfer_design,
    )
    from llb.bench.memory.window_elision.transfer_report import (
        format_window_elision_transfer_table,
        persist_window_elision_transfer,
    )
    from llb.cli.helpers import best_effort_gpu_readers, cli_error

    try:
        design = load_window_elision_transfer_design(design_path)
        validate_window_elision_transfer_design(design)
    except ValueError as exc:
        cli_error(str(exc))
    roster = cast(list[dict[str, object]], design["candidate_roster"])
    available = set(list_models())
    missing = [cast(str, row["model"]) for row in roster if row["model"] not in available]
    if missing:
        cli_error(f"window-elision transfer roster models are not installed: {missing}")
    _echo_placements(design)
    held = cast(dict[str, object], design["held_fixed"])
    required = int(cast(int, design["required_qualified_families"]))
    vram_reader, pid_reader = best_effort_gpu_readers()
    runs, data_dir = _run_roster(
        design,
        roster,
        required=required,
        vram_reader=vram_reader,
        pid_reader=pid_reader,
        max_model_len=int(cast(int, held["max_model_len"])),
    )
    analysis = analyze_transfer_runs(design, runs)
    if analysis["prototype_required"]:
        _run_prototypes(
            design,
            runs,
            qualified_models=set(cast(list[str], analysis["qualified_models"])),
            vram_reader=vram_reader,
            pid_reader=pid_reader,
            max_model_len=int(cast(int, held["max_model_len"])),
        )
        analysis = analyze_transfer_runs(design, runs)
    table = format_window_elision_transfer_table(analysis)
    if data_dir is None:
        cli_error("window-elision transfer roster is empty")
    paths = persist_window_elision_transfer(
        design,
        runs,
        analysis,
        data_dir=data_dir,
        table=table,
    )
    typer.echo(table)
    typer.echo(f"[window-elision-transfer] aggregate -> {paths['manifest']}")
    if len(cast(list[str], analysis["qualified_models"])) < required:
        raise typer.Exit(code=2)


def _echo_placements(design: dict[str, object]) -> None:
    from llb.bench.memory.window_elision.transfer_design import transfer_placements

    for row in transfer_placements(design):
        typer.echo(
            f"[window-elision-transfer] stratum={row['declared_stratum']} "
            f"stage={row['fact_stage']} span={row['fact_start']}:{row['fact_end']} "
            f"kept-head-end={row['head_end']} kept-tail-start={row['tail_start']}"
        )


def _run_roster(
    design: dict[str, object],
    roster: list[dict[str, object]],
    *,
    required: int,
    vram_reader: Any,
    pid_reader: Any,
    max_model_len: int,
) -> tuple[list[Any], Path | None]:
    runs: list[Any] = []
    data_dir: Path | None = None
    for candidate in roster:
        run, cfg_data_dir = _drive_candidate(
            design,
            candidate,
            vram_reader=vram_reader,
            pid_reader=pid_reader,
            max_model_len=max_model_len,
        )
        runs.append(run)
        data_dir = cfg_data_dir
        typer.echo(
            f"[window-elision-transfer] family={run.model_family} "
            f"eligible={str(run.analysis['eligible']).lower()} throughput={run.tokens_per_s:.2f}"
        )
        if sum(bool(item.analysis["eligible"]) for item in runs) >= required:
            break
    return runs, data_dir


def _run_prototypes(
    design: dict[str, object],
    runs: list[Any],
    *,
    qualified_models: set[str],
    vram_reader: Any,
    pid_reader: Any,
    max_model_len: int,
) -> None:
    for run in runs:
        if run.model in qualified_models:
            _drive_prototype(
                design,
                run,
                vram_reader=vram_reader,
                pid_reader=pid_reader,
                max_model_len=max_model_len,
            )


def _drive_candidate(
    design: dict[str, object],
    candidate: dict[str, object],
    *,
    vram_reader: Any,
    pid_reader: Any,
    max_model_len: int,
) -> tuple[Any, Path]:
    from llb.bench.memory.window_elision.transfer import run_transfer_family
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.helpers import load_config

    model = cast(str, candidate["model"])
    backend = cast(str, candidate["backend"])
    cfg = load_config(
        None,
        model=model,
        backend=backend,
        max_model_len=max_model_len,
        seed=int(cast(int, design["seed"])),
        temperature=0.0,
    )
    meter = ThroughputMeter()

    def execute(complete: LLMComplete) -> Any:
        return run_transfer_family(design, candidate, complete=complete)

    run = drive_with_backend(
        cfg,
        execute,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        meter=meter,
    )
    run.tokens_per_s = meter.tokens_per_s
    run.analysis["tokens_per_s"] = meter.tokens_per_s
    return run, cfg.data_dir


def _drive_prototype(
    design: dict[str, object],
    run: Any,
    *,
    vram_reader: Any,
    pid_reader: Any,
    max_model_len: int,
) -> None:
    from llb.bench.memory.window_elision.transfer import (
        TransferFamilyRun,
        run_entry_aware_prototype,
    )
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.helpers import load_config

    family_run = cast(TransferFamilyRun, run)
    cfg = load_config(
        None,
        model=family_run.model,
        backend=family_run.backend,
        max_model_len=max_model_len,
        seed=int(cast(int, design["seed"])),
        temperature=0.0,
    )
    meter = ThroughputMeter()

    def execute(complete: LLMComplete) -> None:
        run_entry_aware_prototype(design, family_run, complete=complete)

    drive_with_backend(
        cfg,
        execute,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        meter=meter,
    )
    family_run.prototype_tokens_per_s = meter.tokens_per_s
