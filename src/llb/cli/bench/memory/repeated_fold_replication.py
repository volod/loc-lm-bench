"""Roster driving for the two-family repeated-fold completion replication.

The command seam lives in `repeated_fold.py`, which dispatches here on the design's study kind:
one make target, two committed designs, and no second entry point to keep in sync.
"""

from pathlib import Path
from typing import Any, cast

import typer


def run_replication(design_path: Path) -> None:
    """Qualify families control-first and read the fold-count rule across the ones that pass."""
    from llb.backends.ollama import list_models
    from llb.bench.memory.repeated_fold.replication import (
        ReplicationFamilyRun,
        analyze_replication_runs,
    )
    from llb.bench.memory.repeated_fold.replication_design import (
        load_repeated_fold_replication_design,
        replication_roster,
        validate_replication_design,
    )
    from llb.bench.memory.repeated_fold.replication_report import (
        format_replication_table,
        persist_replication_run,
    )
    from llb.cli.helpers import best_effort_gpu_readers, cli_error

    try:
        design = load_repeated_fold_replication_design(design_path)
        validate_replication_design(design)
    except ValueError as exc:
        cli_error(str(exc))
    roster = replication_roster(design)
    available = set(list_models())
    missing = [cast(str, row["model"]) for row in roster if row["model"] not in available]
    if missing:
        cli_error(f"repeated-fold replication roster models are not installed: {missing}")
    _echo_cells(design)
    required = int(cast(int, design["required_qualified_families"]))
    vram_reader, pid_reader = best_effort_gpu_readers()
    runs: list[ReplicationFamilyRun] = []
    data_dir: Path | None = None
    for candidate in roster:
        run, cfg_data_dir = _drive_candidate(
            design, candidate, vram_reader=vram_reader, pid_reader=pid_reader
        )
        runs.append(run)
        data_dir = cfg_data_dir
        typer.echo(
            f"[repeated-fold-replication] family={run.model_family} model={run.model} "
            f"eligible={str(run.analysis['control_eligible']).lower()} "
            f"powered-fold-limit={run.analysis['powered_fold_limit']} "
            f"throughput={run.tokens_per_s:.2f}"
        )
        if sum(bool(item.analysis["control_eligible"]) for item in runs) >= required:
            break
    if data_dir is None:
        cli_error("the repeated-fold replication roster is empty")
    analysis = analyze_replication_runs(design, runs)
    table = format_replication_table(analysis)
    paths = persist_replication_run(
        design, runs, analysis, data_dir=cast(Path, data_dir), table=table
    )
    typer.echo(table)
    typer.echo(f"[repeated-fold-replication] aggregate -> {paths['manifest']}")
    if len(cast(list[str], analysis["qualified_models"])) < required:
        raise typer.Exit(code=2)


def _echo_cells(design: dict[str, object]) -> None:
    from llb.bench.memory.repeated_fold.design import completion_cells, probe_completion_cell

    held = cast(dict[str, object], design["held_fixed"])
    for cell in completion_cells(design):
        probe = probe_completion_cell(cell, held)
        typer.echo(
            f"[repeated-fold-replication] cell={cell['cell_id']} "
            f"guard={cell['max_prompt_chars']} oracle-folds={probe['oracle_folds']} "
            f"cap-fitting={str(probe['cap_fitting']).lower()}"
        )


def _drive_candidate(
    design: dict[str, object],
    candidate: dict[str, object],
    *,
    vram_reader: Any,
    pid_reader: Any,
) -> tuple[Any, Path]:
    from llb.bench.memory.repeated_fold.replication import run_replication_family
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.helpers import load_config

    held = cast(dict[str, object], design["held_fixed"])
    cfg = load_config(
        None,
        model=cast(str, candidate["model"]),
        backend=cast(str, candidate["backend"]),
        max_model_len=int(cast(int, held["max_model_len"])),
        seed=int(cast(int, design["seed"])),
        temperature=0.0,
    )
    meter = ThroughputMeter()

    def execute(complete: LLMComplete) -> Any:
        return run_replication_family(design, candidate, complete=complete)

    run = drive_with_backend(
        cfg, execute, vram_reader=vram_reader, pid_usage_reader=pid_reader, meter=meter
    )
    run.tokens_per_s = meter.tokens_per_s
    run.analysis["tokens_per_s"] = meter.tokens_per_s
    return run, cfg.data_dir
