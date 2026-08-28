"""CLI orchestration for the entry-aware summary-fold adoption study."""

from pathlib import Path
from typing import Any, cast

import typer

from llb.cli.app import app


@app.command("bench-agentic-context-summary-trim-adoption")
def bench_agentic_context_summary_trim_adoption_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_summary_trim_adoption_design.json"),
        "--design",
        help="workload set, arms, and roster for the entry-aware summary-fold adoption study",
    ),
    audit_only: bool = typer.Option(
        False,
        "--audit-only",
        help="print the model-free geometry and policy-change audit without warming a model",
    ),
) -> None:
    """Compare both summary-fold trim strategies across every workload on two model families."""
    from llb.bench.summary_trim.analysis import analyze_summary_trim_runs, audit_default_change
    from llb.bench.summary_trim.design import (
        load_summary_trim_design,
        validate_summary_trim_design,
    )
    from llb.bench.summary_trim.report import (
        format_summary_trim_table,
        persist_summary_trim_adoption,
    )
    from llb.cli.helpers import best_effort_gpu_readers, cli_error

    try:
        design = load_summary_trim_design(design_path)
        validate_summary_trim_design(design)
    except ValueError as exc:
        cli_error(str(exc))
    audit = audit_default_change()
    if audit_only:
        typer.echo(format_summary_trim_table(analyze_summary_trim_runs(design, [], audit=audit)))
        return
    roster = cast(list[dict[str, object]], design["candidate_roster"])
    _require_installed(roster)
    held = cast(dict[str, object], design["held_fixed"])
    vram_reader, pid_reader = best_effort_gpu_readers()
    runs, data_dir = _run_roster(
        design,
        roster,
        required=int(cast(int, design["required_qualified_families"])),
        vram_reader=vram_reader,
        pid_reader=pid_reader,
        max_model_len=int(cast(int, held["max_model_len"])),
    )
    if data_dir is None:
        cli_error("the adoption roster is empty")
    analysis = analyze_summary_trim_runs(design, runs, audit=audit)
    table = format_summary_trim_table(analysis)
    paths = persist_summary_trim_adoption(design, runs, analysis, data_dir=data_dir, table=table)
    typer.echo(table)
    typer.echo(f"[summary-trim-adoption] aggregate -> {paths['manifest']}")
    if len(cast(list[str], analysis["qualified_models"])) < int(
        cast(int, design["required_qualified_families"])
    ):
        raise typer.Exit(code=2)


def _require_installed(roster: list[dict[str, object]]) -> None:
    from llb.backends.ollama import list_models
    from llb.cli.helpers import cli_error

    available = set(list_models())
    missing = [cast(str, row["model"]) for row in roster if row["model"] not in available]
    if len(roster) - len(missing) < 2:
        cli_error(f"fewer than two adoption roster models are installed (missing: {missing})")


def _run_roster(
    design: dict[str, object],
    roster: list[dict[str, object]],
    *,
    required: int,
    vram_reader: Any,
    pid_reader: Any,
    max_model_len: int,
) -> tuple[list[Any], Path | None]:
    """Walk the roster in order, stopping once enough families have qualified."""
    from llb.backends.ollama import list_models
    from llb.bench.summary_trim.analysis import family_eligibility

    available = set(list_models())
    runs: list[Any] = []
    data_dir: Path | None = None
    qualified = 0
    for candidate in roster:
        if candidate["model"] not in available:
            typer.echo(f"[summary-trim-adoption] skipping uninstalled {candidate['model']}")
            continue
        run, cfg_data_dir = _drive_candidate(
            design,
            candidate,
            vram_reader=vram_reader,
            pid_reader=pid_reader,
            max_model_len=max_model_len,
        )
        runs.append(run)
        data_dir = cfg_data_dir
        eligible, reason = family_eligibility(design, run)
        qualified += 1 if eligible else 0
        typer.echo(
            f"[summary-trim-adoption] family={run.model_family} "
            f"eligible={str(eligible).lower()} ({reason}) throughput={run.tokens_per_s:.2f}"
        )
        if qualified >= required:
            break
    return runs, data_dir


def _drive_candidate(
    design: dict[str, object],
    candidate: dict[str, object],
    *,
    vram_reader: Any,
    pid_reader: Any,
    max_model_len: int,
) -> tuple[Any, Path]:
    from llb.bench.summary_trim.run import run_summary_trim_family
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.helpers import load_config

    cfg = load_config(
        None,
        model=cast(str, candidate["model"]),
        backend=cast(str, candidate["backend"]),
        max_model_len=max_model_len,
        seed=int(cast(int, design["seed"])),
        temperature=0.0,
    )
    meter = ThroughputMeter()

    def execute(complete: LLMComplete) -> Any:
        return run_summary_trim_family(design, candidate, complete=complete)

    run = drive_with_backend(
        cfg, execute, vram_reader=vram_reader, pid_usage_reader=pid_reader, meter=meter
    )
    run.tokens_per_s = meter.tokens_per_s
    return run, cfg.data_dir
