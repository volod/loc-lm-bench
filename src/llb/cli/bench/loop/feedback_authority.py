"""CLI entry point for Gemma controller-authority feedback transfer."""

from pathlib import Path

import typer

from llb.cli.app import app
from llb.cli.bench.loop.feedback_neutral import run_neutral_feedback_study


@app.command("bench-agentic-loop-repeat-feedback-controller-authority-transfer")
def bench_agentic_loop_repeat_feedback_controller_authority_transfer_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_loop_feedback_controller_authority_design.json"),
        "--design",
        help="immutable authority notice, fresh digest, seeds, and prospective gates",
    ),
    tasks_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_loop_feedback_controller_authority.json"),
        "--tasks",
        help="fresh balanced read, calculator, search, and mutation holdout ledger",
    ),
    data_verified: bool = typer.Option(False, help="stamp human-verified task data"),
    verification_ref: str | None = typer.Option(
        None, help="verification worksheet, sample manifest, or accepted ledger"
    ),
) -> None:
    """Run the authority-framed Gemma candidate and persist the two-seed decision."""
    from llb.bench.loop_feedback.authority import (
        analyze_feedback_authority,
        validate_feedback_authority_design,
    )
    from llb.bench.loop_feedback.authority_report import persist_feedback_authority

    run_neutral_feedback_study(
        design_path,
        tasks_path,
        data_verified=data_verified,
        verification_ref=verification_ref,
        validate_design=validate_feedback_authority_design,
        analyze=analyze_feedback_authority,
        persist=persist_feedback_authority,
        log_label="feedback-authority",
    )
