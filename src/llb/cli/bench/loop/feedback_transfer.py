"""CLI entry point for Gemma repeat-feedback task-family transfer."""

from pathlib import Path

import typer

from llb.cli.app import app
from llb.cli.bench.loop.feedback_neutral import run_neutral_feedback_study


@app.command("bench-agentic-loop-repeat-feedback-task-family-transfer")
def bench_agentic_loop_repeat_feedback_task_family_transfer_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_loop_feedback_task_family_transfer_design.json"),
        "--design",
        help="immutable notice, fresh ledger digest, seeds, response floor, and paired gates",
    ),
    tasks_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_loop_feedback_task_family_transfer.json"),
        "--tasks",
        help="fresh balanced read, calculator, search, and mutation holdout ledger",
    ),
    data_verified: bool = typer.Option(False, help="stamp human-verified task data"),
    verification_ref: str | None = typer.Option(
        None, help="verification worksheet, sample manifest, or accepted ledger"
    ),
) -> None:
    """Run the fixed Gemma candidate on two seeds and persist the transfer decision."""
    from llb.bench.loop_feedback.transfer import analyze_feedback_transfer
    from llb.bench.loop_feedback.transfer_design import validate_feedback_transfer_design
    from llb.bench.loop_feedback.transfer_report import persist_feedback_transfer

    run_neutral_feedback_study(
        design_path,
        tasks_path,
        data_verified=data_verified,
        verification_ref=verification_ref,
        validate_design=validate_feedback_transfer_design,
        analyze=analyze_feedback_transfer,
        persist=persist_feedback_transfer,
        log_label="feedback-transfer",
    )
