"""Run and persist a paired sweep over the framework-free agent loop policy."""

import time
from dataclasses import replace
from pathlib import Path

from llb.backends.context_budget import ContextBudget, unbounded_budget
from llb.bench.agentic.episode_state import Clock
from llb.bench.agentic.loop_policy import DEFAULT_REPEAT_FEEDBACK
from llb.bench.agentic.model import AgenticTask
from llb.bench.context_policy.run import task_set_digest
from llb.bench.loop_policy.cell import run_policy_cell
from llb.bench.loop_policy.grid import policy_grid
from llb.bench.loop_policy.report import (
    METHOD,
    AgenticLoopPolicyRun,
    LoopPolicyCell,
    LoopPolicyReport,
    build_recommendation,
    format_policy_table,
    pair_reports,
)
from llb.bench.common import LLMComplete, Mirror, render_board, verified_data_config
from llb.bench.common_backend import ThroughputMeter
from llb.core.contracts.results import BoardRow


def _validate_study_design(
    tasks: list[AgenticTask],
    cells: list[LoopPolicyCell],
    *,
    repeat_power_design: dict[str, object] | None,
    repeat_feedback_design: dict[str, object] | None,
    model_family: str | None,
    run_seed: int | None,
) -> None:
    """Check the prospective design this run is executing, before any model call.

    At most one study at a time: the two designs pin overlapping fields, and a run under both would
    be reported as evidence for a contract neither of them stated.
    """
    if repeat_power_design is not None and repeat_feedback_design is not None:
        raise ValueError("choose either repeat_power_design or repeat_feedback_design")
    if repeat_power_design is not None:
        from llb.bench.loop_policy.power import validate_repeat_power_design

        validate_repeat_power_design(
            repeat_power_design, tasks, cells=cells, model_family=model_family
        )
    if repeat_feedback_design is not None:
        from llb.bench.loop_feedback.run import validate_repeat_feedback_design

        validate_repeat_feedback_design(
            repeat_feedback_design,
            tasks,
            cells=cells,
            model_family=model_family,
            run_seed=run_seed,
        )


def _study_analysis(
    tasks: list[AgenticTask],
    reports: list[LoopPolicyReport],
    *,
    repeat_power_design: dict[str, object] | None,
    repeat_feedback_design: dict[str, object] | None,
    model_family: str | None,
    run_seed: int | None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """The study reading over the measured cells, if this run is executing a study at all."""
    if repeat_power_design is not None:
        from llb.bench.loop_policy.power import analyze_repeat_power

        return (
            analyze_repeat_power(repeat_power_design, tasks, reports, model_family=model_family),
            None,
        )
    if repeat_feedback_design is not None:
        from llb.bench.loop_feedback.run import analyze_repeat_feedback

        return (
            None,
            analyze_repeat_feedback(
                repeat_feedback_design,
                tasks,
                reports,
                model_family=model_family,
                run_seed=run_seed,
            ),
        )
    return None, None


def _render_tables(
    reports: list[LoopPolicyReport], repeat_feedback_analysis: dict[str, object] | None
) -> tuple[list[BoardRow], str]:
    """The board over the cells, the per-policy table, and the feedback table when there is one."""
    board, board_table = render_board(
        [replace(report.run.result, model=report.cell.cell_id) for report in reports]
    )
    table = f"{board_table}\n\n{format_policy_table(reports)}"
    if repeat_feedback_analysis is not None:
        from llb.bench.loop_feedback.report import format_repeat_feedback_table

        table = f"{table}\n\n{format_repeat_feedback_table(repeat_feedback_analysis)}"
    return board, table


def run_agentic_loop_policy(
    tasks: list[AgenticTask],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    max_steps: list[int],
    malformed_policies: list[str],
    repeated_call_policies: list[str],
    repeated_feedback_variants: list[str] | None = None,
    budget: ContextBudget | None = None,
    data_dir: Path | str | None = None,
    run_name: str = METHOD,
    persist: bool = True,
    mirror: Mirror | None = None,
    data_verified: bool = False,
    verification_ref: str | None = None,
    meter: ThroughputMeter | None = None,
    repeat_power_design: dict[str, object] | None = None,
    repeat_feedback_design: dict[str, object] | None = None,
    model_family: str | None = None,
    run_seed: int | None = None,
    clock: Clock = time.monotonic,
) -> AgenticLoopPolicyRun:
    """Measure every cell on the identical task set and recommend one policy per model."""
    if not tasks:
        raise SystemExit("no agentic tasks provided")
    feedback_variants = repeated_feedback_variants or [DEFAULT_REPEAT_FEEDBACK]
    cells = policy_grid(max_steps, malformed_policies, repeated_call_policies, feedback_variants)
    _validate_study_design(
        tasks,
        cells,
        repeat_power_design=repeat_power_design,
        repeat_feedback_design=repeat_feedback_design,
        model_family=model_family,
        run_seed=run_seed,
    )
    resolved_budget = budget if budget is not None else unbounded_budget()
    reports = [
        run_policy_cell(
            tasks,
            cell,
            model=model,
            backend=backend,
            complete=complete,
            budget=resolved_budget,
            meter=meter,
            clock=clock,
        )
        for cell in cells
    ]
    tokens_per_s = meter.tokens_per_s if meter is not None else 0.0
    for report in reports:
        report.run.result = replace(report.run.result, tokens_per_s=tokens_per_s)
    pair_reports(reports)
    repeat_power_analysis, repeat_feedback_analysis = _study_analysis(
        tasks,
        reports,
        repeat_power_design=repeat_power_design,
        repeat_feedback_design=repeat_feedback_design,
        model_family=model_family,
        run_seed=run_seed,
    )
    board, table = _render_tables(reports, repeat_feedback_analysis)
    recommendation = build_recommendation(
        model,
        reports,
        repeat_power_analysis=repeat_power_analysis,
        repeat_feedback_analysis=repeat_feedback_analysis,
    )
    digest = task_set_digest(tasks)
    if persist and data_dir is not None:
        from llb.bench.loop_policy.persist import persist_reports

        persist_reports(
            reports,
            data_dir=data_dir,
            run_name=run_name,
            model=model,
            backend=backend,
            task_digest=digest,
            table=table,
            recommendation=recommendation,
            max_steps=max_steps,
            malformed_policies=malformed_policies,
            repeated_call_policies=repeated_call_policies,
            repeated_feedback_variants=feedback_variants,
            budget=resolved_budget,
            verification_config=verified_data_config(
                data_verified=data_verified,
                verification_ref=verification_ref,
            ),
            model_family=model_family,
            run_seed=run_seed,
            repeat_power_design=repeat_power_design,
            repeat_power_analysis=repeat_power_analysis,
            repeat_feedback_design=repeat_feedback_design,
            repeat_feedback_analysis=repeat_feedback_analysis,
            mirror=mirror,
        )
    return AgenticLoopPolicyRun(
        model=model,
        backend=backend,
        reports=reports,
        board=board,
        table=table,
        recommendation=recommendation,
        task_set_digest=digest,
        repeat_power_analysis=repeat_power_analysis,
        repeat_feedback_analysis=repeat_feedback_analysis,
    )
