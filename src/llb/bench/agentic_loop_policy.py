"""Run and persist a paired sweep over the framework-free agent loop policy."""

from dataclasses import replace
from pathlib import Path

from llb.bench.agentic.context_budget import ContextBudget, unbounded_budget
from llb.bench.agentic.loop_policy import (
    DEFAULT_REPEAT_FEEDBACK,
    MALFORMED_POLICIES,
    REPEATED_CALL_POLICIES,
    REPEATED_NOOP,
    REPEAT_FEEDBACK_VARIANTS,
    LoopPolicy,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_context import task_set_digest
from llb.bench.agentic_loop_policy_cell import run_policy_cell
from llb.bench.agentic_loop_policy_report import (
    BASELINE_MAX_STEPS,
    BASELINE_POLICY,
    METHOD,
    AgenticLoopPolicyRun,
    LoopPolicyCell,
    build_recommendation,
    format_policy_table,
    pair_reports,
)
from llb.bench.common import LLMComplete, Mirror, render_board, verified_data_config
from llb.bench.common_backend import ThroughputMeter


def policy_grid(
    max_steps: list[int],
    malformed_policies: list[str],
    repeated_call_policies: list[str],
    repeated_feedback_variants: list[str] | None = None,
) -> list[LoopPolicyCell]:
    """Validate and expand the grid, requiring its exact legacy baseline."""
    if not max_steps or any(value < 1 for value in max_steps):
        raise SystemExit("agent max steps must be a non-empty list of positive integers")
    unknown_malformed = [name for name in malformed_policies if name not in MALFORMED_POLICIES]
    if unknown_malformed:
        raise SystemExit(
            f"unknown malformed-call policies: {unknown_malformed}; choose from {MALFORMED_POLICIES}"
        )
    unknown_repeated = [
        name for name in repeated_call_policies if name not in REPEATED_CALL_POLICIES
    ]
    if unknown_repeated:
        raise SystemExit(
            f"unknown repeated-call policies: {unknown_repeated}; "
            f"choose from {REPEATED_CALL_POLICIES}"
        )
    feedback_variants = repeated_feedback_variants or [DEFAULT_REPEAT_FEEDBACK]
    unknown_feedback = [name for name in feedback_variants if name not in REPEAT_FEEDBACK_VARIANTS]
    if unknown_feedback:
        raise SystemExit(
            f"unknown repeat-feedback variants: {unknown_feedback}; "
            f"choose from {REPEAT_FEEDBACK_VARIANTS}"
        )
    cells = [
        LoopPolicyCell(steps, LoopPolicy(malformed, repeated, feedback))
        for steps in dict.fromkeys(max_steps)
        for malformed in dict.fromkeys(malformed_policies)
        for repeated in dict.fromkeys(repeated_call_policies)
        for feedback in (
            dict.fromkeys(feedback_variants)
            if repeated == REPEATED_NOOP
            else [DEFAULT_REPEAT_FEEDBACK]
        )
    ]
    if not any(cell.is_baseline for cell in cells):
        raise SystemExit(
            f"grid must include baseline max_steps={BASELINE_MAX_STEPS}, "
            f"malformed={BASELINE_POLICY.malformed_call}, "
            f"repeated={BASELINE_POLICY.repeated_call}"
        )
    return cells


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
) -> AgenticLoopPolicyRun:
    """Measure every cell on the identical task set and recommend one policy per model."""
    if not tasks:
        raise SystemExit("no agentic tasks provided")
    feedback_variants = repeated_feedback_variants or [DEFAULT_REPEAT_FEEDBACK]
    cells = policy_grid(
        max_steps,
        malformed_policies,
        repeated_call_policies,
        feedback_variants,
    )
    if repeat_power_design is not None and repeat_feedback_design is not None:
        raise ValueError("choose either repeat_power_design or repeat_feedback_design")
    if repeat_power_design is not None:
        from llb.bench.agentic_loop_policy_power import validate_repeat_power_design

        validate_repeat_power_design(
            repeat_power_design,
            tasks,
            cells=cells,
            model_family=model_family,
        )
    if repeat_feedback_design is not None:
        from llb.bench.agentic_loop_feedback import validate_repeat_feedback_design

        validate_repeat_feedback_design(
            repeat_feedback_design,
            tasks,
            cells=cells,
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
        )
        for cell in cells
    ]
    tokens_per_s = meter.tokens_per_s if meter is not None else 0.0
    for report in reports:
        report.run.result = replace(report.run.result, tokens_per_s=tokens_per_s)
    pair_reports(reports)
    repeat_power_analysis = None
    repeat_feedback_analysis = None
    if repeat_power_design is not None:
        from llb.bench.agentic_loop_policy_power import analyze_repeat_power

        repeat_power_analysis = analyze_repeat_power(
            repeat_power_design,
            tasks,
            reports,
            model_family=model_family,
        )
    if repeat_feedback_design is not None:
        from llb.bench.agentic_loop_feedback import (
            analyze_repeat_feedback,
        )
        from llb.bench.agentic_loop_feedback_report import format_repeat_feedback_table

        repeat_feedback_analysis = analyze_repeat_feedback(
            repeat_feedback_design,
            tasks,
            reports,
            model_family=model_family,
            run_seed=run_seed,
        )
    board, board_table = render_board(
        [replace(report.run.result, model=report.cell.cell_id) for report in reports]
    )
    policy_table = format_policy_table(reports)
    table = f"{board_table}\n\n{policy_table}"
    if repeat_feedback_analysis is not None:
        table = f"{table}\n\n{format_repeat_feedback_table(repeat_feedback_analysis)}"
    recommendation = build_recommendation(
        model,
        reports,
        repeat_power_analysis=repeat_power_analysis,
        repeat_feedback_analysis=repeat_feedback_analysis,
    )
    digest = task_set_digest(tasks)
    if persist and data_dir is not None:
        from llb.bench.agentic_loop_policy_persist import persist_reports

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
