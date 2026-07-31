"""Run and persist a paired sweep over the framework-free agent loop policy."""

import logging
from dataclasses import replace
from pathlib import Path

from llb.bench.agentic.context import ContextPolicy
from llb.bench.agentic.context_budget import ContextBudget, unbounded_budget
from llb.bench.agentic.loop_policy import (
    MALFORMED_POLICIES,
    REPEATED_CALL_POLICIES,
    LoopPolicy,
)
from llb.bench.agentic.model import HARNESS_LOOP, AgenticTask, Episode
from llb.bench.agentic.run import run_agentic
from llb.bench.agentic_context import task_set_digest
from llb.bench.agentic_loop_policy_report import (
    BASELINE_MAX_STEPS,
    BASELINE_POLICY,
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
from llb.bench.harness.base import loop_harness
from llb.core.contracts.benchmarks import ToolDef

_LOG = logging.getLogger(__name__)


def policy_grid(
    max_steps: list[int],
    malformed_policies: list[str],
    repeated_call_policies: list[str],
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
    cells = [
        LoopPolicyCell(steps, LoopPolicy(malformed, repeated))
        for steps in dict.fromkeys(max_steps)
        for malformed in dict.fromkeys(malformed_policies)
        for repeated in dict.fromkeys(repeated_call_policies)
    ]
    if not any(cell.is_baseline for cell in cells):
        raise SystemExit(
            f"grid must include baseline max_steps={BASELINE_MAX_STEPS}, "
            f"malformed={BASELINE_POLICY.malformed_call}, "
            f"repeated={BASELINE_POLICY.repeated_call}"
        )
    return cells


def _run_cell(
    tasks: list[AgenticTask],
    cell: LoopPolicyCell,
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    budget: ContextBudget,
    meter: ThroughputMeter | None,
) -> LoopPolicyReport:
    task_number = 0

    def harness(
        task: AgenticTask,
        complete: LLMComplete,
        catalog: dict[str, ToolDef],
        *,
        max_steps: int = BASELINE_MAX_STEPS,
        policy: ContextPolicy | None = None,
        budget: ContextBudget | None = None,
    ) -> Episode:
        nonlocal task_number
        task_number += 1
        _LOG.info(
            "[agentic-loop-policy] cell=%s task=%d/%d id=%s",
            cell.cell_id,
            task_number,
            len(tasks),
            task.id,
        )
        episode = loop_harness(
            task,
            complete,
            catalog,
            max_steps=max_steps,
            policy=policy,
            budget=budget,
            loop_policy=cell.policy,
        )
        _LOG.info(
            "[agentic-loop-policy] cell=%s task=%d/%d done success=%s steps=%d calls=%d "
            "malformed=%d repeats=%d noops=%d wall=%.1fs",
            cell.cell_id,
            task_number,
            len(tasks),
            episode.success,
            episode.n_steps,
            episode.n_tool_calls,
            episode.n_malformed_calls,
            episode.n_repeated_calls,
            episode.n_repeated_noops,
            episode.elapsed_s,
        )
        return episode

    _LOG.info("[agentic-loop-policy] cell=%s tasks=%d", cell.cell_id, len(tasks))
    run = run_agentic(
        tasks,
        model=model,
        backend=backend,
        complete=complete,
        max_steps=cell.max_steps,
        harness_name=HARNESS_LOOP,
        harness=harness,
        policy=ContextPolicy(),
        budget=budget,
        persist=False,
        meter=meter,
    )
    return LoopPolicyReport(cell=cell, run=run)


def run_agentic_loop_policy(
    tasks: list[AgenticTask],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    max_steps: list[int],
    malformed_policies: list[str],
    repeated_call_policies: list[str],
    budget: ContextBudget | None = None,
    data_dir: Path | str | None = None,
    run_name: str = METHOD,
    persist: bool = True,
    mirror: Mirror | None = None,
    data_verified: bool = False,
    verification_ref: str | None = None,
    meter: ThroughputMeter | None = None,
    repeat_power_design: dict[str, object] | None = None,
    model_family: str | None = None,
) -> AgenticLoopPolicyRun:
    """Measure every cell on the identical task set and recommend one policy per model."""
    if not tasks:
        raise SystemExit("no agentic tasks provided")
    cells = policy_grid(max_steps, malformed_policies, repeated_call_policies)
    if repeat_power_design is not None:
        from llb.bench.agentic_loop_policy_power import validate_repeat_power_design

        validate_repeat_power_design(
            repeat_power_design,
            tasks,
            cells=cells,
            model_family=model_family,
        )
    resolved_budget = budget if budget is not None else unbounded_budget()
    reports = [
        _run_cell(
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
    if repeat_power_design is not None:
        from llb.bench.agentic_loop_policy_power import analyze_repeat_power

        repeat_power_analysis = analyze_repeat_power(
            repeat_power_design,
            tasks,
            reports,
            model_family=model_family,
        )
    board, board_table = render_board(
        [replace(report.run.result, model=report.cell.cell_id) for report in reports]
    )
    policy_table = format_policy_table(reports)
    table = f"{board_table}\n\n{policy_table}"
    recommendation = build_recommendation(
        model,
        reports,
        repeat_power_analysis=repeat_power_analysis,
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
            budget=resolved_budget,
            verification_config=verified_data_config(
                data_verified=data_verified,
                verification_ref=verification_ref,
            ),
            model_family=model_family,
            repeat_power_design=repeat_power_design,
            repeat_power_analysis=repeat_power_analysis,
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
    )
