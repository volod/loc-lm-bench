"""Execute one fixed cell of the agent-loop policy grid."""

import logging
import time

from llb.bench.agentic.context_policy import ContextPolicy
from llb.bench.agentic.context_budget import ContextBudget
from llb.bench.agentic.episode_state import Clock
from llb.bench.agentic.model import HARNESS_LOOP, AgenticTask, Episode
from llb.bench.agentic.run import run_agentic
from llb.bench.loop_policy.report import (
    BASELINE_MAX_STEPS,
    LoopPolicyCell,
    LoopPolicyReport,
)
from llb.bench.common import LLMComplete
from llb.bench.common_backend import ThroughputMeter
from llb.bench.harness.base import loop_harness
from llb.core.contracts.benchmarks import ToolDef

_LOG = logging.getLogger(__name__)


def run_policy_cell(
    tasks: list[AgenticTask],
    cell: LoopPolicyCell,
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    budget: ContextBudget,
    meter: ThroughputMeter | None,
    clock: Clock = time.monotonic,
) -> LoopPolicyReport:
    """Run one fresh episode per task for a single immutable policy cell."""
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
            clock=clock,
        )
        _LOG.info(
            "[agentic-loop-policy] cell=%s task=%d/%d done success=%s steps=%d calls=%d "
            "malformed=%d repeats=%d noops=%d redirected=%s wall=%.1fs",
            cell.cell_id,
            task_number,
            len(tasks),
            episode.success,
            episode.n_steps,
            episode.n_tool_calls,
            episode.n_malformed_calls,
            episode.n_repeated_calls,
            episode.n_repeated_noops,
            episode.repeat_feedback_redirected,
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


__all__ = ["run_policy_cell"]
