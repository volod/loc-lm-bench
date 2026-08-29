"""Run several context-policy arms TASK-ADJACENT, with execution order balanced across the set.

A multi-arm agentic study driven through ONE serving endpoint carries a seam the arms cannot see
from inside: walk every episode of arm A and then every episode of arm B, and "the second arm" is
perfectly confounded with "the arm under test". That is not hypothetical on this host -- an
elision-free control whose two arms build byte-identical prompts still moves the model-input cost,
because the completion an endpoint returns depends on the requests before it, so an episode can end
its walk early in the second arm alone and leave the stratum it belonged to under-powered.

This module removes the confound instead of bounding it. Both arms of ONE task run adjacently, and
which arm goes first rotates with the task index, so position is balanced across the task set
rather than aligned with the treatment. Episodes are still SCORED in task order under their own
arm, so every reading downstream -- per-case pairing, per-stratum counts, completion rates -- reads
exactly what it read before; only the order the endpoint saw them in changed.

Balance is exact when the arm count divides the task count and off by one task otherwise; `offset`
lets a caller carry the rotation across workloads or flip its phase per family so the remainder
cancels over the whole run rather than accumulating on one arm.
"""

import logging
from dataclasses import dataclass, field
from typing import Mapping

from llb.backends.context_budget import ContextBudget, unbounded_budget
from llb.bench.agentic.context_policy import ContextPolicy
from llb.bench.agentic.episode import run_episode
from llb.bench.agentic.model import DEFAULT_MAX_STEPS, AgenticTask, Episode
from llb.bench.context_policy.report import PolicyReport
from llb.bench.context_policy.run import score_policy_episodes
from llb.bench.common import LLMComplete
from llb.bench.tool_world import tool_catalog

_LOG = logging.getLogger(__name__)

# How the arm order is chosen. Named in the persisted run so a reading can state the schedule it
# was measured under instead of inferring it from the row order.
ORDER_ALTERNATING = "alternating_by_task_index"
ORDER_FIXED = "fixed_arm_blocks"


@dataclass(slots=True)
class InterleavedRun:
    """One report per arm, plus the schedule the episodes actually executed in."""

    reports: dict[str, PolicyReport] = field(default_factory=dict)
    schedule: list[dict[str, object]] = field(default_factory=list)


def alternating_arm_schedule(
    arms: tuple[str, ...], n_tasks: int, *, offset: int = 0
) -> list[tuple[str, ...]]:
    """Per task, the order its arms run in -- rotated by task index so no arm holds a position."""
    if not arms:
        raise ValueError("an interleaved schedule needs at least one arm")
    if len(set(arms)) != len(arms):
        raise ValueError(f"arm names must be unique, got {arms!r}")
    return [
        tuple(arms[(position + index + offset) % len(arms)] for position in range(len(arms)))
        for index in range(n_tasks)
    ]


def run_arms_interleaved(
    tasks: list[AgenticTask],
    policies: Mapping[str, ContextPolicy],
    *,
    backend: str,
    complete: LLMComplete,
    max_steps: int = DEFAULT_MAX_STEPS,
    budget: ContextBudget | None = None,
    preserve_memory_markers: bool = True,
    offset: int = 0,
) -> InterleavedRun:
    """Walk every task once per arm, both arms of a task adjacent, first position alternating."""
    arms = tuple(policies)
    budget = budget if budget is not None else unbounded_budget()
    catalog = tool_catalog()
    schedule = alternating_arm_schedule(arms, len(tasks), offset=offset)
    episodes: dict[str, list[Episode]] = {arm: [] for arm in arms}
    log: list[dict[str, object]] = []
    for index, (task, order) in enumerate(zip(tasks, schedule, strict=True)):
        for position, arm in enumerate(order, start=1):
            _LOG.info(
                "[agentic-context] arm=%s position=%d/%d task=%d/%d id=%s",
                arm,
                position,
                len(order),
                index + 1,
                len(tasks),
                task.id,
            )
            episodes[arm].append(
                run_episode(
                    task,
                    complete,
                    catalog=catalog,
                    max_steps=max_steps,
                    policy=policies[arm],
                    budget=budget,
                    preserve_memory_markers=preserve_memory_markers,
                )
            )
            log.append(
                {
                    "task_index": index,
                    "item_id": task.id,
                    "arm": arm,
                    "position": position,
                    "first_arm": order[0],
                }
            )
    return InterleavedRun(
        reports={
            arm: score_policy_episodes(
                tasks, episodes[arm], policy=policies[arm].name, backend=backend
            )
            for arm in arms
        },
        schedule=log,
    )
