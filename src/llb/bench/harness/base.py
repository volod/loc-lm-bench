"""agentic harness comparison harness seam -- the pure `loop` harness.

The loop harness is the framework-free controller->execute->controller cycle (`run_episode`)
presented as a named `Harness`, so "loop" sits on the same comparison axis as the LangGraph app
and the CrewAI crew. It adds NO behavior: it just forwards to `run_episode` with the shared tool
catalog, so the loop's results are unchanged by the agentic harness comparison refactor.
"""

import time

from llb.bench.agentic.context_policy import ContextPolicy
from llb.bench.agentic.context_budget import ContextBudget
from llb.bench.agentic.episode import run_episode
from llb.bench.agentic.episode_state import Clock
from llb.bench.agentic.loop_policy import LoopPolicy
from llb.bench.agentic.model import DEFAULT_MAX_STEPS, AgenticTask, Episode
from llb.bench.common import LLMComplete
from llb.core.contracts.benchmarks import ToolDef


def loop_harness(
    task: AgenticTask,
    complete: LLMComplete,
    catalog: dict[str, ToolDef],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    policy: ContextPolicy | None = None,
    budget: ContextBudget | None = None,
    loop_policy: LoopPolicy | None = None,
    clock: Clock = time.monotonic,
) -> Episode:
    """The pure loop as a `Harness`: drive one task through `run_episode` over `catalog`."""
    return run_episode(
        task,
        complete,
        catalog=catalog,
        max_steps=max_steps,
        policy=policy,
        budget=budget,
        loop_policy=loop_policy,
        clock=clock,
    )
