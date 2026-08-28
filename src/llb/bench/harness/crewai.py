"""agentic harness comparison CrewAI agentic harness -- the SAME task/tools/candidate driven by a single-agent crew.

CrewAI is an OPT-IN, lazy extra (`[crewai]`): the base install and `make ci` never import it. The
harness wraps the SAME deterministic `ToolWorld` tools as crew tools and the SAME candidate
`complete` as the crew's LLM, runs a single-agent crew over the task, then adapts the crew's result
back into the canonical `Episode` -- so `check_success`, the scorer, and the gated judge are
UNCHANGED and only the framework differs.

CrewAI owns its conversation transcript AND its controller decisions: the harness ACCEPTS
`policy`/`budget`/`loop_policy` on the shared `Harness` protocol so the operator can pass the same
knobs as `loop`/`langgraph`, but it does NOT apply them. Episodes set
`context_policy_supported=False` and `loop_policy_supported=False` and still record the prompt sizes
the framework actually sent, so the harness comparison never labels a CrewAI row as our `full`
policy or as the loop cell that was asked for.

The crew driver is injectable (`crew_runner`): a FAKE crew proves the whole adaptation path with no
dependency / GPU (the same injectable discipline as the rest of category suite/extended workflow), so CI covers the wiring
while the real CrewAI path is exercised only on a host that has the extra installed.
"""

from llb.bench.agentic.context import ContextTelemetry
from llb.bench.agentic.context_policy import ContextPolicy
from llb.bench.agentic.loop_policy import LoopPolicy
from llb.backends.context_budget import ContextBudget
from llb.bench.agentic.model import (
    DEFAULT_MAX_STEPS,
    AgenticTask,
    Episode,
    Harness,
)
from llb.bench.common import LLMComplete
from llb.bench.tool_world import ToolWorld
from llb.core.contracts.benchmarks import ToolDef
from llb.bench.harness.crewai_runtime import CrewRunner, episode_from_outcome, run_real_crew


def make_crewai_harness(crew_runner: CrewRunner | None = None) -> Harness:
    """Build the CrewAI `Harness`. `crew_runner` is injectable (a fake crew in tests); the default
    real runner lazily imports CrewAI and is exercised only on a host with the `[crewai]` extra."""
    runner = crew_runner or run_real_crew

    def harness(
        task: AgenticTask,
        complete: LLMComplete,
        catalog: dict[str, ToolDef],
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        policy: ContextPolicy | None = None,
        budget: ContextBudget | None = None,
        loop_policy: LoopPolicy | None = None,
    ) -> Episode:
        # accepted for protocol parity; CrewAI owns the transcript and the loop decisions
        del policy, budget, loop_policy
        world = ToolWorld.from_setup(task.setup)
        telemetry = ContextTelemetry()
        outcome = runner(task, complete, catalog, world, max_steps, telemetry=telemetry)
        return episode_from_outcome(task, world, outcome, telemetry=telemetry)

    return harness


# CrewAI emits a tracing/telemetry preference panel and (optionally) phones home; disable both so
# benchmark logs stay line-oriented ASCII and the run has no egress (validated on crewai 1.15.0).
