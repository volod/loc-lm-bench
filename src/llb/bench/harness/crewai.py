"""agentic harness comparison CrewAI agentic harness -- the SAME task/tools/candidate driven by a single-agent crew.

CrewAI is an OPT-IN, lazy extra (`[crewai]`): the base install and `make ci` never import it. The
harness wraps the SAME deterministic `ToolWorld` tools as crew tools and the SAME candidate
`complete` as the crew's LLM, runs a single-agent crew over the task, then adapts the crew's result
back into the canonical `Episode` -- so `check_success`, the scorer, and the gated judge are
UNCHANGED and only the framework differs.

The crew driver is injectable (`crew_runner`): a FAKE crew proves the whole adaptation path with no
dependency / GPU (the same injectable discipline as the rest of category suite/extended workflow), so CI covers the wiring
while the real CrewAI path is exercised only on a host that has the extra installed.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from llb.bench.agentic import (
    DEFAULT_MAX_STEPS,
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    AgenticTask,
    Episode,
    Harness,
    build_agent_prompt,
    check_success,
)
from llb.bench.common import LLMComplete
from llb.bench.tool_world import FINISH, ToolWorld
from llb.contracts import ToolDef
from llb.prompts import render_text_map

_LOG = logging.getLogger(__name__)

Transcript = list[tuple[str, dict[str, Any], str]]
ToolExecutor = Callable[[str, dict[str, Any]], str]


@dataclass(slots=True)
class CrewOutcome:
    """The canonical, framework-agnostic result a crew driver returns for one task."""

    answer: str
    transcript: Transcript = field(default_factory=list)
    n_steps: int = 0  # crew LLM calls (efficiency, comparable to the loop's n_steps)
    n_tool_calls: int = 0  # sandbox tools the crew executed
    finished: bool = True  # the crew produced a final answer (vs aborted / budget exhausted)


class CrewRunner(Protocol):
    """Drive a single-agent crew over one task, returning the canonical `CrewOutcome`."""

    def __call__(
        self,
        task: AgenticTask,
        complete: LLMComplete,
        catalog: dict[str, ToolDef],
        world: ToolWorld,
        max_steps: int,
    ) -> CrewOutcome: ...


@dataclass(slots=True)
class CrewToolSpec:
    """A crew-facing tool: name + UA description + parameter names + a bound executor."""

    name: str
    description: str
    params: list[str]  # the ToolDef property names -> the crew tool's args schema
    execute: ToolExecutor


def make_recording_executor(world: ToolWorld, transcript: Transcript) -> ToolExecutor:
    """A `(name, args) -> observation` callable that executes against `world` AND records the call.

    The recorded transcript is what the canonical `Episode` carries, so the trajectory is faithful
    regardless of HOW the crew framework chooses to invoke the tool."""

    def execute(name: str, args: dict[str, Any]) -> str:
        observation = world.execute(name, dict(args))
        transcript.append((name, dict(args), observation))
        return observation

    return execute


def crew_tool_specs(catalog: dict[str, ToolDef], executor: ToolExecutor) -> list[CrewToolSpec]:
    """Map the sandbox tool catalog (minus `finish`) into crew tool specs over one executor.

    The ToolDef's JSON-schema `properties` become the crew tool's parameter names (used to build a
    pydantic args schema CrewAI validates the agent's Action Input against). `finish` is excluded:
    ending the episode is the crew's final answer, not an executed tool -- as the loop treats it."""
    specs: list[CrewToolSpec] = []
    for name, tool in catalog.items():
        if name == FINISH:
            continue
        properties = tool.get("parameters", {}).get("properties", {})
        specs.append(
            CrewToolSpec(
                name=name,
                description=tool["description"],
                params=list(properties),
                execute=executor,
            )
        )
    return specs


def episode_from_outcome(task: AgenticTask, world: ToolWorld, outcome: CrewOutcome) -> Episode:
    """Adapt a `CrewOutcome` into the canonical `Episode` (success re-checked objectively)."""
    return Episode(
        success=check_success(task, world, outcome.answer),
        status=STATUS_COMPLETED if outcome.finished else STATUS_INCOMPLETE,
        n_steps=outcome.n_steps,
        n_tool_calls=outcome.n_tool_calls,
        answer=outcome.answer,
        world=world,
        transcript=list(outcome.transcript),
    )


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
    ) -> Episode:
        world = ToolWorld.from_setup(task.setup)
        outcome = runner(task, complete, catalog, world, max_steps)
        return episode_from_outcome(task, world, outcome)

    return harness


# CrewAI emits a tracing/telemetry preference panel and (optionally) phones home; disable both so
# benchmark logs stay line-oriented ASCII and the run has no egress (validated on crewai 1.15.0).
_CREWAI_QUIET_ENV = {
    "CREWAI_TRACING_ENABLED": "false",
    "CREWAI_DISABLE_TELEMETRY": "true",
    "OTEL_SDK_DISABLED": "true",
}


def run_real_crew(
    task: AgenticTask,
    complete: LLMComplete,
    catalog: dict[str, ToolDef],
    world: ToolWorld,
    max_steps: int,
) -> CrewOutcome:
    """Drive a real single-agent CrewAI crew (lazy import; needs the `[crewai]` extra; 1.15.x).

    The crew's LLM is the candidate `complete` (wrapped as a `BaseLLM` subclass); the crew's tools
    execute against the SAME `world` through a recording executor, so the returned `CrewOutcome` is
    faithful no matter how CrewAI orchestrates the ReAct turns. This path is host-only and is not run
    in CI -- the fake-crew tests cover the adaptation; see `docs/guides/benchmarking/crewai-harness.md` for the
    validation how-to and the actor/model/document extension guide.
    """
    try:
        from crewai import Agent, Crew, Task
    except ImportError as exc:
        raise SystemExit(
            'ERROR: the crewai harness needs the [crewai] extra. Run: uv pip install -e ".[crewai]"'
        ) from exc

    for key, value in _CREWAI_QUIET_ENV.items():
        os.environ.setdefault(key, value)

    transcript: Transcript = []
    executor = make_recording_executor(world, transcript)
    specs = crew_tool_specs(catalog, executor)
    llm, calls = _make_candidate_llm(complete)
    tools = [_build_crew_tool(spec) for spec in specs]
    agent_prompt = render_text_map("bench.harness.crewai.agent")
    agent = Agent(
        role=agent_prompt["role"],
        goal=agent_prompt["goal"],
        backstory=agent_prompt["backstory"],
        tools=tools,
        llm=llm,
        max_iter=max_steps,
        verbose=False,
    )
    crew_task = Task(
        description=build_agent_prompt(task, catalog, transcript),
        expected_output=agent_prompt["expected_output"],
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[crew_task], verbose=False)
    result = crew.kickoff()
    answer = str(getattr(result, "raw", result) or "").strip()
    return CrewOutcome(
        answer=answer,
        transcript=transcript,
        n_steps=max(calls["n"], len(transcript)),
        n_tool_calls=len(transcript),
        finished=bool(answer),
    )


def _make_candidate_llm(complete: LLMComplete) -> tuple[Any, dict[str, int]]:
    """Wrap the candidate `complete` (prompt -> text) as a CrewAI `BaseLLM` subclass.

    `BaseLLM` is a pydantic ABC (abstract `call`), so the subclass is built lazily inside the extra;
    the call-count rides in a mutable closure dict (avoiding pydantic attribute fights). CrewAI passes
    the ReAct conversation as a message list, flattened here to the single prompt string the candidate
    backend understands. Returns (llm, calls) so the runner can read `calls["n"]` for `n_steps`."""
    from crewai.llms.base_llm import BaseLLM

    calls = {"n": 0}

    class _CandidateLLM(BaseLLM):
        def call(self, messages: Any, *_args: Any, **_kwargs: Any) -> str:
            calls["n"] += 1
            if isinstance(messages, str):
                prompt = messages
            else:
                prompt = "\n\n".join(
                    str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in messages
                )
            return complete(prompt)

        def supports_function_calling(self) -> bool:
            return False

        def supports_stop_words(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            return 8192

    return _CandidateLLM(model="loc-lm-bench-candidate"), calls


def _build_crew_tool(spec: CrewToolSpec) -> Any:
    """Wrap one `CrewToolSpec` as a CrewAI `BaseTool` whose args schema mirrors the ToolDef params.

    The agent's parsed Action Input (a dict) is validated against a pydantic schema built from the
    tool's parameter names (all optional strings, so a partial call reaches the world rather than
    failing validation), then forwarded to the recording executor as `(name, kwargs)`."""
    from typing import Optional

    from crewai.tools import BaseTool
    from pydantic import create_model

    fields: dict[str, Any] = {param: (Optional[str], None) for param in spec.params}
    args_model = create_model(f"{spec.name}_args", **fields)

    class _SandboxTool(BaseTool):
        name: str = spec.name
        description: str = spec.description
        args_schema: type = args_model

        def _run(self, **kwargs: Any) -> str:
            return spec.execute(spec.name, {k: v for k, v in kwargs.items() if v is not None})

    return _SandboxTool()
