"""Pure controller-to-tool episode loop for the agentic benchmark."""

import time
from collections.abc import Callable

from llb.bench.agentic.context import ContextState
from llb.backends.context_budget import ContextBudget, unbounded_budget
from llb.bench.agentic.context_policy import ContextPolicy
from llb.bench.agentic.controller_channel import ControllerChannel
from llb.bench.agentic.episode_controller import ControllerSeam, ask_controller
from llb.bench.agentic.episode_prompt import step_prompt
from llb.bench.agentic.episode_state import Clock, EpisodeTally
from llb.bench.agentic.loop_policy import REPEATED_NOOP, LoopPolicy, call_key
from llb.bench.agentic.loop_policy import repeated_noop_observation
from llb.bench.agentic.model import DEFAULT_MAX_STEPS, AgenticTask, Episode
from llb.bench.common import LLMChat, LLMComplete
from llb.bench.tool_world import FINISH, ToolWorld, tool_catalog
from llb.core.contracts.benchmarks import ToolDef
from llb.core.contracts.common import ChatMessage
from llb.scoring.tooling.tool_calls import ToolCall


def _record_call(
    state: ContextState,
    world: ToolWorld,
    tally: EpisodeTally,
    *,
    call: ToolCall,
    policy: ContextPolicy,
    loop_policy: LoopPolicy,
    feedback_channel: ControllerChannel | None,
) -> None:
    """Execute and record one tool call, applying repeated-call feedback."""
    current_call_key = call_key(call.name, call.arguments)
    if tally.awaiting_redirect_key is not None and current_call_key != tally.awaiting_redirect_key:
        tally.repeat_feedback_redirected = True
        tally.awaiting_redirect_key = None
    repeated_call = current_call_key == tally.previous_call_key
    repeated_noop = loop_policy.repeated_call == REPEATED_NOOP and repeated_call
    observation = (
        repeated_noop_observation(loop_policy.repeat_feedback)
        if repeated_noop
        else world.execute(call.name, call.arguments)
    )
    tally.n_tool_calls += 1
    tally.n_repeated_calls += 1 if repeated_call else 0
    tally.n_repeated_noops += 1 if repeated_noop else 0
    if repeated_noop:
        tally.awaiting_redirect_key = current_call_key
    if repeated_noop and feedback_channel is not None:
        state.record_channel_feedback(
            policy, call.name, call.arguments, observation, feedback_channel
        )
    else:
        state.record(policy, call.name, call.arguments, observation)
    tally.previous_call_key = current_call_key


def run_episode(
    task: AgenticTask,
    complete: LLMComplete,
    *,
    catalog: dict[str, ToolDef] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    policy: ContextPolicy | None = None,
    budget: ContextBudget | None = None,
    loop_policy: LoopPolicy | None = None,
    chat: LLMChat | None = None,
    feedback_channel: ControllerChannel | None = None,
    feedback_backend: str = "ollama",
    feedback_serialization: dict[str, dict[str, list[dict[str, str]]]] | None = None,
    snapshot: Callable[[list[ChatMessage]], None] | None = None,
    on_refused_prompt: Callable[[str], None] | None = None,
    preserve_memory_markers: bool = True,
    summary_trim_strategy: str = "head_tail",
    clock: Clock = time.monotonic,
) -> Episode:
    """Drive one task to completion or a typed prompt/step limit."""
    world = ToolWorld.from_setup(task.setup)
    catalog = catalog if catalog is not None else tool_catalog()
    policy = policy if policy is not None else ContextPolicy()
    loop_policy = loop_policy if loop_policy is not None else LoopPolicy()
    seam = ControllerSeam(
        complete=complete,
        budget=budget if budget is not None else unbounded_budget(),
        chat=chat,
        feedback_channel=feedback_channel,
        feedback_backend=feedback_backend,
        feedback_serialization=feedback_serialization,
        snapshot=snapshot,
        on_refused_prompt=on_refused_prompt,
    )
    state = ContextState(
        preserve_memory_markers=preserve_memory_markers,
        summary_trim_strategy=summary_trim_strategy,
    )
    tally = EpisodeTally(started=clock(), clock=clock)
    for step in range(1, max_steps + 1):
        tally.steps = step
        prompt = step_prompt(task, catalog, policy, state, seam.budget, complete)
        messages = seam.messages_for(prompt, state)
        if not seam.accepts(prompt, messages, state):
            tally.overflowed(before_the_step=True)
            break
        reply = ask_controller(
            seam,
            state,
            tally,
            prompt=prompt,
            messages=messages,
            catalog=catalog,
            loop_policy=loop_policy,
        )
        if reply.ended:
            break
        if reply.retry:
            continue
        call = reply.parsed.call
        if call is None:
            tally.finish_with(reply.raw.strip())
            break
        if call.name == FINISH:
            tally.finish_with(str(call.arguments.get("answer", "")))
            break
        _record_call(
            state,
            world,
            tally,
            call=call,
            policy=policy,
            loop_policy=loop_policy,
            feedback_channel=feedback_channel,
        )
    return tally.build(task, world, state)


# Compatibility imports for pre-existing batch internals. These are removed when that API is made public.
from llb.bench.agentic.batch import (  # noqa: E402,F401
    _resolve_harness,
    _row,
    _run_episodes,
    _score_episodes,
)
