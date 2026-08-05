"""The controller->execute->controller cycle: build each step prompt, drive one task to an
`Episode` in the deterministic sandbox, resolve/run harnesses, and score a batch of episodes.

`run_episode` is the pure `loop` harness; the LangGraph and CrewAI harnesses (in `llb.bench.harness`)
reuse `build_agent_prompt` + `check_success` to produce the SAME canonical `Episode`.
"""

import json
import time
from collections.abc import Callable

from llb.bench.agentic.context import (
    POLICY_COMPACT,
    SUMMARY_INPUT_CAP_TRIGGER,
    ContextPolicy,
    ContextState,
    TranscriptEntry,
    compact_state,
    format_entry,
    policy_history_lines,
    summarize_entries,
    summary_prompt_overhead_chars,
)
from llb.bench.agentic.context_budget import ContextBudget, unbounded_budget
from llb.bench.agentic.controller_channel import (
    CONTROLLER_CHANNELS,
    ControllerChannel,
    serialize_controller_transcript,
    transcript_chars,
)
from llb.bench.agentic.loop_policy import (
    MALFORMED_ANSWER,
    MALFORMED_REPAIR_ONCE,
    REPEATED_NOOP,
    LoopPolicy,
    call_key,
    repair_prompt,
    repeated_noop_observation,
    strict_feedback,
)
from llb.bench.agentic.model import (
    DEFAULT_MAX_STEPS,
    STATUS_COMPLETED,
    STATUS_CONTEXT_OVERFLOW,
    STATUS_INCOMPLETE,
    AgenticTask,
    Episode,
)
from llb.bench.agentic.success import check_success
from llb.bench.common import LLMChat, LLMComplete
from llb.core.contracts.common import ChatMessage
from llb.bench.tool_world import FINISH, ToolWorld, tool_catalog
from llb.core.contracts.benchmarks import ToolDef
from llb.prompts.registry import render_text
from llb.scoring.tool_calls import parse_tool_call_detailed


def build_agent_prompt(
    task: AgenticTask,
    catalog: dict[str, ToolDef],
    transcript: list[TranscriptEntry],
) -> str:
    """The next-step prompt: available tools, the task, and the running observation transcript."""
    return build_agent_prompt_lines(task, catalog, [format_entry(entry) for entry in transcript])


def build_agent_prompt_lines(
    task: AgenticTask,
    catalog: dict[str, ToolDef],
    history_lines: list[str],
) -> str:
    """The next-step prompt from ALREADY-RENDERED history lines.

    The policy seam: a context policy decides which lines (and which markers) the step sees, and
    this assembles the identical prompt scaffold around them. `full` passes every entry through
    verbatim, so its prompt is byte-identical to the pre-policy loop's.
    """
    tools_json = json.dumps(list(catalog.values()), ensure_ascii=False, indent=2)
    history_block = (
        render_text("bench.agentic.history_block", {"history": "\n".join(history_lines)})
        if history_lines
        else ""
    )
    return render_text(
        "bench.agentic.agent_step",
        {
            "tools_json": tools_json,
            "task_prompt": task.prompt,
            "history_block": history_block,
        },
    )


def summary_input_cap_chars(policy: ContextPolicy, budget: ContextBudget) -> int:
    """The char cap the summarize call's INPUT is trimmed to under this policy.

    `window` is a property of the RESOLVED BUDGET alone, so every guard that folds the same step
    summarizes the identical transcript and the compact cost is a pure step function of the fold
    step. `trigger` reads `compact_share * guard`, which moves continuously inside one fold step.
    """
    if policy.summary_input_cap == SUMMARY_INPUT_CAP_TRIGGER:
        return budget.compaction_trigger_chars(policy.compact_share)
    return budget.summary_input_cap_chars(summary_prompt_overhead_chars())


def step_prompt(
    task: AgenticTask,
    catalog: dict[str, ToolDef],
    policy: ContextPolicy,
    state: ContextState,
    budget: ContextBudget,
    complete: LLMComplete,
) -> str:
    """This step's prompt under `policy`, compacting first when the prompt crosses the trigger.

    At most ONE compaction per step: if the compacted prompt still does not fit, the guard --
    not another round of summarizing -- is what ends the episode. The summarize call's input is
    capped too, because it is the very transcript that just blew the step prompt and an over-long
    summarize call would come back silently truncated; `summary_input_cap` picks WHICH bound.
    """
    prompt = build_agent_prompt_lines(task, catalog, policy_history_lines(policy, state))
    if policy.name != POLICY_COMPACT:
        return prompt
    # `compact_share` is the initial trigger. Once a running summary exists, let live work grow
    # to the full guard before folding again. This hysteresis avoids paying a summary call after
    # nearly every later tool call while still compacting before an oversized prompt is sent.
    trigger_share = 1.0 if state.summary else policy.compact_share
    trigger = budget.compaction_trigger_chars(trigger_share)
    if trigger <= 0 or len(prompt) <= trigger:
        return prompt
    summarize = lambda older: summarize_entries(  # noqa: E731
        complete,
        older,
        summary_input_cap_chars(policy, budget),
        prior_summary=state.summary,
        telemetry=state.telemetry,
    )
    if not compact_state(policy, state, summarize):
        return prompt
    return build_agent_prompt_lines(task, catalog, policy_history_lines(policy, state))


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
) -> Episode:
    """Drive one task to completion (or the step budget) in the deterministic sandbox.

    This is the pure `loop` harness: the controller->execute->controller cycle with no agent
    framework. `catalog` is injectable so every harness shares ONE tool catalog; it defaults to
    the canonical `tool_catalog()` (so existing callers are unchanged). `policy` selects the
    context-management policy (default `full`: the whole transcript, today's behavior) and
    `budget` the per-step prompt guard (default unbounded: nothing is refused)."""
    world = ToolWorld.from_setup(task.setup)
    catalog = catalog if catalog is not None else tool_catalog()
    policy = policy if policy is not None else ContextPolicy()
    budget = budget if budget is not None else unbounded_budget()
    loop_policy = loop_policy if loop_policy is not None else LoopPolicy()
    if (chat is None) != (feedback_channel is None):
        raise ValueError("chat and feedback_channel must be configured together")
    if feedback_channel is not None and feedback_channel not in CONTROLLER_CHANNELS:
        raise ValueError(f"unknown feedback channel: {feedback_channel!r}")
    state = ContextState()
    answer = ""
    status = STATUS_INCOMPLETE
    n_tool_calls = 0
    n_controller_calls = 0
    n_malformed_calls = 0
    n_repair_attempts = 0
    n_repeated_calls = 0
    n_repeated_noops = 0
    repeat_feedback_redirected = False
    awaiting_redirect_key: str | None = None
    previous_call_key: str | None = None
    steps = 0
    started = time.monotonic()
    for steps in range(1, max_steps + 1):
        prompt = step_prompt(task, catalog, policy, state, budget, complete)
        messages = (
            serialize_controller_transcript(
                prompt,
                state.controller_feedback,
                backend=feedback_backend,
                serializer_transforms=feedback_serialization,
            )
            if chat is not None
            else None
        )
        prompt_chars = transcript_chars(messages) if messages is not None else len(prompt)
        state.telemetry.prompt_chars.append(prompt_chars)
        if not budget.fits(prompt_chars):
            # The prompt cannot fit the resolved window: end as a TYPED overflow rather than
            # sending it and scoring whatever comes back as the model's answer.
            steps -= 1
            status = STATUS_CONTEXT_OVERFLOW
            break
        state.telemetry.model_input_prompt_chars += prompt_chars
        if messages is not None:
            assert chat is not None
            if snapshot is not None:
                snapshot(messages)
            raw = chat(messages)
        else:
            raw = complete(prompt)
        n_controller_calls += 1
        parsed = parse_tool_call_detailed(raw)
        call = parsed.call
        if call is None and parsed.attempted:
            n_malformed_calls += 1
            if loop_policy.malformed_call == MALFORMED_ANSWER:
                repeat_feedback_redirected = (
                    repeat_feedback_redirected or awaiting_redirect_key is not None
                )
                answer = str(raw).strip()
                status = STATUS_COMPLETED
                break
            if loop_policy.malformed_call == MALFORMED_REPAIR_ONCE:
                repaired = repair_prompt(
                    prompt,
                    catalog,
                    str(raw),
                    parsed.error or "unreadable tool call",
                )
                repaired_messages: list[ChatMessage] | None = (
                    serialize_controller_transcript(
                        repaired,
                        state.controller_feedback,
                        backend=feedback_backend,
                        serializer_transforms=feedback_serialization,
                    )
                    if chat is not None
                    else None
                )
                repaired_chars = (
                    transcript_chars(repaired_messages)
                    if repaired_messages is not None
                    else len(repaired)
                )
                state.telemetry.prompt_chars.append(repaired_chars)
                if not budget.fits(repaired_chars):
                    status = STATUS_CONTEXT_OVERFLOW
                    break
                state.telemetry.model_input_prompt_chars += repaired_chars
                state.telemetry.n_repair_prompts += 1
                n_repair_attempts += 1
                if repaired_messages is not None:
                    assert chat is not None
                    if snapshot is not None:
                        snapshot(repaired_messages)
                    raw = chat(repaired_messages)
                else:
                    raw = complete(repaired)
                n_controller_calls += 1
                parsed = parse_tool_call_detailed(raw)
                call = parsed.call
                if call is None and parsed.attempted:
                    n_malformed_calls += 1
            if call is None and parsed.attempted:
                state.record_feedback(strict_feedback(parsed.error or "unreadable tool call"))
                continue
        if call is None:  # the model answered in prose -> treat as the final answer
            repeat_feedback_redirected = (
                repeat_feedback_redirected or awaiting_redirect_key is not None
            )
            answer = str(raw).strip()
            status = STATUS_COMPLETED
            break
        if call.name == FINISH:
            repeat_feedback_redirected = (
                repeat_feedback_redirected or awaiting_redirect_key is not None
            )
            answer = str(call.arguments.get("answer", ""))
            status = STATUS_COMPLETED
            break
        current_call_key = call_key(call.name, call.arguments)
        if awaiting_redirect_key is not None and current_call_key != awaiting_redirect_key:
            repeat_feedback_redirected = True
            awaiting_redirect_key = None
        repeated_call = current_call_key == previous_call_key
        repeated_noop = loop_policy.repeated_call == REPEATED_NOOP and repeated_call
        observation = (
            repeated_noop_observation(loop_policy.repeat_feedback)
            if repeated_noop
            else world.execute(call.name, call.arguments)
        )
        n_tool_calls += 1
        n_repeated_calls += 1 if repeated_call else 0
        n_repeated_noops += 1 if repeated_noop else 0
        if repeated_noop:
            awaiting_redirect_key = current_call_key
        if repeated_noop and feedback_channel is not None:
            state.record_channel_feedback(
                policy,
                call.name,
                call.arguments,
                observation,
                feedback_channel,
            )
        else:
            state.record(policy, call.name, call.arguments, observation)
        previous_call_key = current_call_key
    success = check_success(task, world, answer)
    return Episode(
        success=success,
        status=status,
        n_steps=steps,
        n_tool_calls=n_tool_calls,
        answer=answer,
        world=world,
        transcript=state.executed,
        telemetry=state.telemetry,
        context_policy_supported=True,
        n_model_calls=n_controller_calls + state.telemetry.n_compactions,
        n_malformed_calls=n_malformed_calls,
        n_repair_attempts=n_repair_attempts,
        n_repeated_calls=n_repeated_calls,
        n_repeated_noops=n_repeated_noops,
        repeat_feedback_redirected=repeat_feedback_redirected,
        elapsed_s=time.monotonic() - started,
    )


# Compatibility imports for callers that used the pre-split episode module.
from llb.bench.agentic.batch import (  # noqa: E402,F401
    _resolve_harness,
    _row,
    _run_episodes,
    _score_episodes,
)
