"""The controller->execute->controller cycle: build each step prompt, drive one task to an
`Episode` in the deterministic sandbox, resolve/run harnesses, and score a batch of episodes.

`run_episode` is the pure `loop` harness; the LangGraph and CrewAI harnesses (in `llb.bench.harness`)
reuse `build_agent_prompt` + `check_success` to produce the SAME canonical `Episode`.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass

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
from llb.scoring.tool_calls import ToolCall, ToolCallParse, parse_tool_call_detailed


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


@dataclass(slots=True)
class _EpisodeTally:
    """What one episode accumulates on its way to an `Episode`.

    A builder rather than eleven locals threaded through one long loop: the counters are named once,
    the two places an episode can END agree on what "finished" means (`finish_with`), and `build`
    is the single point that turns the run into the immutable record every harness returns.
    """

    started: float
    answer: str = ""
    status: str = STATUS_INCOMPLETE
    steps: int = 0
    n_tool_calls: int = 0
    n_controller_calls: int = 0
    n_malformed_calls: int = 0
    n_repair_attempts: int = 0
    n_repeated_calls: int = 0
    n_repeated_noops: int = 0
    repeat_feedback_redirected: bool = False
    awaiting_redirect_key: str | None = None
    previous_call_key: str | None = None

    def finish_with(self, answer: str) -> None:
        """The model produced its final answer; a redirect still pending counts as followed."""
        self.repeat_feedback_redirected = (
            self.repeat_feedback_redirected or self.awaiting_redirect_key is not None
        )
        self.answer = answer
        self.status = STATUS_COMPLETED

    def overflowed(self, *, before_the_step: bool = False) -> None:
        """The guard refused a prompt: a TYPED end, not an answer scored from whatever came back."""
        if before_the_step:
            self.steps -= 1
        self.status = STATUS_CONTEXT_OVERFLOW

    def build(self, task: AgenticTask, world: ToolWorld, state: ContextState) -> Episode:
        """The immutable record of this run, scored against the world the calls actually left."""
        return Episode(
            success=check_success(task, world, self.answer),
            status=self.status,
            n_steps=self.steps,
            n_tool_calls=self.n_tool_calls,
            answer=self.answer,
            world=world,
            transcript=state.executed,
            telemetry=state.telemetry,
            context_policy_supported=True,
            n_model_calls=self.n_controller_calls + state.telemetry.n_compactions,
            n_malformed_calls=self.n_malformed_calls,
            n_repair_attempts=self.n_repair_attempts,
            n_repeated_calls=self.n_repeated_calls,
            n_repeated_noops=self.n_repeated_noops,
            repeat_feedback_redirected=self.repeat_feedback_redirected,
            elapsed_s=time.monotonic() - self.started,
        )


@dataclass(frozen=True, slots=True)
class _ControllerSeam:
    """How this episode reaches the model, and what the prompt guard does with what it builds.

    One object because the four transport settings (`chat`, the channel, the backend, the
    serializer transforms) are meaningless apart -- a prompt is either a plain completion or a
    serialized transcript -- and every step needs the same answer to "how do I send this?".
    """

    complete: LLMComplete
    budget: ContextBudget
    chat: LLMChat | None = None
    feedback_channel: ControllerChannel | None = None
    feedback_backend: str = "ollama"
    feedback_serialization: dict[str, dict[str, list[dict[str, str]]]] | None = None
    snapshot: Callable[[list[ChatMessage]], None] | None = None
    on_refused_prompt: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        """Refuse a half-configured channel, which would silently send an unserialized prompt."""
        if (self.chat is None) != (self.feedback_channel is None):
            raise ValueError("chat and feedback_channel must be configured together")
        if self.feedback_channel is not None and self.feedback_channel not in CONTROLLER_CHANNELS:
            raise ValueError(f"unknown feedback channel: {self.feedback_channel!r}")

    def messages_for(self, prompt: str, state: ContextState) -> list[ChatMessage] | None:
        """This prompt as a serialized controller transcript, or None on the completion path."""
        if self.chat is None:
            return None
        return serialize_controller_transcript(
            prompt,
            state.controller_feedback,
            backend=self.feedback_backend,
            serializer_transforms=self.feedback_serialization,
        )

    def accepts(self, prompt: str, messages: list[ChatMessage] | None, state: ContextState) -> bool:
        """Record what this prompt costs and say whether the guard lets it be sent.

        The refusal observer fires HERE, because a refused prompt is the only prompt the loop
        builds that no other seam ever sees.
        """
        prompt_chars = transcript_chars(messages) if messages is not None else len(prompt)
        state.telemetry.prompt_chars.append(prompt_chars)
        if not self.budget.fits(prompt_chars):
            if self.on_refused_prompt is not None:
                self.on_refused_prompt(prompt)
            return False
        state.telemetry.model_input_prompt_chars += prompt_chars
        return True

    def send(self, prompt: str, messages: list[ChatMessage] | None) -> str:
        """Hand the model what the guard accepted, snapshotting the transcript form of it."""
        if messages is None:
            return self.complete(prompt)
        assert self.chat is not None
        if self.snapshot is not None:
            self.snapshot(messages)
        return self.chat(messages)


def _repair_round(
    seam: _ControllerSeam,
    state: ContextState,
    tally: _EpisodeTally,
    *,
    prompt: str,
    catalog: dict[str, ToolDef],
    raw: str,
    error: str,
) -> tuple[ToolCallParse, str] | None:
    """Send ONE repair prompt and return its re-parse and reply, or None when the guard refused."""
    repaired = repair_prompt(prompt, catalog, raw, error)
    messages = seam.messages_for(repaired, state)
    if not seam.accepts(repaired, messages, state):
        tally.overflowed()
        return None
    state.telemetry.n_repair_prompts += 1
    tally.n_repair_attempts += 1
    reply = seam.send(repaired, messages)
    parsed = parse_tool_call_detailed(reply)
    tally.n_controller_calls += 1
    if parsed.call is None and parsed.attempted:
        tally.n_malformed_calls += 1
    return parsed, str(reply)


@dataclass(frozen=True, slots=True)
class _Reply:
    """One controller answer, already read through the loop policy's malformed-call rule.

    Three readings, because that is what the rule can produce: the episode is OVER (an answer, or a
    guard refusal during repair), the controller must be asked AGAIN (feedback was recorded), or
    there is a call to act on.
    """

    parsed: ToolCallParse
    raw: str
    ended: bool = False
    retry: bool = False


def _ask_controller(
    seam: _ControllerSeam,
    state: ContextState,
    tally: _EpisodeTally,
    *,
    prompt: str,
    messages: list[ChatMessage] | None,
    catalog: dict[str, ToolDef],
    loop_policy: LoopPolicy,
) -> _Reply:
    """Ask the controller once and apply the malformed-call rule to whatever came back."""
    raw = seam.send(prompt, messages)
    parsed = parse_tool_call_detailed(raw)
    tally.n_controller_calls += 1
    if parsed.call is not None or not parsed.attempted:
        return _Reply(parsed=parsed, raw=str(raw))
    tally.n_malformed_calls += 1
    if loop_policy.malformed_call == MALFORMED_ANSWER:
        tally.finish_with(str(raw).strip())
        return _Reply(parsed=parsed, raw=str(raw), ended=True)
    if loop_policy.malformed_call == MALFORMED_REPAIR_ONCE:
        repaired = _repair_round(
            seam,
            state,
            tally,
            prompt=prompt,
            catalog=catalog,
            raw=str(raw),
            error=parsed.error or "unreadable tool call",
        )
        if repaired is None:
            return _Reply(parsed=parsed, raw=str(raw), ended=True)
        parsed, raw = repaired
    if parsed.call is None and parsed.attempted:
        state.record_feedback(strict_feedback(parsed.error or "unreadable tool call"))
        return _Reply(parsed=parsed, raw=str(raw), retry=True)
    return _Reply(parsed=parsed, raw=str(raw))


def _record_call(
    state: ContextState,
    world: ToolWorld,
    tally: _EpisodeTally,
    *,
    call: ToolCall,
    policy: ContextPolicy,
    loop_policy: LoopPolicy,
    feedback_channel: ControllerChannel | None,
) -> None:
    """Run one tool call -- or answer a repeated no-op with feedback -- and record what came back."""
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
) -> Episode:
    """Drive one task to completion (or the step budget) in the deterministic sandbox.

    This is the pure `loop` harness: the controller->execute->controller cycle with no agent
    framework. `catalog` is injectable so every harness shares ONE tool catalog; it defaults to
    the canonical `tool_catalog()` (so existing callers are unchanged). `policy` selects the
    context-management policy (default `full`: the whole transcript, today's behavior) and
    `budget` the per-step prompt guard (default unbounded: nothing is refused).

    `on_refused_prompt` observes the prompt the guard REFUSES. It is the only prompt the loop
    builds that no other seam can see: `complete`/`chat` are handed what is SENT, and a refusal by
    definition is not, so a caller comparing two runs byte for byte would otherwise have to treat
    the prompt that ended the episode as if it had never existed. It never fires on a run that
    sends everything it builds, and it cannot change what the loop does.

    The body is the CYCLE and nothing else: what the model is asked, what it answered, and what the
    answer did. Counting lives in `_EpisodeTally`, transport in `_ControllerSeam`, the repair round
    and the tool call in their own steps."""
    world = ToolWorld.from_setup(task.setup)
    catalog = catalog if catalog is not None else tool_catalog()
    policy = policy if policy is not None else ContextPolicy()
    loop_policy = loop_policy if loop_policy is not None else LoopPolicy()
    seam = _ControllerSeam(
        complete=complete,
        budget=budget if budget is not None else unbounded_budget(),
        chat=chat,
        feedback_channel=feedback_channel,
        feedback_backend=feedback_backend,
        feedback_serialization=feedback_serialization,
        snapshot=snapshot,
        on_refused_prompt=on_refused_prompt,
    )
    state = ContextState()
    tally = _EpisodeTally(started=time.monotonic())
    for step in range(1, max_steps + 1):
        tally.steps = step
        prompt = step_prompt(task, catalog, policy, state, seam.budget, complete)
        messages = seam.messages_for(prompt, state)
        if not seam.accepts(prompt, messages, state):
            tally.overflowed(before_the_step=True)
            break
        reply = _ask_controller(
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
        if call is None:  # the model answered in prose -> treat as the final answer
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


# Compatibility imports for callers that used the pre-split episode module.
from llb.bench.agentic.batch import (  # noqa: E402,F401
    _resolve_harness,
    _row,
    _run_episodes,
    _score_episodes,
)
