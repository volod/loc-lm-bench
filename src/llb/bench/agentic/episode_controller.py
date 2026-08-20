"""Controller transport, prompt guard, parsing, and one-shot repair."""

from collections.abc import Callable
from dataclasses import dataclass

from llb.bench.agentic.context import ContextState
from llb.bench.agentic.context_budget import ContextBudget
from llb.bench.agentic.controller_channel import (
    CONTROLLER_CHANNELS,
    ControllerChannel,
    serialize_controller_transcript,
    transcript_chars,
)
from llb.bench.agentic.episode_state import EpisodeTally
from llb.bench.agentic.loop_policy import (
    MALFORMED_ANSWER,
    MALFORMED_REPAIR_ONCE,
    LoopPolicy,
    repair_prompt,
    strict_feedback,
)
from llb.bench.common import LLMChat, LLMComplete
from llb.core.contracts.benchmarks import ToolDef
from llb.core.contracts.common import ChatMessage
from llb.scoring.tooling.tool_calls import ToolCallParse, parse_tool_call_detailed


@dataclass(frozen=True, slots=True)
class ControllerSeam:
    """Model transport and prompt-guard configuration for one episode."""

    complete: LLMComplete
    budget: ContextBudget
    chat: LLMChat | None = None
    feedback_channel: ControllerChannel | None = None
    feedback_backend: str = "ollama"
    feedback_serialization: dict[str, dict[str, list[dict[str, str]]]] | None = None
    snapshot: Callable[[list[ChatMessage]], None] | None = None
    on_refused_prompt: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        if (self.chat is None) != (self.feedback_channel is None):
            raise ValueError("chat and feedback_channel must be configured together")
        if self.feedback_channel is not None and self.feedback_channel not in CONTROLLER_CHANNELS:
            raise ValueError(f"unknown feedback channel: {self.feedback_channel!r}")

    def messages_for(self, prompt: str, state: ContextState) -> list[ChatMessage] | None:
        if self.chat is None:
            return None
        return serialize_controller_transcript(
            prompt,
            state.controller_feedback,
            backend=self.feedback_backend,
            serializer_transforms=self.feedback_serialization,
        )

    def accepts(
        self,
        prompt: str,
        messages: list[ChatMessage] | None,
        state: ContextState,
    ) -> bool:
        prompt_chars = transcript_chars(messages) if messages is not None else len(prompt)
        state.telemetry.prompt_chars.append(prompt_chars)
        if not self.budget.fits(prompt_chars):
            if self.on_refused_prompt is not None:
                self.on_refused_prompt(prompt)
            return False
        state.telemetry.model_input_prompt_chars += prompt_chars
        return True

    def send(self, prompt: str, messages: list[ChatMessage] | None) -> str:
        if messages is None:
            return self.complete(prompt)
        assert self.chat is not None
        if self.snapshot is not None:
            self.snapshot(messages)
        return self.chat(messages)


@dataclass(frozen=True, slots=True)
class ControllerReply:
    """Parsed controller reply plus its effect on the episode loop."""

    parsed: ToolCallParse
    raw: str
    ended: bool = False
    retry: bool = False


def _repair_round(
    seam: ControllerSeam,
    state: ContextState,
    tally: EpisodeTally,
    *,
    prompt: str,
    catalog: dict[str, ToolDef],
    raw: str,
    error: str,
) -> tuple[ToolCallParse, str] | None:
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


def ask_controller(
    seam: ControllerSeam,
    state: ContextState,
    tally: EpisodeTally,
    *,
    prompt: str,
    messages: list[ChatMessage] | None,
    catalog: dict[str, ToolDef],
    loop_policy: LoopPolicy,
) -> ControllerReply:
    """Ask the controller once and apply the malformed-call policy."""
    raw = seam.send(prompt, messages)
    parsed = parse_tool_call_detailed(raw)
    tally.n_controller_calls += 1
    if parsed.call is not None or not parsed.attempted:
        return ControllerReply(parsed=parsed, raw=str(raw))
    tally.n_malformed_calls += 1
    if loop_policy.malformed_call == MALFORMED_ANSWER:
        tally.finish_with(str(raw).strip())
        return ControllerReply(parsed=parsed, raw=str(raw), ended=True)
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
            return ControllerReply(parsed=parsed, raw=str(raw), ended=True)
        parsed, raw = repaired
    if parsed.call is None and parsed.attempted:
        state.record_feedback(strict_feedback(parsed.error or "unreadable tool call"))
        return ControllerReply(parsed=parsed, raw=str(raw), retry=True)
    return ControllerReply(parsed=parsed, raw=str(raw))
