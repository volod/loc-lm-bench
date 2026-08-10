"""Mutable episode counters and final immutable episode projection."""

from collections.abc import Callable
from dataclasses import dataclass

from llb.bench.agentic.context import ContextState
from llb.bench.agentic.model import (
    STATUS_COMPLETED,
    STATUS_CONTEXT_OVERFLOW,
    STATUS_INCOMPLETE,
    AgenticTask,
    Episode,
)
from llb.bench.agentic.success import check_success
from llb.bench.tool_world import ToolWorld

Clock = Callable[[], float]


@dataclass(slots=True)
class EpisodeTally:
    """Counters accumulated while driving one episode."""

    started: float
    clock: Clock
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
        self.repeat_feedback_redirected = (
            self.repeat_feedback_redirected or self.awaiting_redirect_key is not None
        )
        self.answer = answer
        self.status = STATUS_COMPLETED

    def overflowed(self, *, before_the_step: bool = False) -> None:
        if before_the_step:
            self.steps -= 1
        self.status = STATUS_CONTEXT_OVERFLOW

    def build(self, task: AgenticTask, world: ToolWorld, state: ContextState) -> Episode:
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
            elapsed_s=self.clock() - self.started,
        )
