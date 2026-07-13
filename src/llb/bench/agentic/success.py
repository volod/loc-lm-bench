"""Objective task success: evaluate the planted assertions over the final env-state / answer.

This is the OBJECTIVE completion signal (the headline under TIER_AGENTIC) -- a task succeeds only
when every planted assertion holds -- kept separate from the opt-in trajectory-quality judge.
"""

from typing import Any

from llb.bench.agentic.model import (
    ASSERT_ANSWER_CONTAINS,
    ASSERT_DB_EQUALS,
    ASSERT_FILE_CONTAINS,
    ASSERT_FILE_EQUALS,
    AgenticTask,
)
from llb.bench.tool_world import ToolWorld


def _norm(value: Any) -> str:
    return str(value).strip().casefold()


def check_assertion(assertion: dict[str, Any], world: ToolWorld, answer: str) -> bool:
    """Evaluate one success assertion against the final env-state / answer."""
    kind = assertion.get("kind")
    if kind == ASSERT_FILE_EQUALS:
        return _norm(world.files.get(str(assertion.get("path", "")), "")) == _norm(
            assertion.get("value", "")
        )
    if kind == ASSERT_FILE_CONTAINS:
        return _norm(assertion.get("value", "")) in _norm(
            world.files.get(str(assertion.get("path", "")), "")
        )
    if kind == ASSERT_DB_EQUALS:
        return _norm(world.db.get(str(assertion.get("key", "")), "")) == _norm(
            assertion.get("value", "")
        )
    if kind == ASSERT_ANSWER_CONTAINS:
        return _norm(assertion.get("value", "")) in _norm(answer)
    return False


def check_success(task: AgenticTask, world: ToolWorld, answer: str) -> bool:
    """A task succeeds when EVERY planted assertion holds (an empty assertion list never passes)."""
    return bool(task.success) and all(check_assertion(a, world, answer) for a in task.success)
