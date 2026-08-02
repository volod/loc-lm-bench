"""Deterministic worst-case prompt probe for cap-fitting boundary geometry.

A cap-fitting cell is only usable when the prompt guard sits in a narrow band: ABOVE the largest
prompt `observation_cap` builds under perfect play (otherwise cap overflows and the cell measures
overflow rescue, not cost), and BELOW that peak divided by the compact trigger share (otherwise
compact never fires and the cell is inactive). Both bounds are computable with NO model: the
memory-dependent tool world is deterministic, so an oracle controller that always plays the next
workflow token reproduces the exact prompt sequence a perfect controller would send.
"""

import json
import re

from llb.bench.agentic.context import (
    DEFAULT_OBSERVATION_CAP_CHARS,
    OBSERVATION_HEAD_SHARE,
    POLICY_OBSERVATION_CAP,
    ContextPolicy,
)
from llb.bench.agentic.context_budget import unbounded_budget
from llb.bench.agentic.episode import run_episode
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_memory_transcript import (
    DEFAULT_MEMORY_PAD_CHARS,
    build_memory_dependent_tasks,
)
from llb.bench.tool_world import ADVANCE, FINISH, OBS_WORKFLOW_COMPLETE

_NEXT_TOKEN = re.compile(r"\[next token: ([^\]]+)\]")
_START_TOKEN = re.compile(r'токеном "([^"]+)"')
_MEMORY_FACT = re.compile(r"\[memory: final_code=([^\]]+)\]")
_WORKFLOW_DONE = OBS_WORKFLOW_COMPLETE.split("]", 1)[0] + "]"


def oracle_controller(prompt: str) -> str:
    """Play the token chain perfectly: advance on the freshest token, then finish with the code."""
    if _WORKFLOW_DONE in prompt:
        memory = _MEMORY_FACT.findall(prompt)
        answer = memory[-1] if memory else ""
        return json.dumps({"name": FINISH, "arguments": {"answer": answer}}, ensure_ascii=False)
    tokens = _NEXT_TOKEN.findall(prompt) or _START_TOKEN.findall(prompt)
    token = tokens[-1] if tokens else ""
    return json.dumps({"name": ADVANCE, "arguments": {"token": token}}, ensure_ascii=False)


def cap_peak_prompt_chars(
    *,
    depth: int,
    n_tasks: int,
    pad_chars: int = DEFAULT_MEMORY_PAD_CHARS,
    max_steps_margin: int = 4,
    observation_cap_chars: int = DEFAULT_OBSERVATION_CAP_CHARS,
    observation_head_share: float = OBSERVATION_HEAD_SHARE,
) -> int:
    """The largest `observation_cap` step prompt this geometry produces under perfect play."""
    policy = ContextPolicy(
        name=POLICY_OBSERVATION_CAP,
        observation_cap_chars=observation_cap_chars,
        observation_head_share=observation_head_share,
    )
    peaks = [
        run_episode(
            AgenticTask.from_record(record),
            oracle_controller,
            max_steps=depth + max_steps_margin,
            policy=policy,
            budget=unbounded_budget(),
        ).telemetry.max_prompt_chars
        for record in build_memory_dependent_tasks(
            n_tasks=n_tasks, depth=depth, pad_chars=pad_chars
        )
    ]
    return max(peaks)


def usable_guard_band(peak_prompt_chars: int, compact_share: float) -> tuple[int, int]:
    """The open prompt-guard interval where cap fits AND compact still crosses its trigger.

    Below the lower bound cap overflows; at or above the upper bound the compact trigger
    (`compact_share * guard`) never reaches the peak prompt, so compaction never activates.
    """
    if peak_prompt_chars <= 0:
        raise ValueError("peak prompt chars must be positive")
    if not 0.0 < compact_share <= 1.0:
        raise ValueError(f"compact share must be in (0, 1], got {compact_share}")
    return peak_prompt_chars, int(peak_prompt_chars / compact_share)


def guard_is_cap_fitting(guard_chars: int, peak_prompt_chars: int, compact_share: float) -> bool:
    """Whether one predeclared guard lies strictly inside the usable band."""
    low, high = usable_guard_band(peak_prompt_chars, compact_share)
    return low < guard_chars < high
