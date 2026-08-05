"""Deterministic worst-case prompt probe for cap-fitting boundary geometry.

A cap-fitting cell is only usable when the prompt guard sits in a narrow band: ABOVE the largest
prompt `observation_cap` builds under perfect play (otherwise cap overflows and the cell measures
overflow rescue, not cost), and BELOW that peak divided by the compact trigger share (otherwise
compact never fires and the cell is inactive). Both bounds are computable with NO model: the
memory-dependent tool world is deterministic, so an oracle controller that always plays the next
workflow token reproduces the exact prompt sequence a perfect controller would send.
"""

import json
import math
import re

from llb.bench.agentic.context import (
    DEFAULT_OBSERVATION_CAP_CHARS,
    DEFAULT_SUMMARY_INPUT_CAP,
    OBSERVATION_HEAD_SHARE,
    POLICY_COMPACT,
    POLICY_OBSERVATION_CAP,
    ContextPolicy,
    is_summary_prompt,
)
from llb.bench.agentic.context_budget import fixed_budget, unbounded_budget
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
# The constant a probe answers the summarize call with, so a compact walk has no model in it.
ORACLE_SUMMARY = "[стислий підсумок попередніх кроків]"


def oracle_controller(prompt: str) -> str:
    """Play the token chain perfectly: advance on the freshest token, then finish with the code."""
    if _WORKFLOW_DONE in prompt:
        memory = _MEMORY_FACT.findall(prompt)
        answer = memory[-1] if memory else ""
        return json.dumps({"name": FINISH, "arguments": {"answer": answer}}, ensure_ascii=False)
    tokens = _NEXT_TOKEN.findall(prompt) or _START_TOKEN.findall(prompt)
    token = tokens[-1] if tokens else ""
    return json.dumps({"name": ADVANCE, "arguments": {"token": token}}, ensure_ascii=False)


def oracle_compacting_controller(prompt: str) -> str:
    """The oracle plus a FIXED reply to the summarize call, so a compact walk stays deterministic.

    A compact episode makes two kinds of model call, and only the controller one has an oracle. Any
    constant summary would do; a short one keeps the summary marker from dominating later prompts,
    which is what the probe measures around.
    """
    return ORACLE_SUMMARY if is_summary_prompt(prompt) else oracle_controller(prompt)


def compact_fold_input_probe(
    *,
    depth: int,
    n_tasks: int,
    max_prompt_chars: int,
    compact_share: float,
    summary_input_cap: str = DEFAULT_SUMMARY_INPUT_CAP,
    pad_chars: int = DEFAULT_MEMORY_PAD_CHARS,
    max_steps_margin: int = 4,
    observation_cap_chars: int = DEFAULT_OBSERVATION_CAP_CHARS,
    observation_head_share: float = OBSERVATION_HEAD_SHARE,
) -> dict[str, int]:
    """What ONE guard offers the summarizer under perfect play, and how much its cap elides.

    The elided span is the transcript the running summary is never shown, and it is decided with no
    model at all: the tool world is deterministic, so an oracle controller folding at the same step
    a real controller folds at offers the summarizer the same bytes. This is what lets a design
    predeclare that its reference arm actually HAS an elision to price, and that the step-aligned
    arm has none.
    """
    policy = ContextPolicy(
        name=POLICY_COMPACT,
        observation_cap_chars=observation_cap_chars,
        observation_head_share=observation_head_share,
        compact_share=compact_share,
        summary_input_cap=summary_input_cap,
    )
    telemetries = [
        run_episode(
            AgenticTask.from_record(record),
            oracle_compacting_controller,
            max_steps=depth + max_steps_margin,
            policy=policy,
            budget=fixed_budget(max_prompt_chars),
        ).telemetry
        for record in build_memory_dependent_tasks(
            n_tasks=n_tasks, depth=depth, pad_chars=pad_chars
        )
    ]
    return {
        "n_compactions": max(item.n_compactions for item in telemetries),
        "summary_input_chars": max(item.summary_input_chars for item in telemetries),
        "summary_input_elided_chars": max(item.summary_input_elided_chars for item in telemetries),
        "n_trimmed_summary_inputs": max(item.n_trimmed_summary_inputs for item in telemetries),
    }


def cap_prompt_sequence(
    *,
    depth: int,
    n_tasks: int,
    pad_chars: int = DEFAULT_MEMORY_PAD_CHARS,
    max_steps_margin: int = 4,
    observation_cap_chars: int = DEFAULT_OBSERVATION_CAP_CHARS,
    observation_head_share: float = OBSERVATION_HEAD_SHARE,
) -> list[int]:
    """The per-step `observation_cap` prompt sizes this geometry produces under perfect play."""
    policy = ContextPolicy(
        name=POLICY_OBSERVATION_CAP,
        observation_cap_chars=observation_cap_chars,
        observation_head_share=observation_head_share,
    )
    episodes = [
        run_episode(
            AgenticTask.from_record(record),
            oracle_controller,
            max_steps=depth + max_steps_margin,
            policy=policy,
            budget=unbounded_budget(),
        ).telemetry.prompt_chars
        for record in build_memory_dependent_tasks(
            n_tasks=n_tasks, depth=depth, pad_chars=pad_chars
        )
    ]
    if len({len(sizes) for sizes in episodes}) != 1:
        raise ValueError("perfect play produced different step counts across the task set")
    return [max(step) for step in zip(*episodes, strict=True)]


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
    return max(
        cap_prompt_sequence(
            depth=depth,
            n_tasks=n_tasks,
            pad_chars=pad_chars,
            max_steps_margin=max_steps_margin,
            observation_cap_chars=observation_cap_chars,
            observation_head_share=observation_head_share,
        )
    )


def first_fold_step(prompt_sequence: list[int], trigger_chars: int) -> int | None:
    """The 1-based step at which a `compact` trigger first fires, or None if it never does.

    The trigger is what the policy actually reacts to, and it reaches the transcript only through
    THIS step: two different (share, guard) pairs with the same trigger fold the same step, which
    is why a cost delta measured at one pair is expected to hold at the other.
    """
    for step, size in enumerate(prompt_sequence, start=1):
        if size > trigger_chars:
            return step
    return None


def fold_step_trigger_interval(prompt_sequence: list[int], step: int) -> tuple[int, int]:
    """The half-open trigger interval `[low, high)` whose every value folds at `step`.

    The inverse of `first_fold_step`, and the reason the cost delta is a STEP function: a trigger
    selects `step` exactly when it is at least every earlier step's prompt (so nothing folded
    sooner) and below this step's own prompt. Every trigger inside the interval therefore produces
    the identical transcript. The interval is EMPTY (`low >= high`) when an earlier step already
    reached this one's size, because no trigger can select such a step.
    """
    if not 1 <= step <= len(prompt_sequence):
        raise ValueError(f"fold step {step} is outside a {len(prompt_sequence)}-step sequence")
    return max(prompt_sequence[: step - 1], default=0), prompt_sequence[step - 1]


def reachable_fold_steps(prompt_sequence: list[int]) -> list[int]:
    """Every 1-based step some trigger can actually select, in order."""
    return [
        step
        for step in range(1, len(prompt_sequence) + 1)
        if _has_triggers(fold_step_trigger_interval(prompt_sequence, step))
    ]


def compaction_trigger_chars(max_prompt_chars: int, compact_share: float) -> int:
    """The trigger the runtime will compute, taken from the runtime's own arithmetic."""
    return fixed_budget(max_prompt_chars).compaction_trigger_chars(compact_share)


def smallest_guard_reaching(trigger_chars: int, compact_share: float) -> int:
    """The smallest prompt guard whose runtime trigger reaches `trigger_chars` at this share.

    Resolved against `compaction_trigger_chars` itself rather than by dividing, so the truncation
    the runtime performs -- not a float inverse of it -- decides which guard lands in which step.
    """
    if not 0.0 < compact_share <= 1.0:
        raise ValueError(f"compact share must be in (0, 1], got {compact_share}")
    guard = max(math.ceil(trigger_chars / compact_share), 0)
    while guard > 0 and compaction_trigger_chars(guard - 1, compact_share) >= trigger_chars:
        guard -= 1
    while compaction_trigger_chars(guard, compact_share) < trigger_chars:
        guard += 1
    return guard


def fold_step_guard_interval(
    prompt_sequence: list[int], step: int, compact_share: float
) -> tuple[int, int]:
    """The half-open prompt-guard interval `[low, high)` that folds at `step` for this share."""
    low, high = fold_step_trigger_interval(prompt_sequence, step)
    return (
        smallest_guard_reaching(low, compact_share),
        smallest_guard_reaching(high, compact_share),
    )


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


def _has_triggers(interval: tuple[int, int]) -> bool:
    low, high = interval
    return low < high
