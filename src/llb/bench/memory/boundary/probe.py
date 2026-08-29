"""Deterministic worst-case prompt probe for cap-fitting boundary geometry.

A cap-fitting cell is only usable when the prompt guard sits in a narrow band: ABOVE the largest
prompt `observation_cap` builds under perfect play (otherwise cap overflows and the cell measures
overflow rescue, not cost), and BELOW that peak divided by the compact trigger share (otherwise
compact never fires and the cell is inactive). Both bounds are computable with NO model: the
memory-dependent tool world is deterministic, so an oracle controller that always plays the next
workflow token reproduces the exact prompt sequence a perfect controller would send.

Every probe here RUNS episodes, so each call costs a full workflow walk per task. The interval
arithmetic those walks feed -- which step a trigger folds at, which guards select it, and which
band is cap-fitting -- is pure and lives in `agentic_memory_fold_step_ladder`.

The CONTROLLER is a parameter rather than a constant, because perfect play is one walk of this
world and not the only one a guard has to survive. `agentic_memory_worst_case_probe` passes the
imperfect controller that spends the whole step budget, and reads the same fields back.
"""

import json
import re
from collections.abc import Callable

from llb.bench.agentic.context_policy import (
    DEFAULT_OBSERVATION_CAP_CHARS,
    DEFAULT_SUMMARY_INPUT_CAP,
    DEFAULT_SUMMARY_TRIM_STRATEGY,
    OBSERVATION_HEAD_SHARE,
    POLICY_COMPACT,
    POLICY_OBSERVATION_CAP,
    ContextPolicy,
)
from llb.bench.agentic.context_summary import is_summary_prompt
from llb.backends.context_budget import fixed_budget, unbounded_budget
from llb.bench.agentic.episode import run_episode
from llb.bench.agentic.model import AgenticTask
from llb.bench.memory.fold_step.ladder import measured_cap_peak
from llb.bench.memory.transcript import (
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
# What a fitted fold length is padded with, so a probed summary of a measured length is
# ordinary prose rather than one repeated character.
ORACLE_SUMMARY_FILLER = "стислий підсумок попередніх кроків; "


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


def fold_length_controller(summary_chars: int) -> Callable[[str], str]:
    """The oracle walk with a summarizer that writes EXACTLY `summary_chars` characters.

    `oracle_compacting_controller` answers every summarize call with one short constant, which
    makes the walk deterministic and makes the fold count a property of the geometry alone. That
    is the right probe for placing a cap-fitting band and the wrong one for asking how many times
    a PARTICULAR model folds: a longer running summary spends more of the guard in every later
    prompt, so the post-fold prompt re-crosses the trigger sooner. Handing the probe a measured
    fold length turns the same model-free walk into a per-family one.
    """
    if summary_chars < 0:
        raise ValueError(f"a fold length is a character count, got {summary_chars}")
    repeats = summary_chars // len(ORACLE_SUMMARY_FILLER) + 1
    body = (ORACLE_SUMMARY_FILLER * repeats)[:summary_chars]

    def controller(prompt: str) -> str:
        return body if is_summary_prompt(prompt) else oracle_controller(prompt)

    return controller


def compact_fold_input_probe(
    *,
    depth: int,
    n_tasks: int,
    max_prompt_chars: int,
    compact_share: float,
    summary_input_cap: str = DEFAULT_SUMMARY_INPUT_CAP,
    summary_trim_strategy: str = DEFAULT_SUMMARY_TRIM_STRATEGY,
    pad_chars: int = DEFAULT_MEMORY_PAD_CHARS,
    max_steps_margin: int = 4,
    observation_cap_chars: int = DEFAULT_OBSERVATION_CAP_CHARS,
    observation_head_share: float = OBSERVATION_HEAD_SHARE,
    controller: Callable[[str], str] = oracle_compacting_controller,
) -> dict[str, object]:
    """What ONE guard offers the summarizer under perfect play, and how much its cap elides.

    The elided span is the transcript the running summary is never shown, and it is decided with no
    model at all: the tool world is deterministic, so an oracle controller folding at the same step
    a real controller folds at offers the summarizer the same bytes. This is what lets a design
    predeclare that its reference arm actually HAS an elision to price, and that the step-aligned
    arm has none. `summary_fold_input_chars` is the per-fold breakdown of the summed
    `summary_input_chars`, so a multi-fold episode keeps each offered transcript addressable.

    `controller` swaps the walk without swapping the reading: the worst-case probe passes the
    controller that spends the whole step budget, and the elision fields then describe the
    transcript a real controller can grow rather than the shortest one that finishes.
    """
    tasks = [
        AgenticTask.from_record(record)
        for record in build_memory_dependent_tasks(
            n_tasks=n_tasks, depth=depth, pad_chars=pad_chars
        )
    ]
    return compact_tasks_fold_input_probe(
        tasks,
        max_steps=depth + max_steps_margin,
        max_prompt_chars=max_prompt_chars,
        compact_share=compact_share,
        summary_input_cap=summary_input_cap,
        summary_trim_strategy=summary_trim_strategy,
        observation_cap_chars=observation_cap_chars,
        observation_head_share=observation_head_share,
        controller=controller,
    )


def compact_tasks_fold_input_probe(
    tasks: list[AgenticTask],
    *,
    max_steps: int,
    max_prompt_chars: int,
    compact_share: float,
    summary_input_cap: str = DEFAULT_SUMMARY_INPUT_CAP,
    summary_trim_strategy: str = DEFAULT_SUMMARY_TRIM_STRATEGY,
    observation_cap_chars: int = DEFAULT_OBSERVATION_CAP_CHARS,
    observation_head_share: float = OBSERVATION_HEAD_SHARE,
    controller: Callable[[str], str] = oracle_compacting_controller,
) -> dict[str, object]:
    """Probe arbitrary deterministic agent tasks through the same compact episode seam."""
    policy = ContextPolicy(
        name=POLICY_COMPACT,
        observation_cap_chars=observation_cap_chars,
        observation_head_share=observation_head_share,
        compact_share=compact_share,
        summary_input_cap=summary_input_cap,
        summary_trim_strategy=summary_trim_strategy,
    )
    telemetries = [
        run_episode(
            task,
            controller,
            max_steps=max_steps,
            policy=policy,
            budget=fixed_budget(max_prompt_chars),
        ).telemetry
        for task in tasks
    ]
    fold_inputs = _max_by_fold_ordinal([item.summary_fold_input_chars for item in telemetries])
    fold_steps = _max_by_fold_ordinal([item.summary_fold_steps for item in telemetries])
    return {
        "n_compactions": max(item.n_compactions for item in telemetries),
        "compaction_prompt_chars": max(item.compaction_prompt_chars for item in telemetries),
        # What the whole episode would send a model under perfect play -- controller prompts plus
        # summarize calls. It is the cost side of the same deterministic walk the fold fields
        # describe, so a design can predeclare what a cell costs before any GPU is warmed.
        "model_input_prompt_chars": max(item.model_input_prompt_chars for item in telemetries),
        "summary_input_chars": max(item.summary_input_chars for item in telemetries),
        "summary_input_elided_chars": max(item.summary_input_elided_chars for item in telemetries),
        "n_trimmed_summary_inputs": max(item.n_trimmed_summary_inputs for item in telemetries),
        "summary_fold_input_chars": fold_inputs,
        # WHEN each fold lands, beside what it offered. A walk that ends before its first fold
        # step never enters the folding regime at all, so this is what a per-family guard fit
        # measures a family's walk length against.
        "summary_fold_steps": fold_steps,
    }


def _max_by_fold_ordinal(per_task: list[list[int]]) -> list[int]:
    """One value per fold ordinal, taking the max across tasks at each ordinal.

    The probe's other fields already take a max across tasks so a single worst-case episode names
    the geometry; the per-fold lists do the same ordinal-wise so a multi-fold answer stays about
    one fold at a time rather than a mix of tasks.
    """
    if not per_task:
        return []
    n_folds = max((len(folds) for folds in per_task), default=0)
    return [
        max((folds[index] for folds in per_task if index < len(folds)), default=0)
        for index in range(n_folds)
    ]


def cap_prompt_sequence(
    *,
    depth: int,
    n_tasks: int,
    pad_chars: int = DEFAULT_MEMORY_PAD_CHARS,
    max_steps_margin: int = 4,
    observation_cap_chars: int = DEFAULT_OBSERVATION_CAP_CHARS,
    observation_head_share: float = OBSERVATION_HEAD_SHARE,
    controller: Callable[[str], str] = oracle_controller,
) -> list[int]:
    """The per-step `observation_cap` prompt sizes this geometry produces under one controller.

    Defaulted to perfect play, which is the walk every cap-fitting band was placed against. A
    caller that hands a different controller gets that controller's prompt sizes over the same
    world, and the step-count agreement check below still applies -- the sequence is only a
    geometry when every task walks it the same number of steps.
    """
    policy = ContextPolicy(
        name=POLICY_OBSERVATION_CAP,
        observation_cap_chars=observation_cap_chars,
        observation_head_share=observation_head_share,
    )
    episodes = [
        run_episode(
            AgenticTask.from_record(record),
            controller,
            max_steps=depth + max_steps_margin,
            policy=policy,
            budget=unbounded_budget(),
        ).telemetry.prompt_chars
        for record in build_memory_dependent_tasks(
            n_tasks=n_tasks, depth=depth, pad_chars=pad_chars
        )
    ]
    if len({len(sizes) for sizes in episodes}) != 1:
        raise ValueError("the probed controller produced different step counts across the task set")
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
    return measured_cap_peak(
        cap_prompt_sequence(
            depth=depth,
            n_tasks=n_tasks,
            pad_chars=pad_chars,
            max_steps_margin=max_steps_margin,
            observation_cap_chars=observation_cap_chars,
            observation_head_share=observation_head_share,
        ),
        geometry=f"depth {depth} under perfect play",
    )
