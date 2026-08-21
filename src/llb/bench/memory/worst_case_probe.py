"""The cap peak perfect play cannot certify: what the STEP BUDGET lets a real controller reach.

Every cap-fitting band in this repo is placed against `cap_prompt_sequence`, which walks the
memory-dependent world with an ORACLE controller. That walk is the SHORTEST one that finishes: it
plays the freshest token every time and calls `finish` the step after the workflow completes, so
the peak it reports is the perfect-play peak. A real controller does not get that guarantee. It
re-sends a stale token, or keeps calling `advance` after the workflow says stop -- the failure the
task prompt shouts about precisely because models do it -- and every wasted step appends one more
transcript entry that every later prompt then carries. A guard chosen just above the perfect-play
peak can therefore still overflow on the run, and an overflowed cap arm measures rescue rather than
cost.

The gap is bounded and measurable, because the step budget bounds it. An episode runs at most
`max_steps` steps, so at most `max_steps - 1` transcript entries can stand behind its last prompt;
perfect play uses `depth + 1` of those steps, and whatever is left is what an imperfect controller
can spend. Walking the world with a controller that spends ALL of it prices the gap exactly, with
no model, in the same deterministic tool world the perfect-play walk uses.

WHICH imperfect walk is the worst one is decided by the tool world rather than assumed. A wasted
step appends the observation the world returns for it, so the largest transcript comes from the
wasted call with the largest observation: `advance` past the end of the workflow returns the
workflow-complete notice, which is longer than the wrong-token line a stale token returns. Playing
the chain perfectly and then stalling on `advance` therefore keeps every workflow observation AND
makes every wasted entry the largest one available -- so `stalling_controller` is the worst case
within the family of controllers that only drive the workflow tool.

That family is the boundary of the claim, and it is a deliberate one: a controller free to call
`write_file` with arbitrary content can grow a prompt without limit, so there is no worst case to
state over all controllers. The margin below prices imperfect play of the task, not adversarial use
of the sandbox.
"""

import json
import re
from typing import cast

from llb.bench.agentic.context_policy import (
    DEFAULT_OBSERVATION_CAP_CHARS,
    DEFAULT_SUMMARY_INPUT_CAP,
    OBSERVATION_HEAD_SHARE,
)
from llb.bench.agentic.context_summary import is_summary_prompt
from llb.bench.memory.boundary.probe import (
    ORACLE_SUMMARY,
    cap_prompt_sequence,
    compact_fold_input_probe,
)
from llb.bench.memory.fold_step.ladder import measured_cap_peak
from llb.bench.memory.transcript import DEFAULT_MEMORY_PAD_CHARS
from llb.bench.tool_world import ADVANCE

_NEXT_TOKEN = re.compile(r"\[next token: ([^\]]+)\]")
_START_TOKEN = re.compile(r'токеном "([^"]+)"')

# A geometry whose step budget leaves no room for a wasted step has no worst case to measure: its
# imperfect-play peak IS its perfect-play peak, and a zero margin certified as a margin would state
# a safety property the run cannot have. Refuse it rather than publish a vacuous zero.
MIN_BUDGETED_EXTRA_STEPS = 1


def stalling_controller(prompt: str) -> str:
    """Walk the token chain perfectly and then NEVER finish -- the whole step budget, spent.

    Identical to the oracle up to the workflow's last observation; from there it keeps calling
    `advance`, which the world answers with the workflow-complete notice and no progress. The
    episode ends on the step limit instead of on `finish`, which is exactly the transcript a
    controller that misses the stop cue produces.
    """
    tokens = _NEXT_TOKEN.findall(prompt) or _START_TOKEN.findall(prompt)
    token = tokens[-1] if tokens else ""
    return json.dumps({"name": ADVANCE, "arguments": {"token": token}}, ensure_ascii=False)


def stalling_compacting_controller(prompt: str) -> str:
    """The stalling walk plus the probe's fixed summary reply, so a compact walk stays model-free."""
    return ORACLE_SUMMARY if is_summary_prompt(prompt) else stalling_controller(prompt)


def worst_case_cap_prompt_sequence(
    *,
    depth: int,
    n_tasks: int,
    pad_chars: int = DEFAULT_MEMORY_PAD_CHARS,
    max_steps_margin: int = 4,
    observation_cap_chars: int = DEFAULT_OBSERVATION_CAP_CHARS,
    observation_head_share: float = OBSERVATION_HEAD_SHARE,
) -> list[int]:
    """The per-step `observation_cap` prompt sizes when the controller spends the whole budget."""
    return cap_prompt_sequence(
        depth=depth,
        n_tasks=n_tasks,
        pad_chars=pad_chars,
        max_steps_margin=max_steps_margin,
        observation_cap_chars=observation_cap_chars,
        observation_head_share=observation_head_share,
        controller=stalling_controller,
    )


def worst_case_fold_input_probe(
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
) -> dict[str, object]:
    """What the summarizer is offered on the longest transcript the step budget allows.

    The perfect-play probe answers what the fold offers on the shortest finishing transcript. A
    summarize-input cap the oracle transcript never touches can still be reached by a longer real
    one, so an elision verdict read only under perfect play is a verdict about the easy case.
    """
    return compact_fold_input_probe(
        depth=depth,
        n_tasks=n_tasks,
        max_prompt_chars=max_prompt_chars,
        compact_share=compact_share,
        summary_input_cap=summary_input_cap,
        pad_chars=pad_chars,
        max_steps_margin=max_steps_margin,
        observation_cap_chars=observation_cap_chars,
        observation_head_share=observation_head_share,
        controller=stalling_compacting_controller,
    )


def cap_peak_margin(
    *,
    depth: int,
    n_tasks: int,
    pad_chars: int = DEFAULT_MEMORY_PAD_CHARS,
    max_steps_margin: int = 4,
    observation_cap_chars: int = DEFAULT_OBSERVATION_CAP_CHARS,
    observation_head_share: float = OBSERVATION_HEAD_SHARE,
) -> dict[str, object]:
    """Both peaks for one geometry, and the safety margin between them.

    `margin_chars` is the number a guard must clear ON TOP of the perfect-play peak before the cell
    it places is cap-fitting for the controller that actually runs it. It is measured, not chosen:
    two walks of the same deterministic world, one that finishes as early as it can and one that
    finishes as late as the step budget lets it.
    """
    geometry = {
        "depth": depth,
        "n_tasks": n_tasks,
        "pad_chars": pad_chars,
        "max_steps_margin": max_steps_margin,
        "observation_cap_chars": observation_cap_chars,
        "observation_head_share": observation_head_share,
    }
    perfect = cap_prompt_sequence(**geometry)  # type: ignore[arg-type]
    worst = worst_case_cap_prompt_sequence(**geometry)  # type: ignore[arg-type]
    perfect_peak = measured_cap_peak(perfect, geometry=f"depth {depth} under perfect play")
    worst_peak = measured_cap_peak(worst, geometry=f"depth {depth} under imperfect play")
    budgeted = len(worst) - len(perfect)
    if budgeted < MIN_BUDGETED_EXTRA_STEPS:
        raise ValueError(
            f"depth {depth} leaves {budgeted} step(s) beyond the {len(perfect)} perfect play "
            f"needs, so its step budget (margin {max_steps_margin}) admits no imperfect play and "
            "there is no safety margin to certify a guard against"
        )
    return {
        "depth": depth,
        "perfect_play_peak_chars": perfect_peak,
        "worst_case_peak_chars": worst_peak,
        "margin_chars": worst_peak - perfect_peak,
        "margin_ratio": worst_peak / perfect_peak,
        "perfect_play_steps": len(perfect),
        "worst_case_steps": len(worst),
        "budgeted_extra_steps": budgeted,
    }


def margin_peaks(margin: dict[str, object]) -> tuple[int, int]:
    """The `(worst_case, perfect_play)` peaks a margin record places a guard band with."""
    return (
        int(cast(int, margin["worst_case_peak_chars"])),
        int(cast(int, margin["perfect_play_peak_chars"])),
    )
