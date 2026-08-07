"""Fold-step ladder arithmetic: which step folds, and which guards select it.

Everything here is a pure function of a prompt-size SEQUENCE (what
`agentic_memory_boundary_probe.cap_prompt_sequence` measures once) plus a share or a guard. No
episode runs, no model is called, and no tool world is built, so these are cheap enough to call in
a tight loop -- the fold-step placement rules, the summarize-cap ladder, and the policy-change band
solver all sweep them over candidate guards.

The one runtime fact the ladder needs is how a prompt budget turns a guard into a compact trigger,
which it takes from the runtime's own arithmetic rather than restating the formula.
"""

import math

from llb.bench.agentic.context_budget import fixed_budget

# A fold needs a transcript to fold: `compact_state` returns False when the state holds no entry, so
# a step whose prompt is built from fewer live entries than this never folds however the guard moves.
MIN_LIVE_ENTRIES_TO_FOLD = 1


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


def live_entries_at_fold_step(step: int) -> int:
    """How many transcript entries the prompt at `step` is built from, before any fold.

    The loop appends one entry per completed step and rebuilds the prompt from the whole transcript,
    so the prompt at step `s` is built from `s - 1` entries -- and that count, not the guard, is what
    decides both whether a fold can happen at all and how much of the transcript it leaves live.
    """
    return step - 1


def foldable_fold_steps(prompt_sequence: list[int]) -> list[int]:
    """Every step a trigger can select AND an episode can actually fold at, in order.

    `reachable_fold_steps` answers a question about triggers alone: step 1 is reachable whenever the
    first prompt is non-empty, because a small enough guard trips on it. No episode folds there --
    the step-1 prompt is built from ZERO entries, so `compact_state` finds nothing older to summarize
    and returns False -- so a study that predeclares it measures a `compact` arm that never compacts,
    and a band condition stated there is about a fold that cannot happen. This is the ladder both
    read: the steps where a fold is a real event rather than a reachable trigger.
    """
    return [
        step
        for step in reachable_fold_steps(prompt_sequence)
        if live_entries_at_fold_step(step) >= MIN_LIVE_ENTRIES_TO_FOLD
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


def measured_cap_peak(prompt_sequence: list[int], *, geometry: str) -> int:
    """A measured sequence's largest prompt, or a refusal naming the GEOMETRY it was walked over.

    Reducing a probe walk to its peak is the one step every cap-fitting caller takes before the band
    arithmetic, and taking it as a bare `max` puts the refusal for a geometry that measured NOTHING
    two layers below the caller: `usable_guard_band` states the non-positive peak it will not build a
    band around, but an empty walk never reaches it -- the builtin fails first as `max() iterable
    argument is empty`, naming neither the geometry nor what the peak was wanted for. Every caller
    reads the peak through here, so the same fact reads the same way wherever it surfaces.
    """
    peak = max(prompt_sequence, default=0)
    if peak <= 0:
        raise ValueError(
            f"{geometry} measured no prompt under perfect play ({len(prompt_sequence)} steps), so "
            "it has no cap peak and no usable guard band to place cells in"
        )
    return peak


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
