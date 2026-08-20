"""The elision arithmetic behind `compact_share` x `summary_input_cap`.

The pair's conditions are an inequality about ONE offered transcript: how many characters the
episode hands the summarizer at the fold this step selects, and which of the two shares elides it.
A multi-fold episode has several such transcripts, so picking the fold the inequality is stated
against -- and saying so when no fold separates the shares -- is the whole substance of this module.
`agentic_policy_change_interaction_conditions` states the conditions it returns.
"""

from typing import cast

from llb.bench.agentic.context_policy import SUMMARY_INPUT_CAP_WINDOW
from llb.bench.memory.boundary.probe import compact_fold_input_probe
from llb.bench.memory.fold_step.ladder import (
    smallest_guard_reaching,
)
from llb.bench.policy_change.audit import PolicyChange
from llb.bench.policy_change.interaction.terms import (
    FIELD_BOUND,
    FIELD_SHARE,
    SEPARATING_BOUNDS,
    StepGeometry,
)


def offered_for_elision(
    guard_chars: int,
    step: StepGeometry,
    *,
    baseline_share: float,
    candidate_share: float,
    both_low: int,
    both_high: int,
) -> tuple[int | None, str]:
    """The offered transcript the elision inequality is about, plus a note on later folds.

    Probed under the window bound so nothing is elided. The guard folds at this step by construction,
    so the first compaction is the fold this step names. When the episode folds again later, each
    later offered span is checked against the same both-shares interval: if none of them opens a
    band, the note records that the extra folds never separate; if one did, its offered span would
    widen the answer (the first fold that flips still names the inequality).
    """
    probe = compact_fold_input_probe(
        max_prompt_chars=guard_chars,
        compact_share=step.share,
        summary_input_cap=SUMMARY_INPUT_CAP_WINDOW,
        **step.geometry,
    )
    if int(cast(int, probe["summary_input_elided_chars"])) != 0:
        return (
            None,
            "the window-bound probe elided a summarize input, so the offered transcript is not the "
            "full fold",
        )
    fold_inputs = [int(chars) for chars in cast(list[int], probe["summary_fold_input_chars"])]
    if not fold_inputs:
        return None, "the probe measured no summarize call at a guard that folds at this step"
    chosen_index, offered = _first_fold_that_flips_elision(
        fold_inputs,
        baseline_share=baseline_share,
        candidate_share=candidate_share,
        both_low=both_low,
        both_high=both_high,
    )
    later = fold_inputs[chosen_index + 1 :]
    if not later:
        return offered, ""
    later_bands = [
        _elision_band(chars, baseline_share, candidate_share, both_low, both_high)
        for chars in later
    ]
    if any(low < high for low, high in later_bands):
        # A later fold opens a band of its own: keep the first flipping fold for the inequality and
        # name the widening so the report is not a silent subset of the multi-fold geometry.
        widened = ", ".join(
            f"fold {chosen_index + 1 + offset} offered {chars} -> [{low}, {high})"
            for offset, (chars, (low, high)) in enumerate(zip(later, later_bands, strict=True))
            if low < high
        )
        return offered, f"later folds also separate ({widened})"
    return offered, f"{len(later)} later fold(s) never separate"


def _first_fold_that_flips_elision(
    fold_inputs: list[int],
    *,
    baseline_share: float,
    candidate_share: float,
    both_low: int,
    both_high: int,
) -> tuple[int, int]:
    """The earliest fold whose elision inequalities leave a guard inside the both-shares interval.

    When every fold leaves that interval empty, the first fold still names the (empty) inequality --
    the multi-fold refusal is gone, and the empty band is the arithmetic answer rather than a blind
    spot.
    """
    for index, offered in enumerate(fold_inputs):
        low, high = _elision_band(offered, baseline_share, candidate_share, both_low, both_high)
        if low < high:
            return index, offered
    return 0, fold_inputs[0]


def _elision_band(
    offered: int,
    baseline_share: float,
    candidate_share: float,
    both_low: int,
    both_high: int,
) -> tuple[int, int]:
    """Guards inside the both-shares interval where baseline clears the offered and candidate does not."""
    low = max(both_low, smallest_guard_reaching(offered, baseline_share))
    high = min(both_high, smallest_guard_reaching(offered, candidate_share))
    return low, high


def separating_shares(change: PolicyChange) -> tuple[float, float]:
    """The two shares, once the change is confirmed to be one the band arithmetic can answer."""
    bounds = (change.baseline[FIELD_BOUND], change.candidate[FIELD_BOUND])
    if bounds != SEPARATING_BOUNDS:
        raise ValueError(
            f"only a {SEPARATING_BOUNDS[0]} -> {SEPARATING_BOUNDS[1]} bound move can separate the "
            f"two readings, got {bounds[0]} -> {bounds[1]}"
        )
    return (
        float(cast(float, change.baseline[FIELD_SHARE])),
        float(cast(float, change.candidate[FIELD_SHARE])),
    )
