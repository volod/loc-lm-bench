"""Where the separating guards ARE -- solved from the interval arithmetic, not scanned for.

The interaction fixture (`agentic_policy_change_interaction_fixture`) separates the compound
policy-change reading from the per-field one only inside a narrow guard band, and the band is a
property of the task world: edit a prompt template by a few chars and every prompt size shifts, the
band moves, and the committed guards fall out of it. Found by hand, that costs a re-derivation.
Solved, it costs re-reading one line of a failure message.

Three conditions define the band at one fold step, and all three are intervals the boundary probe
already computes:

  - BOTH shares must fold at that step, or the share alone already changes the prompts. The trigger
    interval that selects a step is `fold_step_trigger_interval`, and `fold_step_guard_interval`
    inverts it into guards per share, so the fold-step condition is the intersection of the two
    shares' guard intervals.
  - the BASELINE share must elide nothing, or the bound audited alone already reports the change:
    `trigger(guard, baseline_share) >= offered`, which is `guard >= smallest_guard_reaching(offered,
    baseline_share)`.
  - the CANDIDATE share must elide, or the compound reading has nothing to report either:
    `trigger(guard, candidate_share) < offered`, which is `guard < smallest_guard_reaching(offered,
    candidate_share)`.

`offered` is the transcript the fold hands the summarizer, measured once per fold step with an
oracle controller and no model. The direction is fixed: the baseline bound must be `window` (so the
share alone and the bound alone both read as invariant) and the candidate `trigger` (so the moved
share moves the cap). The reverse direction cannot separate -- a baseline that already elides makes
the bound audited alone report the change -- and is refused rather than answered.

The bands this returns are exact, not approximate: `tests/llb/bench/test_agentic_policy_change_
interaction.py` replays the audit at both edges of a solved band and at the guards just outside it.
"""

from dataclasses import dataclass
from typing import Any, cast

from llb.bench.agentic.context import SUMMARY_INPUT_CAP_TRIGGER, SUMMARY_INPUT_CAP_WINDOW
from llb.bench.agentic_memory_boundary_probe import (
    cap_prompt_sequence,
    compact_fold_input_probe,
    fold_step_guard_interval,
    fold_step_trigger_interval,
    reachable_fold_steps,
    smallest_guard_reaching,
)
from llb.bench.agentic_policy_change_audit import PolicyChange
from llb.bench.agentic_policy_change_interaction_fixture import (
    FIELD_BOUND,
    FIELD_SHARE,
    INTERACTING_FIELDS,
    geometry_kwargs,
)

# The only direction that can separate: a baseline that elides nothing, and a candidate whose cap
# rides the moved share down onto the folded transcript.
SEPARATING_BOUNDS = (SUMMARY_INPUT_CAP_WINDOW, SUMMARY_INPUT_CAP_TRIGGER)


@dataclass(frozen=True, slots=True)
class SeparatingBand:
    """The half-open guard interval `[low, high)` that separates the two readings at one fold step.

    Empty (`low >= high`) is a real answer: at that fold step the offered transcript does not fall
    between the two shares' triggers, so no guard there tells the compound reading from the
    per-field one. `separating_guard_bands` drops the empty ones.
    """

    fold_step: int
    low: int
    high: int
    summary_input_chars: int
    trigger_interval: tuple[int, int]

    @property
    def is_empty(self) -> bool:
        return self.low >= self.high

    def contains(self, guard_chars: int) -> bool:
        return self.low <= guard_chars < self.high

    def describe(self) -> str:
        return (
            f"fold step {self.fold_step}: guards [{self.low}, {self.high}) separate "
            f"(offered {self.summary_input_chars}, folds at triggers "
            f"[{self.trigger_interval[0]}, {self.trigger_interval[1]}))"
        )


def separating_guard_bands(
    change: PolicyChange, *, depth: int, held: dict[str, object]
) -> list[SeparatingBand]:
    """Every guard band that separates the two readings at this depth, one per fold step."""
    shares = _shares(change)
    geometry = geometry_kwargs(depth, held)
    sequence = cap_prompt_sequence(**geometry)
    bands = [
        _band_at_fold_step(step, sequence=sequence, shares=shares, geometry=geometry)
        for step in reachable_fold_steps(sequence)
    ]
    return [band for band in bands if band is not None and not band.is_empty]


def band_at_fold_step(bands: list[SeparatingBand], fold_step: int) -> SeparatingBand | None:
    """The solved band for one fold step, or None when that step separates no guard at all."""
    return next((band for band in bands if band.fold_step == fold_step), None)


def format_band_report(change: PolicyChange, depth: int, bands: list[SeparatingBand]) -> str:
    """The failure message: what the geometry offers NOW, so a drifted guard has a replacement."""
    header = f"[band] depth {depth}, {change.label}"
    if not bands:
        return f"{header}\n  no fold step separates the two readings at this depth"
    return "\n".join([header, *(f"  {band.describe()}" for band in bands)])


def _band_at_fold_step(
    step: int,
    *,
    sequence: list[int],
    shares: tuple[float, float],
    geometry: dict[str, Any],
) -> SeparatingBand | None:
    """Intersect the fold-step condition with the two elision conditions, at one step."""
    baseline_share, candidate_share = shares
    folds = [fold_step_guard_interval(sequence, step, share) for share in shares]
    low, high = max(edge[0] for edge in folds), min(edge[1] for edge in folds)
    if low >= high:
        return None  # the two shares cannot both fold at this step, whatever the guard
    offered = _offered_at_fold_step(low, geometry=geometry, share=baseline_share)
    if offered is None:
        return None
    return SeparatingBand(
        fold_step=step,
        # The baseline share must not elide, and the candidate share must.
        low=max(low, smallest_guard_reaching(offered, baseline_share)),
        high=min(high, smallest_guard_reaching(offered, candidate_share)),
        summary_input_chars=offered,
        trigger_interval=fold_step_trigger_interval(sequence, step),
    )


def _offered_at_fold_step(
    guard_chars: int, *, geometry: dict[str, Any], share: float
) -> int | None:
    """The transcript ONE fold hands the summarizer, or None when the episode folds more than once.

    Probed under the window bound so nothing is elided, and at a guard that folds at the wanted step
    by construction. A second compaction would make `summary_input_chars` a sum over folds, and the
    band arithmetic above is a statement about ONE offered transcript -- so that step is reported as
    no band rather than as a wrong one.
    """
    probe = compact_fold_input_probe(
        max_prompt_chars=guard_chars,
        compact_share=share,
        summary_input_cap=SUMMARY_INPUT_CAP_WINDOW,
        **geometry,
    )
    if probe["n_compactions"] != 1 or probe["summary_input_elided_chars"] != 0:
        return None
    return probe["summary_input_chars"]


def _shares(change: PolicyChange) -> tuple[float, float]:
    """The two shares, once the change is confirmed to be one the band arithmetic can answer."""
    if change.fields != INTERACTING_FIELDS:
        raise ValueError(
            f"a separating band is only defined for {INTERACTING_FIELDS}, got {change.fields}"
        )
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
