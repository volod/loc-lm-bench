"""Where the separating guards ARE -- solved from the interval arithmetic, not scanned for.

The interaction fixture (`agentic_policy_change_interaction_fixture`) separates the compound
policy-change reading from the per-field one only inside a narrow guard band, and the band is a
property of the task world: edit a prompt template by a few chars and every prompt size shifts, the
band moves, and the committed guards fall out of it. Found by hand, that costs a re-derivation.
Solved, it costs re-reading one line of a failure message.

Three conditions define a band at one fold step, and they are the same three for every pair of
moved fields: each field audited ALONE must change no prompt, and the two together must change one.
What differs per pair is the arithmetic those statements turn into, so the conditions live with the
coupling (`agentic_policy_change_interaction_couplings`) and this module only intersects them.

For `compact_share` x `summary_input_cap` all three are intervals solvable before any run -- the two
shares' fold-step guard intervals, which are ladder arithmetic (`agentic_memory_fold_step_ladder`)
over one measured prompt sequence, and the two elision inequalities against the transcript the fold
offers the summarizer, which the boundary probe measures once per fold step by walking the episode
with an oracle controller and no model. The
direction is fixed there: the baseline bound must be `window` and the candidate `trigger`; the
reverse cannot separate and is refused rather than answered. Every OTHER pair states conditions that
contradict each other at every fold step, and the solver reports that as no band -- with the
condition that blocked it, so the answer reads as a derivation rather than as an empty list.

The bands this returns are exact, not approximate: `tests/llb/bench/test_agentic_policy_change_
interaction.py` replays the audit at both edges of a solved band and at the guards just outside it.
"""

from dataclasses import dataclass
from typing import cast

from llb.bench.memory.boundary.probe import cap_prompt_sequence
from llb.bench.memory.fold_step.ladder import foldable_fold_steps, reachable_fold_steps
from llb.bench.policy_change.audit import PolicyChange
from llb.bench.policy_change.interaction.terms import (
    FIELD_SHARE,
    BandCondition,
    StepGeometry,
)
from llb.bench.policy_change.interaction.couplings import Coupling, coupling_for
from llb.bench.policy_change.interaction.fixture import geometry_kwargs


@dataclass(frozen=True, slots=True)
class SeparatingBand:
    """The half-open guard interval `[low, high)` that separates the two readings at one fold step.

    Empty (`low >= high`) is a real answer: at that fold step the coupling's three conditions have
    no guard in common -- for the share/bound pair because the offered transcript does not fall
    between the two shares' triggers, for every other pair because two of the conditions contradict
    each other outright. `separating_guard_bands` drops the empty ones unless asked for them, and
    `blocked_by` names the conditions that emptied one.
    """

    fold_step: int
    low: int
    high: int
    detail: str
    conditions: tuple[BandCondition, ...]

    @property
    def is_empty(self) -> bool:
        return self.low >= self.high

    def contains(self, guard_chars: int) -> bool:
        return self.low <= guard_chars < self.high

    def blocked_by(self) -> tuple[BandCondition, ...]:
        """The conditions no guard satisfies -- why this fold step separates nothing."""
        return tuple(condition for condition in self.conditions if condition.is_empty)

    def describe(self) -> str:
        if self.is_empty:
            blocked = self.blocked_by()
            reason = (
                blocked[0].describe()
                if blocked
                else f"the conditions overlap nowhere ([{self.low}, {self.high}))"
            )
            return f"fold step {self.fold_step}: nothing separates ({self.detail}) -- {reason}"
        return (
            f"fold step {self.fold_step}: guards [{self.low}, {self.high}) separate ({self.detail})"
        )


def separating_guard_bands(
    change: PolicyChange, *, depth: int, held: dict[str, object], include_empty: bool = False
) -> list[SeparatingBand]:
    """Every guard band that separates the two readings at this depth, one per fold step.

    The steps are the ones an episode can actually fold at (`foldable_fold_steps`), not merely the
    ones a trigger can select: a step the loop never folds at states its three conditions about a
    fold that cannot happen, and answers with them anyway.

    `include_empty` keeps the fold steps that separate nothing, each carrying the condition that
    blocked it -- which is what a coupling with no separating geometry has to say for itself.
    """
    coupling = coupling_for(change.fields)
    geometry = geometry_kwargs(depth, held)
    sequence = cap_prompt_sequence(**geometry)
    share = _cell_share(change, held)
    bands = [
        _band_at_fold_step(
            coupling,
            StepGeometry(
                change=change, step=step, sequence=sequence, geometry=geometry, share=share
            ),
        )
        for step in _ladder(depth, sequence)
    ]
    return [band for band in bands if include_empty or not band.is_empty]


def _ladder(depth: int, sequence: list[int]) -> list[int]:
    """The foldable steps to solve at, or a refusal when the geometry has none.

    Every step the conditions are stated at comes from here, so the ladder rules the conditions
    call can never refuse a step through this path -- a `StepGeometry` naming a step off the
    sequence is a fabricated one, and it is left to fail in the ladder's own words. What IS
    reachable is a geometry with NO foldable step: the comprehension then yields nothing and the
    solver reports "no fold step separates the two readings", which reads as a derivation over the
    steps of an episode instead of as a probe that measured no episode. So the empty ladder is
    refused the way the placement rule refuses one, naming what the geometry does offer.
    """
    ladder = foldable_fold_steps(sequence)
    if not ladder:
        raise ValueError(
            f"depth {depth} has no foldable fold step on its measured {len(sequence)}-step prompt "
            f"sequence (reachable steps {reachable_fold_steps(sequence)}), so a no-band answer "
            "there would be a statement about an episode nothing was measured over"
        )
    return ladder


def band_at_fold_step(bands: list[SeparatingBand], fold_step: int) -> SeparatingBand | None:
    """The solved band for one fold step, or None when that step separates no guard at all."""
    return next((band for band in bands if band.fold_step == fold_step), None)


def format_band_report(change: PolicyChange, depth: int, bands: list[SeparatingBand]) -> str:
    """The failure message: what the geometry offers NOW, so a drifted guard has a replacement.

    Given the empty bands too (`include_empty`), the no-band answer also names the condition that
    blocked each fold step, which is the whole answer for a coupling no geometry can separate.
    """
    header = f"[band] depth {depth}, {change.label}"
    separating = [band for band in bands if not band.is_empty]
    if separating:
        return "\n".join([header, *(f"  {band.describe()}" for band in separating)])
    return "\n".join(
        [
            header,
            "  no fold step separates the two readings at this depth",
            f"  {coupling_for(change.fields).describe()}",
            *(f"  {reason}" for reason in _blocking_reasons(bands)),
        ]
    )


def _blocking_reasons(bands: list[SeparatingBand]) -> list[str]:
    """Why each fold step separates nothing: the condition that blocked it, deduplicated.

    A step whose conditions are all individually satisfiable but share no guard is a NEAR miss --
    the drift case, where the answer wanted is how far off the geometry now sits -- so those are
    listed per step rather than folded into one line.
    """
    seen: dict[str, str] = {}
    for band in bands:
        blocked = band.blocked_by()
        if not blocked:
            seen.setdefault(f"overlap@{band.fold_step}", band.describe())
            continue
        for condition in blocked:
            seen.setdefault(condition.name, f"fold step {band.fold_step}: {condition.describe()}")
    return list(seen.values())


def _band_at_fold_step(coupling: Coupling, step: StepGeometry) -> SeparatingBand:
    """Intersect one coupling's conditions at one fold step, into the band they leave open."""
    stated = coupling.conditions(step)
    low, high = _intersect(stated.conditions)
    return SeparatingBand(
        fold_step=step.step,
        low=low,
        high=high,
        detail=stated.detail,
        conditions=stated.conditions,
    )


def _intersect(conditions: tuple[BandCondition, ...]) -> tuple[int, int]:
    """The guards every condition admits. An unsatisfiable condition empties the whole intersection."""
    if any(condition.is_empty for condition in conditions):
        return 0, 0
    lows = [condition.low for condition in conditions if condition.low is not None]
    highs = [condition.high for condition in conditions if condition.high is not None]
    if not highs:
        raise ValueError(
            "a coupling's conditions must bound the guard from above; "
            f"{[condition.name for condition in conditions]} bound nothing"
        )
    return max(lows, default=0), min(highs)


def _cell_share(change: PolicyChange, held: dict[str, object]) -> float:
    """The compact share the geometry runs at: the change's BASELINE where the change moves it.

    Every condition that reads a fold step reads it at this share, because the baseline arm is the
    policy the published numbers were measured under -- the candidate share is what a condition
    compares AGAINST, never what it probes at.
    """
    share = change.baseline.get(FIELD_SHARE, held.get(FIELD_SHARE))
    if share is None:
        raise ValueError("the geometry states no compact_share and the change does not move it")
    return float(cast(float, share))
