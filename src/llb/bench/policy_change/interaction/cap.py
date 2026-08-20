"""What `observation_cap_chars` demands of a guard -- stated over the prompts the loop BUILDS.

The cap is the one auditable field whose entire effect is a prompt SIZE, so its conditions turn on a
distinction no other pair has to make. A replay calls a field silent when no prompt the episode
SENDS moves, and the loop builds one prompt it never sends: `step_prompt` assembles the step prompt,
finds it over the compaction trigger, folds, and rebuilds. That discarded prompt is cap-sensitive,
and everything downstream of the fold is not -- the summarize call is assembled from RAW
observations (`summarize_entries`), and so are the aggregate/memory facts folded into the summary --
so a fold that drops every entry the two caps disagree about leaves the post-fold prompt identical
under both. The cap can therefore move a size nobody observes, which is exactly the case a
`no geometry` answer has to answer rather than assume away.

So the cap's silence has THREE cases, and the conditions state all three:

  - the two caps build the identical prompt at every step -- nothing moves, sent or discarded, so
    the compound has nothing to work with and the fold step cannot move;
  - the caps move a prompt some arm SHOWS -- the per-field union already reports the cap alone;
  - the caps move ONLY prompts the fold discards -- the corner. The cap is then silent exactly while
    the `observation_cap` arm refuses that prompt too (it has no fold to hide it behind, so it either
    sends the moved prompt or overflows on it), which bounds the guard from ABOVE by the smaller of
    the two sizes. Reaching the compound needs a trigger INSIDE the moved range, which is the only
    way a discarded size becomes a moved fold step, and that bounds the guard from BELOW by the
    smallest guard whose trigger reaches the range. A compact share is at most 1, so such a guard is
    never smaller than the prompt it reaches: the two bounds cross at every fold step, and the
    corner closes as a proof instead of an assumption.

The per-entry arithmetic is what makes the third case decidable: a step prompt is a fixed scaffold
plus one line per live entry, so the first difference of a cap's prompt sequence is that entry's own
rendered length, and "the fold discards every entry the caps disagree about" is a comparison of two
already-computed sequences rather than a second replay.
"""

from dataclasses import dataclass
from typing import Any

from llb.bench.memory.boundary.probe import cap_prompt_sequence
from llb.bench.memory.fold_step.ladder import (
    live_entries_at_fold_step,
    smallest_guard_reaching,
)
from llb.bench.policy_change.interaction.conditions import folds_at_this_step
from llb.bench.policy_change.interaction.terms import (
    FIELD_CAP,
    FIELD_KEEP_RECENT,
    FIELD_SHARE,
    BandCondition,
    StepConditions,
    StepGeometry,
    shipped_policy_value,
)

CONDITION_SILENT = "the_cap_audited_alone_is_silent"
CONDITION_MOVES_FOLD = "the_cap_moves_the_fold_step"


@dataclass(frozen=True, slots=True)
class MovedPrompts:
    """Which prompts two observation caps build differently, and how large the first of them is."""

    # The 1-based transcript entries whose rendered length the two caps disagree about.
    entries: tuple[int, ...]
    # The 1-based step whose prompt is the first to move, and the two sizes it moves between.
    step: int
    smallest: int
    largest: int

    @classmethod
    def whole_episode(cls, common_steps: int) -> "MovedPrompts":
        """Two caps that play different STEP COUNTS moved the episode, not one prompt inside it.

        No fold can discard a step only one cap ever plays, so the corner does not apply and there is
        no pair of sizes to bound a guard with -- reported as a move of zero width, which admits no
        guard and blocks the fold-step condition outright.
        """
        return cls(entries=(common_steps,), step=common_steps + 1, smallest=0, largest=0)

    @property
    def label(self) -> str:
        return f"step {self.step}'s prompt ({self.smallest} vs {self.largest} chars)"


def observation_cap_conditions(step: StepGeometry) -> StepConditions:
    """`observation_cap_chars` x anything: either it moves a prompt some arm shows, or nothing.

    The cap reaches the loop through the RENDERED observations alone, so it never reaches the
    summarizer except by deciding which entries a fold folded. What it can still do unobserved is
    move the prompt the fold discards -- see the module docstring for why that closes rather than
    separating.
    """
    caps = step.moved(FIELD_CAP)
    sequences = [cap_prompt_sequence(**{**step.geometry, FIELD_CAP: int(cap)}) for cap in caps]
    return cap_step_conditions(step, sequences[0], sequences[1])


def cap_step_conditions(step: StepGeometry, before: list[int], after: list[int]) -> StepConditions:
    """The same three conditions read on two STATED prompt sequences.

    Split from the probe so the corner is testable: the shipped task world pads every observation
    the same way, so a cap that re-trims the first one re-trims them all and no sequence it can
    build confines a move to the discarded prompt.
    """
    caps = step.moved(FIELD_CAP)
    moved = moved_prompts(before, after)
    folded = folded_entries(step.step, keep_recent_at_fold(step))
    return StepConditions(
        detail=_detail(caps, moved, folded),
        conditions=(
            folds_at_this_step(step),
            _silent_audited_alone(moved, folded),
            _moves_the_fold_step(moved, largest_share(step)),
        ),
    )


def moved_prompts(before: list[int], after: list[int]) -> MovedPrompts | None:
    """What two caps' prompt sequences disagree about, or None when they agree everywhere.

    Compared entry by entry rather than prompt by prompt: two caps that lengthen one entry and
    shorten a later one by as much build different prompts that a size comparison would call equal,
    and the corner turns on WHICH entries moved rather than on how the sizes came out.
    """
    if len(before) != len(after):
        return MovedPrompts.whole_episode(min(len(before), len(after)))
    entries = _moved_entries(before, after)
    if not entries:
        return None
    step = entries[0] + 1
    sizes = sorted((before[step - 1], after[step - 1]))
    return MovedPrompts(entries=entries, step=step, smallest=sizes[0], largest=sizes[1])


def folded_entries(step: int, keep_recent: int) -> int:
    """How many leading transcript entries the fold at `step` removes from the live view.

    `compact_state` folds everything but the last `compact_keep_recent` entries, and falls back to
    folding the WHOLE transcript when the keep would leave nothing to fold -- so the fold-everything
    steps are exactly those where `live_entries_at_fold_step(step) <= compact_keep_recent`, and they
    are the steps at which a cap-moved entry can vanish from every later prompt.
    """
    live = live_entries_at_fold_step(step)
    return live if live <= keep_recent else live - keep_recent


def keep_recent_at_fold(step: StepGeometry) -> int:
    """The keep the fold runs at: the change's BASELINE where it moves it, the shipped value else."""
    keep = step.change.baseline.get(FIELD_KEEP_RECENT, shipped_policy_value(FIELD_KEEP_RECENT))
    return int(keep)


def largest_share(step: StepGeometry) -> float:
    """The largest compact share the change can run at -- whose trigger reaches a prompt soonest.

    The compound reads the fold step at whichever share it can produce, so the condition that a
    trigger reaches the moved range is stated at the most generous of them: a smaller share only
    demands a larger guard.
    """
    shares = [step.share]
    if FIELD_SHARE in step.change.fields:
        shares.append(float(step.change.candidate[FIELD_SHARE]))
    return max(shares)


def _silent_audited_alone(moved: MovedPrompts | None, folded: int) -> BandCondition:
    """Whether the per-field union can miss this cap -- and, in the corner, for which guards."""
    if moved is None:
        return BandCondition.always(
            CONDITION_SILENT,
            "the two caps build the identical prompt at every step, sent or discarded",
        )
    shown = [entry for entry in moved.entries if entry > folded]
    if shown:
        return BandCondition.impossible(
            CONDITION_SILENT,
            f"a differently trimmed observation the fold leaves live is shown in a prompt the "
            f"episode sends; the fold here discards {_discarded(folded)} and entry {shown[0]} "
            f"({len(shown)} of {len(moved.entries)} moved entries) survives it",
        )
    return BandCondition(
        name=CONDITION_SILENT,
        why=(
            f"the only prompt the caps move is the one the fold discards ({moved.label}), so the "
            f"cap is silent only while the `observation_cap` arm -- which has no fold to hide it "
            f"behind -- refuses that prompt too: guards under {moved.smallest}"
        ),
        high=moved.smallest,
    )


def _moves_the_fold_step(moved: MovedPrompts | None, share: float) -> BandCondition:
    """Whether the compound can read the cap at all -- via a trigger inside the moved range."""
    if moved is None:
        return BandCondition.impossible(
            CONDITION_MOVES_FOLD,
            "the compound needs the cap to move a prompt size, which a cap that trims nothing "
            "differently never does -- in a prompt the episode sends or in one the fold discards",
        )
    return BandCondition(
        name=CONDITION_MOVES_FOLD,
        why=(
            f"a size the fold discards reaches the compound only by moving a fold step, which needs "
            f"a trigger inside [{moved.smallest}, {moved.largest}) at share {share}"
        ),
        low=smallest_guard_reaching(moved.smallest, share),
        high=smallest_guard_reaching(moved.largest, share),
    )


def _moved_entries(before: list[int], after: list[int]) -> tuple[int, ...]:
    """The 1-based transcript entries whose RENDERED length the two caps disagree about.

    A step prompt is the fixed scaffold plus one line per live entry, so entry `i` contributes
    exactly the first difference of the sequence at `i` and nothing else contributes there.
    """
    return tuple(
        entry
        for entry in range(1, len(before))
        if before[entry] - before[entry - 1] != after[entry] - after[entry - 1]
    )


def _detail(caps: tuple[Any, Any], moved: MovedPrompts | None, folded: int) -> str:
    move = "no prompt" if moved is None else moved.label
    return (
        f"caps {caps[0]} -> {caps[1]} move {move}, and the fold here discards {_discarded(folded)}"
    )


def _discarded(folded: int) -> str:
    """The entries a fold removes, named for a message -- a step with none folds nothing at all."""
    return f"entries 1-{folded}" if folded else "no entry"
