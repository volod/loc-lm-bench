"""Where a published crossover sits on the fold-step ladder the study that published it measured.

A committed design states two independent things about each published number: the VALUE, and the
fold step it lands in. The restatement checks the value against the second one -- an interpolated
guard holds while it still names its step -- so the annotation has to be right before a run can mean
anything by it, and a mis-transcribed one turns into a spurious `..._moves_under_the_shipped_cap`
the moment a cell at that depth becomes bound-sensitive. Nothing about the check needs a model: the
ladder is a pure function of the study's own held geometry.

Each FORM places differently, which is the whole content of the check:

- an interpolated guard is a point INSIDE its step's guard interval;
- a fold-step boundary is that interval's exclusive upper EDGE, because "fold no later than step k"
  is the rule "keep the guard below this" -- the boundary value is the first guard that folds one
  step later, which is why it is not placed like a guard;
- a portable ratio has no guard of its own. It is DERIVED from another study's, so its annotation is
  checked against that study's row rather than against a value of its own.
"""

from typing import cast

from llb.bench.agentic_memory_boundary_probe import cap_prompt_sequence
from llb.bench.agentic_memory_fold_step_ladder import (
    compaction_trigger_chars,
    first_fold_step,
    fold_step_guard_interval,
    foldable_fold_steps,
)
from llb.bench.agentic_memory_crossover_restatement_reading import (
    FORM_FOLD_STEP,
    FORM_INTERPOLATED,
    FORM_PORTABLE_RATIO,
)
from llb.bench.agentic_policy_change_audit import KIND_SURFACE

# The study a portable ratio is derived from: it is `compact_share * guard` over a cap peak, and the
# guard is the boundary surface's interpolated one. Both the design check below and the restatement
# row that derives the ratio read this, so the two cannot disagree about the source.
DERIVED_RATIO_SOURCE_KIND = KIND_SURFACE


def prompt_sequence(design: dict[str, object], depth: int) -> list[int]:
    """The deterministic per-step prompt sizes an audited study's held geometry produces."""
    held = cast(dict[str, object], design["held_fixed"])
    return cap_prompt_sequence(
        depth=depth,
        n_tasks=int(cast(int, held["n_tasks"])),
        pad_chars=int(cast(int, held["pad_chars"])),
        max_steps_margin=int(cast(int, held["max_steps_margin"])),
        observation_cap_chars=int(cast(int, held["observation_cap_chars"])),
        observation_head_share=float(cast(float, held["observation_head_share"])),
    )


def validate_published_placement(
    kind: str, crossover: dict[str, object], audited: dict[str, object]
) -> None:
    """Refuse a published crossover whose fold-step annotation its own geometry does not support."""
    depth = int(cast(int, crossover["depth"]))
    step = int(cast(int, crossover["fold_step"]))
    share = float(cast(float, crossover["compact_share"]))
    sequence = prompt_sequence(audited, depth)
    ladder = foldable_fold_steps(sequence)
    if step not in ladder:
        raise ValueError(
            f"{kind} depth {depth}: the published crossover is annotated as fold step {step}, "
            f"which this study's geometry cannot fold at (its foldable ladder is {ladder})"
        )
    low, high = fold_step_guard_interval(sequence, step, share)
    form = str(crossover["form"])
    if form == FORM_INTERPOLATED:
        _validate_interpolated_placement(kind, crossover, sequence, (low, high))
    elif form == FORM_FOLD_STEP:
        _validate_boundary_placement(kind, crossover, high)
    # A portable ratio carries no guard to place; `validate_derived_placements` reads it against the
    # row it is derived from, which is the only value the design gives it.


def validate_derived_placements(crossovers: list[dict[str, object]]) -> None:
    """A derived ratio must name the fold step of the guard it is derived from.

    This is the only check the form admits, and it is worth more than it looks: it ties two studies'
    annotations together, so a depth where they disagree is refused whichever of the two is wrong.
    """
    source = {
        int(cast(int, row["depth"])): row
        for row in crossovers
        if row["study_kind"] == DERIVED_RATIO_SOURCE_KIND and row["form"] == FORM_INTERPOLATED
    }
    for row in crossovers:
        if row["form"] != FORM_PORTABLE_RATIO:
            continue
        kind, depth = row["study_kind"], int(cast(int, row["depth"]))
        guard = source.get(depth)
        if guard is None:
            raise ValueError(
                f"{kind} depth {depth}: a {FORM_PORTABLE_RATIO} crossover is derived from the "
                f"{DERIVED_RATIO_SOURCE_KIND} guard at its depth, and no such guard is published "
                "here, so nothing in the run can restate it"
            )
        if int(cast(int, row["fold_step"])) != int(cast(int, guard["fold_step"])):
            raise ValueError(
                f"{kind} depth {depth}: the derived ratio is annotated as fold step "
                f"{row['fold_step']} while the {DERIVED_RATIO_SOURCE_KIND} guard it is derived from "
                f"is annotated as fold step {guard['fold_step']} -- one of the two annotations is "
                "wrong, and the restatement would check them against different steps"
            )


def _validate_interpolated_placement(
    kind: str, crossover: dict[str, object], sequence: list[int], interval: tuple[int, int]
) -> None:
    value = float(cast(float, crossover["value"]))
    low, high = interval
    if low <= value < high:
        return
    share = float(cast(float, crossover["compact_share"]))
    lands = first_fold_step(sequence, compaction_trigger_chars(int(value), share))
    raise ValueError(
        f"{kind} depth {crossover['depth']}: the published {value:.0f}-char crossover guard is "
        f"annotated as fold step {crossover['fold_step']}, whose guard interval at share {share} is "
        f"[{low}, {high}) -- the guard folds at step {lands} instead, so the annotation the "
        "restatement checks a restated guard against is wrong before any bound changes"
    )


def _validate_boundary_placement(kind: str, crossover: dict[str, object], high: int) -> None:
    value = float(cast(float, crossover["value"]))
    if int(value) == high:
        return
    raise ValueError(
        f"{kind} depth {crossover['depth']}: a {FORM_FOLD_STEP} crossover is the EXCLUSIVE upper "
        f"edge of its step's guard interval -- 'fold no later than step {crossover['fold_step']}' "
        f"means keep the guard below it -- so the published {value:.0f} must be that step's own "
        f"{high} at share {crossover['compact_share']}"
    )
