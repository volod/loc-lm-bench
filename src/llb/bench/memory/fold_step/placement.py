"""Where a fold-step ladder's cells must sit -- the placement rules, checked with no model.

A grid can only tell a step change apart from a smooth slide when its cells are placed against the
deterministic step ladder rather than against round guard numbers. These rules are shared by every
study stated on the fold-step axis (the crossover ladder and the summarize-input-cap arms), so a
residual measured by one is on the same scale the other publishes.

Each check takes a `label` naming the ladder it is validating, because the same rule fires for a
depth in one study and for an arm in another and the message has to say which.
"""

from typing import cast

from llb.bench.memory.boundary.gate import SIDE_CAP_CHEAPER, SIDE_COMPACT_CHEAPER
from llb.bench.memory.fold_step.ladder import (
    compaction_trigger_chars,
    first_fold_step,
    fold_step_guard_interval,
    foldable_fold_steps,
    guard_is_cap_fitting,
)

EXPECTED_SIDES = (SIDE_COMPACT_CHEAPER, SIDE_CAP_CHEAPER)


def step_guards(step: dict[str, object]) -> list[int]:
    """One declared step's prompt guards, ascending."""
    return sorted(
        int(cast(int, cell.get("max_prompt_chars", 0)))
        for cell in cast(list[dict[str, object]], step.get("cells", []))
    )


def validate_ladder_shape(label: str, steps: list[dict[str, object]], sequence: list[int]) -> None:
    """Two or more increasing fold steps, ADJACENT on the foldable ladder, sides in cost order.

    The ladder is the steps an episode can actually FOLD at, not every step a trigger can select: a
    step whose prompt is built from zero entries is reachable by a small enough guard and still folds
    nothing, so a cell declared there would measure a `compact` arm that never compacts and publish
    it as a fold-step cost.
    """
    declared = [int(cast(int, step.get("fold_step", 0))) for step in steps]
    if len(steps) < 2 or declared != sorted(declared) or len(set(declared)) != len(declared):
        raise ValueError(f"{label} needs two or more fold steps in increasing order")
    ladder = foldable_fold_steps(sequence)
    positions = [ladder.index(step) for step in declared if step in ladder]
    if len(positions) != len(declared) or positions != list(
        range(positions[0], positions[0] + len(positions))
    ):
        raise ValueError(
            f"{label} must test fold steps that are ADJACENT on the foldable ladder {ladder}, "
            f"got {declared}"
        )
    sides = [str(step.get("expected_side", "")) for step in steps]
    if set(sides) != set(EXPECTED_SIDES) or sides != sorted(sides, key=EXPECTED_SIDES.index):
        raise ValueError(
            f"{label} must predeclare compact-cheaper fold steps below cap-cheaper ones"
        )


def validate_step_cells(
    label: str,
    step: dict[str, object],
    *,
    sequence: list[int],
    compact_share: float,
    peak_prompt_chars: int,
    minimum_guard_span_fraction: float,
    seen: set[str] | None = None,
) -> None:
    """One step's cells: uniquely named, on the step's own side, folding where they claim.

    The span rule is what stops "same step, same cost" from being measured over two nearly
    identical guards: the tested guards must cover at least a declared fraction of the step's whole
    guard interval.
    """
    fold_step = int(cast(int, step["fold_step"]))
    cells = cast(list[dict[str, object]], step.get("cells", []))
    ids = [str(cell.get("cell_id", "")) for cell in cells]
    guards = step_guards(step)
    if len(cells) < 2 or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{label} fold step {fold_step} needs two or more unique cells")
    if seen is not None:
        if seen & set(ids):
            raise ValueError(f"{label} fold step {fold_step} reuses a cell id")
        seen |= set(ids)
    if any(cell.get("expected_side") != step["expected_side"] for cell in cells):
        raise ValueError(f"{label} fold step {fold_step} cells must predeclare the step's own side")
    if len(set(guards)) != len(guards):
        raise ValueError(f"{label} fold step {fold_step} must test distinct guards")
    for guard in guards:
        predicted = first_fold_step(sequence, compaction_trigger_chars(guard, compact_share))
        if predicted != fold_step:
            raise ValueError(
                f"{label} guard {guard} folds at step {predicted}, not the declared {fold_step}"
            )
        if not guard_is_cap_fitting(guard, peak_prompt_chars, compact_share):
            raise ValueError(
                f"{label} guard {guard} falls outside the usable band around a "
                f"{peak_prompt_chars}-char cap peak, where cap fits and compact still activates"
            )
    low, high = fold_step_guard_interval(sequence, fold_step, compact_share)
    required = minimum_guard_span_fraction * (high - low)
    if guards[-1] - guards[0] < required:
        raise ValueError(
            f"{label} fold step {fold_step} guards span {guards[-1] - guards[0]} chars of its "
            f"[{low}, {high}) interval, under the declared {required:.0f}-char minimum"
        )


def validate_step_changes(label: str, steps: list[dict[str, object]], allowed_gap: int) -> None:
    """Each step change must be straddled closely, or a flip is not localized to the step change."""
    for low, high in zip(steps, steps[1:]):
        gap = step_guards(high)[0] - step_guards(low)[-1]
        if not 0 < gap <= allowed_gap:
            raise ValueError(
                f"{label} straddles the step {low['fold_step']}->{high['fold_step']} change by "
                f"{gap} chars, over the declared {allowed_gap}-char maximum"
            )


def validate_window(max_model_len: int, max_guard: int) -> None:
    """The declared window must be able to carry the widest guard the ladder tests."""
    from llb.optimize.tuning_space import CHARS_PER_TOKEN, PROMPT_HEADROOM_TOKENS

    required = int(max_guard / CHARS_PER_TOKEN) + PROMPT_HEADROOM_TOKENS
    if max_model_len < required:
        raise ValueError(
            f"declared max_model_len cannot carry the widest guard ({max_guard} chars needs about "
            f"{required} tokens)"
        )
