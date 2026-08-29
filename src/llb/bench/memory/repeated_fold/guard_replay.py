"""The model-free replay one fitted guard is scored on, and how much slack its count has.

`guard_fit` decides WHICH guard a family runs; this is the walk that answers the question it
decides on -- how many times does this family fold at that guard -- with no model in it. The walk
is the deterministic oracle one every geometry probe uses, driven by a summarizer replaying the
family's measured fold length at the span each fold offers and steps padded to the family's
measured entry length.

Beside the count it also reports the count's SLACK. Every input the replay stands on is measured on
cells other than the one being fitted, so the length it replays is always a little wrong; scanning
either side of it for the point where the predicted count changes is what says whether a per-guard
count is something an operator can read on its own or only something the fit can rank by.
"""

from typing import Callable, cast

from llb.bench.memory.boundary.probe import compact_fold_input_probe, fold_length_controller
from llb.bench.context_policy.guard_band import median_int
from llb.bench.memory.repeated_fold.design import cell_geometry
from llb.bench.memory.repeated_fold.fold_span import FoldLengthModel

# How far the replayed fold length is scanned, either side, for the point where a guard's predicted
# fold count changes. That distance is the SLACK a per-guard count has: the replay takes its fold
# length off the control's fold, which covers a different span of transcript than the fitted cell's
# folds do, so the length handed to the probe is always a little wrong. A guard whose count survives
# this much error in it is one an operator can read without a confirming run; a guard whose count
# flips within it is one the fit can rank but not count. The scan is coarse on purpose -- the answer
# wanted is which side of the replay error the flip sits on, not the flip point to the character.
FOLD_COUNT_MARGIN_SCAN_CHARS = 200
FOLD_COUNT_MARGIN_STEP_CHARS = 10


def oracle_step_entry_chars(cell: dict[str, object], held: dict[str, object]) -> int:
    """What one step of the UNPADDED oracle walk appends to this cell's transcript.

    Measured over the geometry rather than assumed, because the world decides most of it: the
    workflow token the tool hands out is what the call carries, so the length is a property of the
    task set and the depth. It is the baseline a family's measured step length is read against --
    a family at this number grows its context exactly as fast as the walk the probe counts on.
    """
    return median_int(
        cast(
            list[int],
            compact_fold_input_probe(**cell_geometry(cell, held))["step_entry_chars"],  # type: ignore[arg-type]
        )
    )


def fold_count_margin_chars(
    predict: Callable[[int, int], int], guard: int, fold_length: int
) -> int:
    """How far the replayed fold length can be wrong before this guard's fold count changes.

    Scanned both ways rather than assumed monotone: a longer running summary usually crosses the
    trigger sooner, but it also folds sooner, and what a fold leaves behind is not monotone in what
    it replaced. The answer is the largest offset at which the count still holds, capped by the
    scan -- a guard that survives the whole scan is not what an inexact replay can break.
    """
    if fold_length <= 0:
        return 0
    base = predict(guard, fold_length)
    for offset in range(
        FOLD_COUNT_MARGIN_STEP_CHARS,
        FOLD_COUNT_MARGIN_SCAN_CHARS + 1,
        FOLD_COUNT_MARGIN_STEP_CHARS,
    ):
        below = predict(guard, max(0, fold_length - offset))
        above = predict(guard, fold_length + offset)
        if below != base or above != base:
            return offset - FOLD_COUNT_MARGIN_STEP_CHARS
    return FOLD_COUNT_MARGIN_SCAN_CHARS


def fold_count_predictor(
    cell: dict[str, object],
    held: dict[str, object],
    step_entry_chars: int,
    spans: FoldLengthModel,
) -> Callable[[int, int], int]:
    """Fold count as a pure function of (guard, fold length), memoized over the search.

    The walk is deterministic and every task in the set produces the same fold count for a given
    pair, so one probe answers for the whole case set and the grid costs one probe per distinct
    pair rather than one per case. The measured step length and the measured span slope are fixed
    for the whole fit -- they are one family's walk, not one case's -- so they stay out of the
    cache key, and the fold length in it is the case's LEVEL at the span the level was measured
    at, which the span model then moves to whatever span each replayed fold actually offers.
    """
    cache: dict[tuple[int, int], int] = {}

    def predict(guard: int, fold_length: int) -> int:
        key = (guard, fold_length)
        if key not in cache:
            geometry = cell_geometry({**cell, "max_prompt_chars": guard}, held)
            probe = compact_fold_input_probe(
                controller=fold_length_controller(
                    lambda offered: spans.length_at(fold_length, offered),
                    step_entry_chars=step_entry_chars,
                ),
                **geometry,  # type: ignore[arg-type]
            )
            cache[key] = int(cast(int, probe["n_compactions"]))
        return cache[key]

    return predict
