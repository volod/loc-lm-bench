"""What a per-case fold length is worth once it is carried to another cell.

The guard fit replays ONE measured fold length per case: the length that case's summarizer wrote at
the one-fold control, moved to the span the fitted cell's folds offer. That construction carries a
per-case LEVEL across cells, and carrying it is only meaningful if verbosity is a property of the
case. It is not. Pair each case's control fold length against its own first fold at the fitted cell
and the correlation is near zero on both qualified families, so the level a case is handed is not
the level that case then writes at -- what transfers between the two cells is the family's central
tendency, and the case-to-case spread around it is noise the fit cannot assign to anyone.

That leaves the count the fit reports meaning something different from what it looks like. Summing
"how many of these twelve measured lengths land on the target rung" is NOT twelve per-case
predictions; it is the family's own spread, sampled twelve times, read as a RATE and multiplied
back up to a count. The number does not move -- which is why the guard the fit picks does not move
either -- but what it is worth does, and this module is where that is priced:

  - the count is stated as an INTERVAL, the Wilson interval on that rate scaled back to cases and
    rounded outward, so an operator reading a rung at an unfitted guard is told the width the
    family's own case-to-case spread leaves the count uncertain by;
  - a guard whose FLIP WINDOW -- how far the replayed length can move before the predicted count
    changes -- is narrower than that spread is refused as a count. The fit can still RANK it; what
    it cannot do is hand an operator a number whose error is inside the variation of the quantity
    it was measured from.

Nothing here recovers the transfer. The correlation is the run's own answer to whether a third
calibration point would help, and it says no: the level is a family constant plus an irreducible
width, and the width is what gets reported.
"""

import math
from typing import cast

from llb.conflicts.interval_stats import wilson_interval

ARM_TYPED_MARKER = "typed_marker"

# Below this, a case's fold length at one cell explains under a quarter of the variance of its own
# fold length at another, which is not enough to carry a per-case level across a cell boundary. The
# threshold is deliberately generous: the finding it has to survive is a correlation near zero.
LEVEL_TRANSFER_MIN_CORRELATION = 0.5

# Two points define a line, so a correlation over two pairs is +/-1 by construction and says
# nothing. Three is the smallest pairing that can come back near zero on its own.
LEVEL_TRANSFER_MIN_PAIRS = 3

LEVEL_TRANSFER_ABSENT = "the_case_fold_length_level_does_not_transfer_between_cells"
LEVEL_TRANSFER_PRESENT = "the_case_fold_length_level_transfers_between_cells"
LEVEL_TRANSFER_UNMEASURED = "no_case_measured_a_fold_length_in_both_cells_to_pair"

LEVEL_CONSTANT_PER_FAMILY = "no_qualified_family_carries_a_per_case_fold_length_level"
LEVEL_PER_CASE = "a_qualified_family_carries_a_per_case_fold_length_level"
LEVEL_TRANSFER_UNREAD = "no_qualified_family_paired_a_fold_length_level_across_two_cells"

COUNT_READABLE = "the_flip_window_is_wider_than_the_case_to_case_fold_length_spread"
COUNT_RANK_ONLY = "the_flip_window_is_narrower_than_the_case_to_case_fold_length_spread"
COUNT_UNMEASURED = "no_fold_length_spread_was_measured_to_widen_a_count_with"


def fold_length_spread_chars(fold_lengths: list[int]) -> int:
    """How far apart this family's cases wrote, which is the width the level rule cannot remove."""
    return max(fold_lengths) - min(fold_lengths) if fold_lengths else 0


def count_interval(successes: int, total: int) -> list[int]:
    """The predicted case count as an interval, from the rate the measured lengths estimate.

    The count is `total` times a rate estimated from `total` measured fold lengths, so its width is
    the Wilson interval on that rate scaled back to cases. Rounded OUTWARD: an interval on a count
    that rounds inward is one that can exclude the count it was meant to cover.
    """
    if total <= 0:
        return []
    low, high = wilson_interval(successes, total)
    return [math.floor(low * total), math.ceil(high * total)]


def count_reading(margin_chars: int, spread_chars: int) -> str:
    """Whether a guard's predicted count is readable on its own, or only rankable.

    The margin is how far the replayed fold length can be wrong before the count changes; the
    spread is how far the family's own cases sit from each other. A margin inside the spread means
    the replay is being asked to resolve a window narrower than the noise of the quantity it
    replays, and no amount of calibration taken on another cell of the same run closes that.
    """
    if spread_chars <= 0:
        return COUNT_UNMEASURED
    return COUNT_READABLE if margin_chars >= spread_chars else COUNT_RANK_ONLY


def paired_case_levels(
    cells: list[dict[str, object]], source_cell_id: str, fitted_cell_id: str
) -> list[tuple[int, int]]:
    """Each case's control fold length beside its OWN first fold at the cell the fit ran.

    First fold rather than median: the fitted cell folds twice, and the second fold covers a span
    the control never offered, so the first is the only fold of that cell asking the same question
    of the summarizer that the control's single fold asked.
    """
    source = _first_fold_lengths(cells, source_cell_id)
    fitted = _first_fold_lengths(cells, fitted_cell_id)
    return [(source[item], fitted[item]) for item in sorted(set(source) & set(fitted))]


def level_correlation(pairs: list[tuple[int, int]]) -> float:
    """Pearson correlation between the level a case was handed and the level it then wrote."""
    if len(pairs) < LEVEL_TRANSFER_MIN_PAIRS:
        return 0.0
    carried = [float(source) for source, _written in pairs]
    written = [float(value) for _source, value in pairs]
    mean_carried = sum(carried) / len(carried)
    mean_written = sum(written) / len(written)
    covariance = sum(
        (source - mean_carried) * (value - mean_written)
        for source, value in zip(carried, written, strict=True)
    )
    spread_carried = math.sqrt(sum((source - mean_carried) ** 2 for source in carried))
    spread_written = math.sqrt(sum((value - mean_written) ** 2 for value in written))
    if spread_carried == 0.0 or spread_written == 0.0:
        return 0.0
    return covariance / (spread_carried * spread_written)


def level_transfer_reading(correlation: float, n_pairs: int) -> str:
    """Whether the per-case level the fit carries across is a property of the case at all."""
    if n_pairs < LEVEL_TRANSFER_MIN_PAIRS:
        return LEVEL_TRANSFER_UNMEASURED
    return (
        LEVEL_TRANSFER_PRESENT
        if abs(correlation) >= LEVEL_TRANSFER_MIN_CORRELATION
        else LEVEL_TRANSFER_ABSENT
    )


def level_transfer_record(
    cells: list[dict[str, object]], source_cell_id: str, fitted_cell_id: str
) -> dict[str, object]:
    """The level-transfer pairing as the three fields a fit record carries it in.

    A near-zero correlation here is what turns the predicted count from twelve per-case predictions
    into one family rate with a width, so the pairing and the reading travel together with the
    interval rather than being recoverable only by re-pairing the rows.
    """
    pairs = paired_case_levels(cells, source_cell_id, fitted_cell_id)
    correlation = level_correlation(pairs)
    return {
        "level_transfer_pairs": len(pairs),
        "level_transfer_correlation": round(correlation, 3),
        "level_transfer_reading": level_transfer_reading(correlation, len(pairs)),
    }


def count_covers(interval: list[int], count: int) -> bool:
    """Whether a count a family measured falls inside the interval its fit predicted it in."""
    return bool(interval) and interval[0] <= count <= interval[1]


def family_level_transfer_reading(families: list[dict[str, object]]) -> tuple[str, str]:
    """Whether ANY qualified family carries a per-case level, stated with every correlation.

    One family that does would make the level worth keeping per case and the interval below it
    conservative; none that do is the run saying the level is a family constant and the spread is
    the irreducible width of every count read off it.
    """
    measured = [
        (cast(str, family["model_family"]), fit)
        for family in families
        if bool(family["control_eligible"])
        for fit in cast(list[dict[str, object]], family.get("guard_fits", []))
        if fit.get("level_transfer_reading") != LEVEL_TRANSFER_UNMEASURED
    ]
    if not measured:
        return (
            LEVEL_TRANSFER_UNREAD,
            "no qualified family paired a case's control fold length against its own fold at the "
            "fitted cell",
        )
    stated = "; ".join(
        f"{name} r={float(cast(float, fit['level_transfer_correlation'])):+.2f} over "
        f"{fit['level_transfer_pairs']} paired cases"
        for name, fit in measured
    )
    carried = [
        name for name, fit in measured if fit["level_transfer_reading"] == LEVEL_TRANSFER_PRESENT
    ]
    if carried:
        return LEVEL_PER_CASE, f"{carried} carry a per-case fold-length level: {stated}"
    return (
        LEVEL_CONSTANT_PER_FAMILY,
        f"the per-case level is family central tendency plus spread, not a case property: {stated}",
    )


def _first_fold_lengths(cells: list[dict[str, object]], cell_id: str) -> dict[str, int]:
    """The first summarizer output each case wrote in one cell's shipped-policy arm."""
    return {
        cast(str, case["item_id"]): int(cast(list[int], case["summary_output_chars"])[0])
        for row in cells
        if row["cell_id"] == cell_id and row["arm"] == ARM_TYPED_MARKER
        for case in cast(list[dict[str, object]], row["cases"])
        if cast(list[int], case.get("summary_output_chars", []))
    }
