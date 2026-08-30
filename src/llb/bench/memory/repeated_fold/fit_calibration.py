"""What a per-family guard fit turned out to be worth, once the cell it fitted has actually run.

`guard_fit` chooses a guard before the fitted cell runs, from measurements taken on other cells.
Everything here happens AFTER: the fit's prediction stated beside what the family then measured,
the errors that explain the gap, and the cross-family verdicts an operator needs before reading a
predicted rung without a confirming run behind it -- whether every fitted guard predicted the count
its family measured, whether the span correction the replay applies points the same way on every
family that ran one, and whether the per-case level it carried across turned out to be a property
of the case at all (`llb.bench.memory.repeated_fold.level_transfer` prices that last one; this is
where the pairing it reads is taken).

The readings are kept apart from the run loop deliberately: they are pure functions of persisted
rows, so a re-read of a committed bundle answers them without a GPU.
"""

from typing import cast

from llb.bench.context_policy.guard_band import median_int
from llb.bench.memory.repeated_fold.fold_span import (
    SPAN_LENGTH_INTERPOLATED,
    FoldLengthModel,
    measured_fold_points,
)
from llb.bench.memory.repeated_fold.guard_fit import measured_fold_lengths
from llb.bench.memory.repeated_fold.level_transfer import count_covers, level_transfer_record

# What the fitted guard's PREDICTED case count turned out to be worth, across every family that
# ran one. The fit's job is to rank candidate guards; whether its absolute per-guard count is a
# number an operator can read on its own is a separate question, and this is where the run answers
# it rather than leaving it to the next reader of the fold table.
PREDICTION_CALIBRATED = "every_fitted_guard_predicted_the_case_count_its_family_measured"
PREDICTION_DIVERGED = "a_fitted_guard_predicted_a_case_count_its_family_did_not_measure"
PREDICTION_UNREAD = "no_qualified_family_ran_a_fitted_guard"

# Whether the span correction is a RATE of this task or a property of one family. The correction
# says a summarizer offered less writes less, which is a claim about summarizers in general; if the
# families that ran disagree on its SIGN, then what each family measured is its own habit and the
# correction cannot be carried to a family that has not run. This is where the run says which,
# rather than leaving a reader to compare two slopes in a table.
SPAN_SLOPE_AGREES = "every_qualified_family_writes_shorter_summaries_when_offered_less"
SPAN_SLOPE_DISAGREES = "the_qualified_families_disagree_on_the_sign_of_the_span_correction"
SPAN_SLOPE_UNREAD = "no_qualified_family_measured_a_second_fold_span"


def fit_against_measurement(
    fit: dict[str, object], rows: list[dict[str, object]], cells: list[dict[str, object]]
) -> dict[str, object]:
    """State the fitted guard's PREDICTION beside what the family then measured.

    Three things the fitted numbers alone would hide travel with it: the replay error that explains
    a miss (`_replay_error_fields`), whether the per-case level the replay carried across is a
    property of the case at all, and whether the count the family measured falls inside the
    interval the fit stated it in. The last two are the same finding read two ways -- a level that
    does not transfer is why the count has a width -- so they are recorded together.
    """
    target = int(cast(int, fit["target_folds"]))
    predicted = int(cast(int, fit["predicted_target_cases"]))
    measured = [
        int(cast(int, row["n_evidence"]))
        for row in rows
        if int(cast(int, row["measured_folds"])) == target
    ]
    measured_cases = measured[0] if measured else 0
    interval = cast(list[int], fit.get("predicted_target_cases_interval") or [])
    return {
        **fit,
        "measured_target_cases": measured_cases,
        **level_transfer_record(
            cells, cast(str, fit["fold_length_source"]), cast(str, fit["cell_id"])
        ),
        **_replay_error_fields(fit, cells),
        "measured_within_predicted_interval": bool(measured)
        and count_covers(interval, measured_cases),
        "prediction_held": bool(measured) and measured_cases >= predicted,
        "prediction_error_cases": measured_cases - predicted,
        "prediction_exact": bool(measured) and measured_cases == predicted,
    }


def _replay_error_fields(
    fit: dict[str, object], cells: list[dict[str, object]]
) -> dict[str, object]:
    """What the replayed fold length was wrong by, both ways, and whether that fits the margin.

    Two errors rather than one, because they answer different questions:
    `fold_length_replay_error_chars` is what the control's raw length was wrong by, which is what a
    flat replay stood on; `span_replay_error_chars` is what the span-aware length was wrong by at
    the span this cell's own folds turned out to offer, which is what the fit actually used. The
    second is the one a per-guard count that misses is explained by.
    """
    cell_folds = measured_fold_lengths(cells, cast(str, fit["cell_id"]))
    replayed = int(cast(int, fit["median_fold_length_chars"]))
    points = measured_fold_points(cells, cast(str, fit["cell_id"]))
    spans = FoldLengthModel.from_record(fit)
    # Matched PER FOLD, because the fitted cell does not fold one span: a two-fold episode folds a
    # short first span and a longer second one, and a single median over both lands on whichever
    # cluster holds the middle value. Each fold is scored against the length the model replays at
    # that fold's OWN span, and the median of those errors is what the replay was wrong by.
    span_errors = [chars - spans.length_at(replayed, span) for span, chars in points]
    return {
        "fitted_cell_fold_lengths": cell_folds,
        "median_fitted_cell_fold_length_chars": median_int(cell_folds),
        "fitted_cell_fold_span_range": (
            [min(span for span, _chars in points), max(span for span, _chars in points)]
            if points
            else []
        ),
        "span_replay_error_chars": median_int(span_errors) if span_errors else 0,
        "fold_length_replay_error_chars": median_int(cell_folds) - replayed if cell_folds else 0,
        # WHY the prediction held or missed, rather than only that it did: the fit's slack against
        # the error the replay actually made. A prediction inside the margin is one the operator
        # can read on its own; one outside it happened to be right.
        "prediction_within_fold_length_margin": bool(span_errors)
        and abs(median_int(span_errors)) <= int(cast(int, fit.get("fold_count_margin_chars", 0))),
    }


def fit_prediction_reading(families: list[dict[str, object]]) -> tuple[str, str]:
    """Whether every fitted guard's predicted case count is one the run then measured.

    A divergence is named -- family, cell, guard, both counts and the replay error behind them --
    rather than folded into a pass, because the whole point of calibrating the probe is that an
    operator can read a predicted rung without a confirming run standing behind it.
    """
    fits = [
        (cast(str, family["model_family"]), fit)
        for family in families
        if bool(family["control_eligible"])
        for fit in cast(list[dict[str, object]], family.get("guard_fits", []))
    ]
    if not fits:
        return PREDICTION_UNREAD, "no qualified family resolved a fitted guard"
    missed = [
        f"{name}/{fit['cell_id']} at guard {fit['fitted_max_prompt_chars']}: predicted "
        f"{fit['predicted_target_cases']} of {fit['target_folds']}-fold cases, measured "
        f"{fit['measured_target_cases']} (fold length replayed at "
        f"{fit['median_fold_length_chars']}, cell measured "
        f"{fit['median_fitted_cell_fold_length_chars']})"
        for name, fit in fits
        if not bool(fit["prediction_exact"])
    ]
    if missed:
        return PREDICTION_DIVERGED, "; ".join(missed)
    return (
        PREDICTION_CALIBRATED,
        "; ".join(
            f"{name}/{fit['cell_id']} at guard {fit['fitted_max_prompt_chars']}: predicted and "
            f"measured {fit['measured_target_cases']} of {fit['target_folds']}-fold cases"
            for name, fit in fits
        ),
    )


def span_slope_reading(families: list[dict[str, object]]) -> tuple[str, str]:
    """Whether every qualified family's span correction points the same way.

    A slope measured on one family is only a CALIBRATION if the effect it corrects for is real;
    two families whose slopes have opposite signs have measured two habits, and the correction is
    then per-family bookkeeping rather than a rate an unrun family inherits. Reported either way,
    with each family's slope, because a disagreement is the finding and not a failure.
    """
    slopes = [
        (
            cast(str, family["model_family"]),
            float(cast(float, fit["chars_written_per_offered_char"])),
        )
        for family in families
        if bool(family["control_eligible"])
        for fit in cast(list[dict[str, object]], family.get("guard_fits", []))
        if fit.get("fold_span_reading") == SPAN_LENGTH_INTERPOLATED
    ]
    if not slopes:
        return SPAN_SLOPE_UNREAD, "no qualified family measured a second fold span to slope against"
    stated = "; ".join(
        f"{name} {slope:+.5f} written chars per offered char" for name, slope in slopes
    )
    if len({slope > 0 for _name, slope in slopes}) > 1:
        return SPAN_SLOPE_DISAGREES, stated
    return SPAN_SLOPE_AGREES, stated
