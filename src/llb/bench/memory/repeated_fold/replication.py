"""Two-family execution and cross-family analysis for repeated-fold completion.

Each family runs the committed compact-only cells through the SAME runner the single-family
completion study uses, so the cases, the seed, the marker ablation, and the one-fold eligibility
gate are identical by construction rather than by restatement. What this module adds is the layer
above one family: the per-fold paired uncertainty each family's rows imply, and the rule that a
fold count is only claimed as far as every qualified family carries it.
"""

from dataclasses import dataclass, field
from typing import cast

from llb.bench.memory.repeated_fold.completion import (
    RepeatedFoldRun,
    run_repeated_fold_completion,
)
from llb.bench.memory.repeated_fold.fold_span import (
    SPAN_LENGTH_INTERPOLATED,
    FoldLengthModel,
    measured_fold_points,
)
from llb.bench.memory.repeated_fold.guard_fit import (
    fitted_cell_order,
    guard_resolver,
    measured_fold_lengths,
)
from llb.bench.memory.repeated_fold.replication_design import (
    minimum_paired_cases,
    replication_roster,
    roster_digest,
)
from llb.bench.memory.repeated_fold.ladder_coverage import ladder_coverage
from llb.bench.memory.repeated_fold.replication_reading import (
    fold_group_rows,
    powered_fold_limit,
    replication_reading,
)
from llb.bench.common import LLMComplete
from llb.bench.context_policy.guard_band import median_int

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


@dataclass(slots=True)
class ReplicationFamilyRun:
    """One family's compact-only cells plus the fold reading they imply."""

    model_family: str
    model: str
    backend: str
    base: RepeatedFoldRun
    analysis: dict[str, object] = field(default_factory=dict)
    tokens_per_s: float = 0.0


def run_replication_family(
    design: dict[str, object],
    candidate: dict[str, object],
    *,
    complete: LLMComplete,
) -> ReplicationFamilyRun:
    """Run one candidate control-first and fitted-cell-last, then read its measured fold groups.

    The order is what makes the per-family guard fit possible at all: every cell the fit measures
    against runs before the cell it fits, so the fit costs no extra episode and reads the family
    that is actually about to run. The control leads because it is the eligibility gate; the
    fitted cell trails because the never-fitted cell ahead of it is where the replay gets the
    second fold span its length is interpolated across.
    """
    model = cast(str, candidate["model"])
    backend = cast(str, candidate["backend"])
    base = run_repeated_fold_completion(
        design,
        model=model,
        backend=backend,
        complete=complete,
        resolve_guard=guard_resolver(design, evidence_floor=minimum_paired_cases(design)),
        cell_order=fitted_cell_order(design),
    )
    return ReplicationFamilyRun(
        model_family=cast(str, candidate["model_family"]),
        model=model,
        backend=backend,
        base=base,
        analysis=family_fold_analysis(design, base.analysis, candidate),
    )


def family_fold_analysis(
    design: dict[str, object],
    analysis: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    """Attach per-fold paired uncertainty and the powered fold limit to one family."""
    floor = minimum_paired_cases(design)
    eligible = bool(analysis["control_eligible"])
    rows = fold_group_rows(cast(list[dict[str, object]], analysis["cells"]), evidence_floor=floor)
    limit, reason = (
        powered_fold_limit(rows) if eligible else (None, cast(str, analysis["control_reason"]))
    )
    return {
        "model_family": candidate["model_family"],
        "model": analysis["model"],
        "backend": analysis["backend"],
        "task_set_digest": analysis["task_set_digest"],
        "control_eligible": eligible,
        "control_reason": analysis["control_reason"],
        "evidence_floor": floor,
        "guard_fits": [
            _fit_against_measurement(fit, rows, cast(list[dict[str, object]], analysis["cells"]))
            for fit in cast(list[dict[str, object]], analysis["guard_fits"])
        ],
        "fold_groups": rows,
        "powered_fold_limit": limit,
        "powered_fold_reason": reason,
        "fold_count_lost_a_paired_case": eligible and bool(_paired_losses(rows, powered=True)),
        "underpowered_paired_losses": _paired_losses(rows, powered=False),
        "completion_reading": analysis["completion_reading"],
        "completion_reason": analysis["completion_reason"],
        "mechanism_reading": analysis["mechanism_reading"],
        "mechanism_reason": analysis["mechanism_reason"],
        "cells": analysis["cells"],
    }


def _fit_against_measurement(
    fit: dict[str, object], rows: list[dict[str, object]], cells: list[dict[str, object]]
) -> dict[str, object]:
    """State the fitted guard's PREDICTION beside what the family then measured, and the error.

    The fit is a model-free probe replayed at measurements taken on OTHER cells, so it can be
    wrong even with the family's own step length and span slope carried across. Two errors are
    reported rather than one, because they answer different questions:
    `fold_length_replay_error_chars` is what the control's raw length was wrong by, which is what
    a flat replay stood on; `span_replay_error_chars` is what the span-aware length was wrong by
    at the span this cell's own folds turned out to offer, which is what the fit actually used.
    The second is the one a per-guard count that misses is explained by.
    """
    target = int(cast(int, fit["target_folds"]))
    predicted = int(cast(int, fit["predicted_target_cases"]))
    measured = [
        int(cast(int, row["n_evidence"]))
        for row in rows
        if int(cast(int, row["measured_folds"])) == target
    ]
    measured_cases = measured[0] if measured else 0
    cell_folds = measured_fold_lengths(cells, cast(str, fit["cell_id"]))
    replayed = int(cast(int, fit["median_fold_length_chars"]))
    error = median_int(cell_folds) - replayed if cell_folds else 0
    points = measured_fold_points(cells, cast(str, fit["cell_id"]))
    spans = FoldLengthModel.from_record(fit)
    # Matched PER FOLD, because the fitted cell does not fold one span: a two-fold episode folds a
    # short first span and a longer second one, and a single median over both lands on whichever
    # cluster holds the middle value. Each fold is scored against the length the model replays at
    # that fold's OWN span, and the median of those errors is what the replay was wrong by.
    span_errors = [chars - spans.length_at(replayed, span) for span, chars in points]
    return {
        **fit,
        "measured_target_cases": measured_cases,
        "prediction_held": bool(measured) and measured_cases >= predicted,
        "prediction_error_cases": measured_cases - predicted,
        "prediction_exact": bool(measured) and measured_cases == predicted,
        "fitted_cell_fold_lengths": cell_folds,
        "median_fitted_cell_fold_length_chars": median_int(cell_folds),
        "fitted_cell_fold_span_range": (
            [min(span for span, _chars in points), max(span for span, _chars in points)]
            if points
            else []
        ),
        "span_replay_error_chars": median_int(span_errors) if span_errors else 0,
        "fold_length_replay_error_chars": error,
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


def _paired_losses(rows: list[dict[str, object]], *, powered: bool) -> list[int]:
    """Measured fold counts where a task completes at one fold and fails at that count."""
    return [
        int(cast(int, row["measured_folds"]))
        for row in rows
        if bool(row["meets_evidence_floor"]) is powered
        and int(cast(int, cast(dict[str, object], row["paired"])["control_wins"])) > 0
    ]


def analyze_replication_runs(
    design: dict[str, object], runs: list[ReplicationFamilyRun]
) -> dict[str, object]:
    """Read the fold-count rule across every family the roster actually drove."""
    required = int(cast(int, design["required_qualified_families"]))
    families = [run.analysis for run in runs]
    reading, reason, qualified = replication_reading(families, required_families=required)
    digests = sorted({cast(str, row["task_set_digest"]) for row in families})
    limits = [
        int(cast(int, row["powered_fold_limit"]))
        for row in qualified
        if row["powered_fold_limit"] is not None
    ]
    return {
        "study_id": design["study_id"],
        "study_kind": design["study_kind"],
        "seed": design["seed"],
        "required_qualified_families": required,
        "family_digest": roster_digest(
            [
                {"model_family": run.model_family, "model": run.model, "backend": run.backend}
                for run in runs
            ]
        ),
        "roster_digest": roster_digest(replication_roster(design)),
        "task_set_digest": digests[0] if len(digests) == 1 else None,
        "task_set_digests": digests,
        "evidence_floor": minimum_paired_cases(design),
        "families": families,
        "qualified_models": [row["model"] for row in qualified],
        **_fit_prediction(families),
        **_span_slope(families),
        **ladder_coverage(qualified),
        "replication_reading": reading,
        "replication_reason": reason,
        "shared_powered_fold_limit": min(limits) if limits else None,
        "mechanism_readings": {
            cast(str, row["model_family"]): row["mechanism_reading"] for row in families
        },
        "changes_shipped_default": False,
    }


def _fit_prediction(families: list[dict[str, object]]) -> dict[str, object]:
    """The cross-family calibration verdict, as the two fields the report and CI read."""
    reading, reason = fit_prediction_reading(families)
    return {"fit_prediction_reading": reading, "fit_prediction_reason": reason}


def _span_slope(families: list[dict[str, object]]) -> dict[str, object]:
    """Whether the span correction generalizes across the families that ran it."""
    reading, reason = span_slope_reading(families)
    return {"span_slope_reading": reading, "span_slope_reason": reason}
