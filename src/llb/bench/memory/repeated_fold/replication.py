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
from llb.bench.memory.repeated_fold.guard_fit import guard_resolver, measured_fold_lengths
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
    """Run one candidate control-first, then read its measured fold groups.

    Control-first is what makes the per-family guard fit possible at all: the control's own
    telemetry carries the fold length the later cells' guard is resolved from, so the fit costs
    no extra episode and reads the family that is actually about to run.
    """
    model = cast(str, candidate["model"])
    backend = cast(str, candidate["backend"])
    base = run_repeated_fold_completion(
        design,
        model=model,
        backend=backend,
        complete=complete,
        resolve_guard=guard_resolver(design, evidence_floor=minimum_paired_cases(design)),
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

    The fit is a model-free probe replayed at measurements taken on ANOTHER cell -- the one-fold
    control -- so it can be wrong even with the family's own step length carried across: the
    control folds one long span once, while the fitted cell folds several short ones, and a
    summarizer handed less transcript writes less. `fold_length_replay_error_chars` is that gap,
    measured after the fact, and it is what a per-guard count that misses is explained by.
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
    return {
        **fit,
        "measured_target_cases": measured_cases,
        "prediction_held": bool(measured) and measured_cases >= predicted,
        "prediction_error_cases": measured_cases - predicted,
        "prediction_exact": bool(measured) and measured_cases == predicted,
        "fitted_cell_fold_lengths": cell_folds,
        "median_fitted_cell_fold_length_chars": median_int(cell_folds),
        "fold_length_replay_error_chars": error,
        # WHY the prediction held or missed, rather than only that it did: the fit's slack against
        # the error the replay actually made. A prediction inside the margin is one the operator
        # can read on its own; one outside it happened to be right.
        "prediction_within_fold_length_margin": bool(cell_folds)
        and abs(error) <= int(cast(int, fit.get("fold_count_margin_chars", 0))),
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
