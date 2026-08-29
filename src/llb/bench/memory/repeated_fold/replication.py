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
from llb.bench.memory.repeated_fold.guard_fit import guard_resolver
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
            _fit_against_measurement(fit, rows)
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
    fit: dict[str, object], rows: list[dict[str, object]]
) -> dict[str, object]:
    """State the fitted guard's PREDICTION beside what the family then measured.

    The fit is a model-free probe replayed at a measured fold length, so it can be wrong: a family
    whose later folds write longer summaries than its first one lands somewhere else. Recording
    both makes that visible as a number rather than as a surprise in the fold table.
    """
    target = int(cast(int, fit["target_folds"]))
    measured = [
        int(cast(int, row["n_evidence"]))
        for row in rows
        if int(cast(int, row["measured_folds"])) == target
    ]
    return {
        **fit,
        "measured_target_cases": measured[0] if measured else 0,
        "prediction_held": bool(measured)
        and measured[0] >= int(cast(int, fit["predicted_target_cases"])),
    }


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
        **ladder_coverage(qualified),
        "replication_reading": reading,
        "replication_reason": reason,
        "shared_powered_fold_limit": min(limits) if limits else None,
        "mechanism_readings": {
            cast(str, row["model_family"]): row["mechanism_reading"] for row in families
        },
        "changes_shipped_default": False,
    }
