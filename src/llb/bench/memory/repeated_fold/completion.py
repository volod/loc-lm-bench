"""Compact-only completion evidence across one, two, and three measured folds."""

from dataclasses import dataclass
from typing import Callable, cast

from llb.backends.context_budget import fixed_budget
from llb.bench.agentic.context_policy import POLICY_COMPACT, ContextPolicy
from llb.bench.agentic.model import AgenticTask
from llb.bench.context_policy.run import run_policy, task_set_digest
from llb.bench.context_policy.report import PolicyReport
from llb.bench.memory.repeated_fold.design import (
    MECHANISM_ARMS,
    cell_geometry,
    completion_cells,
    probe_completion_cell,
)
from llb.bench.memory.repeated_fold.reading import (
    READING_INSUFFICIENT,
    completion_by_fold,
    completion_cost_reading,
    mechanism_reading,
)
from llb.bench.memory.transcript import build_memory_dependent_tasks
from llb.bench.common import LLMComplete


# Given one declared cell and the rows measured before it, the cell as it will actually run plus
# the record explaining any guard the run moved. A study that holds every guard as declared passes
# no resolver at all; the two-family replication passes one that fits the middle rung per family.
GuardResolver = Callable[
    [dict[str, object], list[dict[str, object]]],
    tuple[dict[str, object], dict[str, object] | None],
]


@dataclass(slots=True)
class RepeatedFoldRun:
    """Analysis plus the reports needed to persist each compact-only arm."""

    analysis: dict[str, object]
    reports: dict[tuple[str, str], PolicyReport]


def run_repeated_fold_completion(
    design: dict[str, object],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    resolve_guard: GuardResolver | None = None,
) -> RepeatedFoldRun:
    """Run every cell under typed-marker and model-summary-only compact arms.

    `resolve_guard` is the seam a per-family guard fit enters through. It is consulted ONCE per
    cell, before the cell's first arm runs, so both mechanism arms of a cell share one geometry and
    the marker ablation stays a comparison of the marker alone.
    """
    held = cast(dict[str, object], design["held_fixed"])
    tasks = [
        AgenticTask.from_record(row)
        for row in build_memory_dependent_tasks(
            n_tasks=int(cast(int, held["n_tasks"])),
            depth=int(cast(int, completion_cells(design)[0]["depth"])),
            pad_chars=int(cast(int, held["pad_chars"])),
        )
    ]
    digest = task_set_digest(tasks)
    reports: dict[tuple[str, str], PolicyReport] = {}
    rows: list[dict[str, object]] = []
    control_eligible = False
    control_reason = "the one-fold control was not run"
    work = [(cell, arm) for cell in completion_cells(design) for arm in MECHANISM_ARMS]
    resolved: dict[str, tuple[dict[str, object], dict[str, object] | None]] = {}
    for index, (cell, arm) in enumerate(work):
        cell_id = cast(str, cell["cell_id"])
        if cell_id not in resolved:
            resolved[cell_id] = resolve_guard(cell, rows) if resolve_guard else (cell, None)
        effective, guard_fit = resolved[cell_id]
        geometry = cell_geometry(effective, held)
        probe = probe_completion_cell(effective, held)
        report = run_policy(
            tasks,
            ContextPolicy(
                name=POLICY_COMPACT,
                observation_cap_chars=int(cast(int, geometry["observation_cap_chars"])),
                observation_head_share=float(cast(float, geometry["observation_head_share"])),
                compact_share=float(cast(float, geometry["compact_share"])),
                summary_input_cap=str(cast(str, geometry["summary_input_cap"])),
            ),
            model=model,
            backend=backend,
            complete=complete,
            max_steps=int(cast(int, geometry["depth"]))
            + int(cast(int, geometry["max_steps_margin"])),
            budget=fixed_budget(int(cast(int, geometry["max_prompt_chars"]))),
            preserve_memory_markers=arm == "typed_marker",
        )
        reports[(cell_id, arm)] = report
        rows.append(_cell_row(cell, effective, arm, report, probe, digest, guard_fit))
        if index == 0:
            control_eligible, control_reason = _control_eligibility(rows[0], held)
            if not control_eligible:
                break
    measured = completion_by_fold(rows)
    reading, reason, fold_limit = completion_cost_reading(measured)
    if not control_eligible:
        reading, reason, fold_limit = READING_INSUFFICIENT, control_reason, None
    mechanism, mechanism_reason = mechanism_reading(rows)
    return RepeatedFoldRun(
        analysis={
            "study_id": design["study_id"],
            "study_kind": design["study_kind"],
            "model": model,
            "backend": backend,
            "seed": design["seed"],
            "task_set_digest": digest,
            "held_fixed": held,
            "control_eligible": control_eligible,
            "control_reason": control_reason,
            "guard_fits": [fit for _cell, fit in resolved.values() if fit is not None],
            "cells": rows,
            "completion_by_measured_fold_count": measured,
            "completion_reading": reading,
            "completion_reason": reason,
            "recommended_fold_count_limit": fold_limit,
            "mechanism_reading": mechanism,
            "mechanism_reason": mechanism_reason,
            "changes_shipped_default": False,
        },
        reports=reports,
    )


def _control_eligibility(row: dict[str, object], held: dict[str, object]) -> tuple[bool, str]:
    """Require the cap-fitting control to complete and actually enter its declared regime."""
    completion = float(cast(float, row["completion"]))
    minimum = float(cast(float, held["minimum_control_completion"]))
    folds = cast(list[int], row["measured_fold_counts"])
    expected = int(cast(int, row["expected_oracle_folds"]))
    eligible = completion >= minimum and all(fold == expected for fold in folds)
    return (
        eligible,
        f"one-fold control completion={completion:.3f} (required {minimum:.3f}); "
        f"measured folds={folds} (required {expected} per case)",
    )


def _cell_row(
    cell: dict[str, object],
    effective: dict[str, object],
    arm: str,
    report: PolicyReport,
    probe: dict[str, object],
    digest: str,
    guard_fit: dict[str, object] | None,
) -> dict[str, object]:
    cases = [
        {
            "item_id": row["item_id"],
            "success": bool(episode.success),
            "status": episode.status,
            "measured_folds": episode.telemetry.n_compactions,
            "n_steps": episode.n_steps,
            # One entry per fold, so the control's rows carry the fold length a later cell's
            # guard is fitted against without a second run.
            "summary_output_chars": list(episode.telemetry.summary_output_chars),
            # One entry per step, for the other half of the same fit: how fast this family's own
            # calls grow the transcript the fold count is counted on.
            "step_entry_chars": list(episode.telemetry.step_entry_chars),
        }
        for row, episode in zip(report.rows, report.episodes, strict=True)
    ]
    return {
        "cell_id": cell["cell_id"],
        "arm": arm,
        "preserve_memory_markers": arm == "typed_marker",
        "depth": cell["depth"],
        "max_prompt_chars": effective["max_prompt_chars"],
        "declared_max_prompt_chars": cell["max_prompt_chars"],
        "guard_fit": guard_fit,
        "cap_fitting_control": cell["cap_fitting_control"],
        "expected_oracle_folds": cell["expected_oracle_folds"],
        **probe,
        "task_set_digest": digest,
        "n_tasks": len(cases),
        "completion": report.result.objective_score,
        "measured_fold_counts": [case["measured_folds"] for case in cases],
        "case_success": [case["success"] for case in cases],
        "cases": cases,
    }
