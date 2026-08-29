"""Run the transcript-fitting control and unavoidable window-elision arm.

The trim strategy is pinned to `head_tail` here rather than taken from the shipped default. This
study PRICES what the whole-transcript trim loses in the middle of a folded transcript, and its
conditional prototype arm is the entry-aware trim measured against it; inheriting the default would
make the two arms the same trim the moment `per_entry_head` shipped as the default, and the study
would report no loss because it had stopped running the arm the loss belongs to.
"""

from dataclasses import dataclass
from typing import cast

from llb.backends.context_budget import fixed_budget
from llb.bench.agentic.context_policy import (
    POLICY_COMPACT,
    SUMMARY_TRIM_HEAD_TAIL,
    ContextPolicy,
)
from llb.bench.agentic.model import AgenticTask, STATUS_CONTEXT_OVERFLOW
from llb.bench.context_policy.run import run_policy, task_set_digest
from llb.bench.context_policy.report import PolicyReport
from llb.bench.memory.transcript import build_memory_dependent_tasks
from llb.bench.memory.window_elision.design import (
    ROLE_ELIDED,
    ROLE_FIT,
    cell_geometry,
    elision_cells,
    probe_elision_cell,
)
from llb.bench.memory.window_elision.reading import (
    completion_reading,
    operator_recommendation,
)
from llb.bench.common import LLMComplete


@dataclass(slots=True)
class WindowElisionRun:
    """Aggregate analysis and the reports persisted for each arm."""

    analysis: dict[str, object]
    reports: dict[str, PolicyReport]


def run_window_elision(
    design: dict[str, object],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
) -> WindowElisionRun:
    """Run the fit control first, then its trigger-matched elided arm."""
    held = cast(dict[str, object], design["held_fixed"])
    tasks = [
        AgenticTask.from_record(row)
        for row in build_memory_dependent_tasks(
            n_tasks=int(cast(int, held["n_tasks"])),
            depth=int(cast(int, held["depth"])),
            pad_chars=int(cast(int, held["pad_chars"])),
        )
    ]
    return run_window_elision_tasks(
        design,
        tasks=tasks,
        model=model,
        backend=backend,
        complete=complete,
    )


def run_window_elision_tasks(
    design: dict[str, object],
    *,
    tasks: list[AgenticTask],
    model: str,
    backend: str,
    complete: LLMComplete,
    case_metadata: dict[str, dict[str, object]] | None = None,
    cell_probes: dict[str, dict[str, object]] | None = None,
    summary_trim_strategy: str = SUMMARY_TRIM_HEAD_TAIL,
) -> WindowElisionRun:
    """Run any prevalidated task set through the shared fitting/elided cell pair."""
    held = cast(dict[str, object], design["held_fixed"])
    digest = task_set_digest(tasks)
    reports: dict[str, PolicyReport] = {}
    rows: list[dict[str, object]] = []
    for cell in elision_cells(design):
        cell_id = cast(str, cell["cell_id"])
        probe = cell_probes[cell_id] if cell_probes is not None else probe_elision_cell(cell, held)
        report, row = run_window_elision_cell(
            tasks,
            cell,
            held,
            model=model,
            backend=backend,
            complete=complete,
            probe=probe,
            task_digest=digest,
            case_metadata=case_metadata,
            summary_trim_strategy=summary_trim_strategy,
        )
        reports[cell_id] = report
        rows.append(row)
        if cell["role"] == ROLE_FIT and not _row_valid(row, held)[0]:
            break
    by_role = {cast(str, row["role"]): row for row in rows}
    fit = by_role.get(ROLE_FIT)
    elided = by_role.get(ROLE_ELIDED)
    eligible, reason = _comparison_eligibility(fit, elided, held)
    reading, reading_reason, paired = completion_reading(
        cast(list[dict[str, object]], fit["cases"]) if fit else [],
        cast(list[dict[str, object]], elided["cases"]) if elided else [],
        eligible=eligible,
        eligibility_reason=reason,
    )
    return WindowElisionRun(
        analysis={
            "study_id": design["study_id"],
            "study_kind": design["study_kind"],
            "model": model,
            "backend": backend,
            "seed": design["seed"],
            "task_set_digest": digest,
            "held_fixed": held,
            "summary_trim_strategy": summary_trim_strategy,
            "cells": rows,
            "comparison_eligible": eligible,
            "eligibility_reason": reason,
            "paired_completion": paired,
            "completion_reading": reading,
            "completion_reason": reading_reason,
            "operator_recommendation": operator_recommendation(reading),
            "changes_shipped_default": False,
        },
        reports=reports,
    )


def run_window_elision_cell(
    tasks: list[AgenticTask],
    cell: dict[str, object],
    held: dict[str, object],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    probe: dict[str, object],
    task_digest: str,
    case_metadata: dict[str, dict[str, object]] | None = None,
    summary_trim_strategy: str = SUMMARY_TRIM_HEAD_TAIL,
) -> tuple[PolicyReport, dict[str, object]]:
    """Run one prevalidated geometry cell through the common compact-policy contract."""
    geometry = cell_geometry(cell, held)
    report = run_policy(
        tasks,
        ContextPolicy(
            name=POLICY_COMPACT,
            observation_cap_chars=int(cast(int, geometry["observation_cap_chars"])),
            observation_head_share=float(cast(float, geometry["observation_head_share"])),
            compact_share=float(cast(float, geometry["compact_share"])),
            summary_input_cap=str(cast(str, geometry["summary_input_cap"])),
            summary_trim_strategy=summary_trim_strategy,
        ),
        model=model,
        backend=backend,
        complete=complete,
        max_steps=int(cast(int, geometry["depth"])) + int(cast(int, geometry["max_steps_margin"])),
        budget=fixed_budget(int(cast(int, geometry["max_prompt_chars"]))),
        preserve_memory_markers=bool(held["preserve_memory_markers"]),
    )
    return report, _cell_row(
        cell,
        report,
        probe,
        task_digest,
        case_metadata=case_metadata,
    )


def _cell_row(
    cell: dict[str, object],
    report: PolicyReport,
    probe: dict[str, object],
    digest: str,
    *,
    case_metadata: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    metadata = case_metadata if case_metadata is not None else {}
    cases = [
        {
            "item_id": row["item_id"],
            "success": bool(episode.success),
            "status": episode.status,
            "measured_folds": episode.telemetry.n_compactions,
            "summary_input_chars": episode.telemetry.summary_input_chars,
            "summary_input_elided_chars": episode.telemetry.summary_input_elided_chars,
            "trimmed_summary_inputs": episode.telemetry.n_trimmed_summary_inputs,
            "compaction_prompt_chars": episode.telemetry.compaction_prompt_chars,
            "n_steps": episode.n_steps,
            **metadata.get(cast(str, row["item_id"]), {}),
        }
        for row, episode in zip(report.rows, report.episodes, strict=True)
    ]
    return {
        "cell_id": cell["cell_id"],
        "role": cell["role"],
        "max_prompt_chars": cell["max_prompt_chars"],
        "compact_share": cell["compact_share"],
        **probe,
        "task_set_digest": digest,
        "n_tasks": len(cases),
        "completion": report.result.objective_score,
        "cases": cases,
    }


def _row_valid(row: dict[str, object], held: dict[str, object]) -> tuple[bool, str]:
    cases = cast(list[dict[str, object]], row["cases"])
    expected_input = int(cast(int, row["summary_input_chars"]))
    expected_elision = int(cast(int, row["summary_input_elided_chars"]))
    expected_folds = int(cast(int, row["n_compactions"]))
    valid = all(
        case["status"] != STATUS_CONTEXT_OVERFLOW
        and int(cast(int, case["measured_folds"])) == expected_folds
        and int(cast(int, case["summary_input_chars"])) == expected_input
        and int(cast(int, case["summary_input_elided_chars"])) == expected_elision
        for case in cases
    )
    if row["role"] == ROLE_FIT:
        valid = valid and float(cast(float, row["completion"])) >= float(
            cast(float, held["minimum_control_completion"])
        )
    return valid, (
        f"cell={row['cell_id']} completion={float(cast(float, row['completion'])):.3f}; "
        f"folds={[case['measured_folds'] for case in cases]}; "
        f"input={[case['summary_input_chars'] for case in cases]}; "
        f"elided={[case['summary_input_elided_chars'] for case in cases]}"
    )


def _comparison_eligibility(
    fit: dict[str, object] | None,
    elided: dict[str, object] | None,
    held: dict[str, object],
) -> tuple[bool, str]:
    if fit is None or elided is None:
        return False, "both the transcript-fitting control and elided arm must run"
    fit_valid, fit_reason = _row_valid(fit, held)
    elided_valid, elided_reason = _row_valid(elided, held)
    same_digest = fit["task_set_digest"] == elided["task_set_digest"]
    return (
        fit_valid and elided_valid and same_digest,
        f"fit: {fit_reason}; elided: {elided_reason}; same task digest={same_digest}",
    )
