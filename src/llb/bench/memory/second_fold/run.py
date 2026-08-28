"""Does the trigger-only routing rule survive a SECOND fold?

The trigger collapse showed that `compact_share` and the prompt guard reach the transcript only
through their product, so a crossover measured at one share converts to any other. Every cell it
measured was cap-fitting, and a cap-fitting cell folds exactly once: trigger hysteresis raises the
trigger to the FULL guard after the first summary, and no cap-fitting transcript grows back that
far. The rule is therefore established only where the guard has nothing left to do.

This study runs the same question one regime over, on the committed repeatedly folding geometry.
There is no cap arm there -- the guards sit below the cap peak, where `observation_cap` overflows --
so the comparison is compact against compact: one equal-trigger family whose members differ only in
the guard, each paired against the family anchor on total model-input tokens. The per-fold
summarize input travels beside the cost, because the shipped `window` bound sizes that input from
the budget rather than from the trigger, which is the second way the guard can re-enter.
"""

from typing import cast

from llb.backends.context_budget import fixed_budget
from llb.bench.agentic.context_policy import POLICY_COMPACT, ContextPolicy
from llb.bench.agentic.design_fields import as_float, as_int, as_mapping, as_str
from llb.bench.agentic.model import AgenticTask
from llb.bench.common import LLMComplete, mean
from llb.bench.context_policy.report import (
    METRIC_COMPACTION_PROMPT_TOKENS,
    METRIC_TOTAL_MODEL_INPUT_TOKENS,
    PolicyReport,
)
from llb.bench.context_policy.run import run_policy, task_set_digest
from llb.bench.memory.second_fold.geometry import (
    cell_geometry,
    probe_second_fold_cell,
    second_fold_cap_peaks,
    second_fold_cells,
)
from llb.bench.memory.second_fold.reading import (
    REPORTING_CONFIDENCE,
    family_rows,
    repeat_geometry_row,
    second_fold_reading,
)
from llb.bench.memory.transcript import build_memory_dependent_tasks


def run_second_fold_cells(
    design: dict[str, object], *, model: str, backend: str, complete: LLMComplete
) -> tuple[list[dict[str, object]], dict[str, PolicyReport]]:
    """Run every declared cell's compact arm in declared order over one shared task set."""
    held = as_mapping(design, "held_fixed")
    cells = second_fold_cells(design)
    tasks = [
        AgenticTask.from_record(record)
        for record in build_memory_dependent_tasks(
            n_tasks=as_int(held, "n_tasks"),
            depth=as_int(cells[0], "depth"),
            pad_chars=as_int(held, "pad_chars"),
        )
    ]
    digest = task_set_digest(tasks)
    rows: list[dict[str, object]] = []
    reports: dict[str, PolicyReport] = {}
    for cell in cells:
        geometry = cell_geometry(cell, held)
        report = run_policy(
            tasks,
            ContextPolicy(
                name=POLICY_COMPACT,
                observation_cap_chars=as_int(geometry, "observation_cap_chars"),
                observation_head_share=as_float(geometry, "observation_head_share"),
                compact_share=as_float(geometry, "compact_share"),
                summary_input_cap=as_str(geometry, "summary_input_cap"),
            ),
            model=model,
            backend=backend,
            complete=complete,
            max_steps=as_int(geometry, "depth") + as_int(geometry, "max_steps_margin"),
            budget=fixed_budget(as_int(geometry, "max_prompt_chars")),
        )
        reports[as_str(cell, "cell_id")] = report
        rows.append(_cell_row(cell, held, report, digest))
    return rows, reports


def analyze_second_fold(
    design: dict[str, object],
    control_row: dict[str, object] | None,
    cell_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Gate the cells, pair every member against its anchor, and decide whether the rule holds."""
    held = as_mapping(design, "held_fixed")
    declared = second_fold_cells(design)
    eligible = bool(control_row is not None and control_row["eligible"])
    cells: list[dict[str, object]] = []
    families: list[dict[str, object]] = []
    repeat: dict[str, object] | None = None
    if eligible:
        if [row["cell_id"] for row in cell_rows] != [cell["cell_id"] for cell in declared]:
            raise ValueError("second-fold rows do not match the exact declared cell order")
        cells = cell_rows
        families = family_rows(design, cells, REPORTING_CONFIDENCE)
        repeat = repeat_geometry_row(cells, families, REPORTING_CONFIDENCE)
    reading, reason = second_fold_reading(design, eligible, cells, families, repeat)
    return {
        "study_id": design["study_id"],
        "study_kind": design["study_kind"],
        "held_fixed": held,
        "reporting_confidence": REPORTING_CONFIDENCE,
        "cap_peak_prompt_chars": {
            str(depth): peak for depth, peak in second_fold_cap_peaks(design).items()
        },
        "control_recheck": control_row,
        "cells": cells,
        "families": families,
        "repeat_geometry": repeat,
        "equivalence": design["equivalence"],
        "second_fold_reading": reading,
        "reason": reason,
        "changes_shipped_default": False,
    }


def _cell_row(
    cell: dict[str, object],
    held: dict[str, object],
    report: PolicyReport,
    digest: str,
) -> dict[str, object]:
    """One compact cell's cost, fold regime, per-fold summarize input, and validity."""
    folds = [episode.telemetry.n_compactions for episode in report.episodes]
    row: dict[str, object] = {
        "cell_id": cell["cell_id"],
        "family_id": cell["family_id"],
        "depth": cell["depth"],
        "compact_share": cell["compact_share"],
        "max_prompt_chars": cell["max_prompt_chars"],
        "summary_input_cap": as_str(held, "summary_input_cap"),
        "repeats_anchor": cell.get("repeats_anchor"),
        "expected_separation": bool(as_mapping(cell, "expected")["separates_from_anchor"]),
        **probe_second_fold_cell(cell, held),
        "task_set_digest": digest,
        "n_tasks": len(report.episodes),
        "completion": report.result.objective_score,
        "mean_steps": report.mean_steps,
        "case_success": list(report.case_success),
        "case_total_model_input_tokens": report.vector(METRIC_TOTAL_MODEL_INPUT_TOKENS),
        "mean_total_model_input_tokens": report.metric_mean(METRIC_TOTAL_MODEL_INPUT_TOKENS),
        "mean_compaction_prompt_tokens": report.metric_mean(METRIC_COMPACTION_PROMPT_TOKENS),
        "mean_controller_prompt_tokens": (
            report.metric_mean(METRIC_TOTAL_MODEL_INPUT_TOKENS)
            - report.metric_mean(METRIC_COMPACTION_PROMPT_TOKENS)
        ),
        "measured_fold_counts": folds,
        "measured_fold_input_chars": measured_fold_inputs(report),
        "context_overflows": report.n_context_overflow,
        "statuses": [episode.status for episode in report.episodes],
    }
    reason = _invalid_reason(row, held)
    return {**row, "valid": reason is None, "invalid_reason": reason}


def measured_fold_inputs(report: PolicyReport) -> list[dict[str, object]]:
    """What the summarizer was offered at each fold ordinal, averaged over the episodes reaching it.

    Kept per ORDINAL rather than summed, because the guard enters the later folds and not the first
    one: a growing offered transcript at fold two is the mechanism a total would hide.
    """
    per_episode = [episode.telemetry.summary_fold_input_chars for episode in report.episodes]
    depth = max((len(folds) for folds in per_episode), default=0)
    return [
        {
            "fold": index + 1,
            "n_episodes": len(reached),
            "mean_offered_chars": mean(reached),
            "max_offered_chars": max(reached),
        }
        for index in range(depth)
        if (reached := [float(folds[index]) for folds in per_episode if index < len(folds)])
    ]


def _invalid_reason(row: dict[str, object], held: dict[str, object]) -> str | None:
    """The preconditions a compact-only cell owes before its cost is comparable."""
    minimum_folds = as_int(held, "minimum_measured_folds")
    folds = cast(list[int], row["measured_fold_counts"])
    if int(cast(int, row["context_overflows"])) > 0:
        return "the compact arm overflowed its window, so the cost is overflow rescue"
    short = [count for count in folds if count < minimum_folds]
    if short:
        return (
            f"{len(short)} case(s) folded fewer than {minimum_folds} times ({folds}), so the cell "
            "left the repeatedly folding regime this study measures"
        )
    completion = float(cast(float, row["completion"]))
    floor = as_float(held, "minimum_cell_completion")
    if completion < floor:
        return f"completion {completion:.3f} is below the {floor:.3f} cell floor"
    return None
