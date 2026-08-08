"""Does pinning the summarize call's input to a step-aligned quantity make compact a step function?

The fold-step study left one term moving inside a fold step: the summarize call, whose input cap WAS
the compaction trigger, so two guards folding the same transcript fed the summarizer different
amounts of it. This study re-runs ONE of that study's ladders twice -- once under the trigger cap it
published, once under the window-derived cap the runtime now ships -- and reads two things: whether
the within-step residual falls to zero without moving the fold-step boundary, and whether the span
the trigger cap elided from the summarizer input was carrying completion.
"""

from pathlib import Path
from typing import cast

from llb.bench.agentic_memory_boundary_surface_cells import surface_cell_row
from llb.bench.agentic_memory_fold_step_ladder import measured_cap_peak
from llb.bench.agentic_memory_summary_cap_design import (
    arm_fold_input_probes,
    declared_cells,
    summary_cap_depth,
    summary_cap_prompt_sequence,
)
from llb.bench.agentic_memory_summary_cap_reading import (
    ELISION_NONE_TO_PRICE,
    READING_EXACT,
    READING_INELIGIBLE,
    READING_INVALID,
    REPORTING_CONFIDENCE,
    ROLE_STEP_ALIGNED,
    elision_reading,
    operator_lines,
    residual_reading,
)
from llb.bench.agentic_memory_summary_cap_rows import (
    arm_rows,
    completion_delta_reading,
    elision_completion_delta,
    reference_elided_chars,
)
from llb.bench.agentic_memory_transfer_cells import run_transfer_cell
from llb.bench.agentic_memory_trigger_collapse_reading import annotate_fold_steps
from llb.bench.common import LLMComplete
from llb.rag.fusion_evidence.paired import PairedComparison


def run_summary_cap_arms(
    design: dict[str, object], *, model: str, backend: str, complete: LLMComplete, data_dir: Path
) -> list[dict[str, object]]:
    """Run every declared cell of every arm, in declared arm-major order."""
    held = cast(dict[str, object], design["held_fixed"])
    return [
        {
            **run_transfer_cell(
                model=model,
                backend=backend,
                complete=complete,
                data_dir=data_dir,
                n_tasks=int(cast(int, held["n_tasks"])),
                pad_chars=int(cast(int, held["pad_chars"])),
                max_steps_margin=int(cast(int, held["max_steps_margin"])),
                observation_cap_chars=int(cast(int, held["observation_cap_chars"])),
                observation_head_share=float(cast(float, held["observation_head_share"])),
                minimum_compaction_rate=float(cast(float, held["minimum_compaction_rate"])),
                compact_share=float(cast(float, held["compact_share"])),
                summary_input_cap=cast(str, cell["summary_input_cap"]),
                cell_id=f"{cast(str, cell['cell_id'])}@{cast(str, cell['arm_id'])}",
                depth=int(cast(int, cell["depth"])),
                max_prompt_chars=int(cast(int, cell["max_prompt_chars"])),
                require_cap_fits=True,
            ),
            "declared_cell_id": cell["cell_id"],
            "declared_fold_step": cell["fold_step"],
            "arm_id": cell["arm_id"],
            "role": cell["role"],
        }
        for cell in declared_cells(design)
    ]


def analyze_summary_cap(
    design: dict[str, object],
    control_row: dict[str, object] | None,
    arm_cell_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Gate the cells, build one fold-step ladder per arm, and read residual plus elision."""
    held = cast(dict[str, object], design["held_fixed"])
    rule = cast(dict[str, object], design["cap_rule"])
    arms = cast(list[dict[str, object]], design["arms"])
    depth = summary_cap_depth(design)
    sequence = summary_cap_prompt_sequence(design)
    declared = declared_cells(design)
    eligible = bool(control_row is not None and control_row["eligible"])
    cells: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    completion: PairedComparison | None = None
    elided = 0.0
    if eligible:
        cells = _cells(arm_cell_rows, declared, held, sequence, depth)
        rows = arm_rows(
            cells,
            arms=arms,
            prompt_sequence=sequence,
            depth=depth,
            compact_share=float(cast(float, held["compact_share"])),
            cap_cost_fraction=float(cast(float, rule["within_step_cap_cost_fraction"])),
        )
        completion = elision_completion_delta(cells, arms, confidence=REPORTING_CONFIDENCE)
        elided = reference_elided_chars(cells, arms)
    residual, residual_reason = residual_reading(
        eligible,
        cells,
        rows,
        tolerance_tokens=float(cast(float, rule["residual_tolerance_tokens"])),
    )
    completion_label = (
        completion_delta_reading(completion, confidence=REPORTING_CONFIDENCE)
        if completion is not None
        else ""
    )
    # An ineligible family or a cell that failed its preconditions invalidates the completion
    # comparison too: the elision reading is cut from the same episodes the residual reading is.
    elision, elision_reason = (
        elision_reading(
            elided, completion_label, cast(dict[str, float], completion["delta"])["mean"]
        )
        if completion is not None and residual not in (READING_INELIGIBLE, READING_INVALID)
        else (ELISION_NONE_TO_PRICE, f"[{residual}] no arm produced a priceable elision")
    )
    shipped = cast(
        str,
        next(arm for arm in arms if arm["role"] == ROLE_STEP_ALIGNED)["summary_input_cap"],
    )
    return {
        "study_id": design["study_id"],
        "held_fixed": held,
        "cap_rule": rule,
        "shipped_summary_input_cap": shipped,
        "depth": depth,
        "cap_prompt_sequence": sequence,
        "cap_peak_prompt_chars": measured_cap_peak(sequence, geometry=f"the depth {depth} ladder"),
        "fold_input_probe": arm_fold_input_probes(design),
        "control_recheck": control_row,
        "cells": cells,
        "arms": rows,
        "expectation_matches": sum(bool(cell["matches_expectation"]) for cell in cells),
        "reference_elided_chars": elided,
        "elision_completion_delta": completion,
        "elision_completion_delta_reading": completion_label,
        "summary_cap_reading": residual,
        "reason": residual_reason,
        "elision_reading": elision,
        "elision_reason": elision_reason,
        "operator_lines": operator_lines(rows, residual, elision, shipped) if rows else [],
        "changes_shipped_default": residual == READING_EXACT,
    }


def _cells(
    arm_cell_rows: list[dict[str, object]],
    declared: list[dict[str, object]],
    held: dict[str, object],
    sequence: list[int],
    depth: int,
) -> list[dict[str, object]]:
    """Gate every measured cell and attach the fold step the deterministic probe predicts for it."""
    measured = [(row["arm_id"], row["declared_cell_id"]) for row in arm_cell_rows]
    if measured != [(cell["arm_id"], cell["cell_id"]) for cell in declared]:
        raise ValueError("summarize-input-cap rows do not match the exact declared arm/cell order")
    cells = annotate_fold_steps(
        [
            surface_cell_row(row, cell, held_fixed=held, confidence=REPORTING_CONFIDENCE)
            for row, cell in zip(arm_cell_rows, declared, strict=True)
        ],
        {depth: sequence},
    )
    drifted = [
        f"{cell['cell_id']}: declared {cell['declared_fold_step']}, "
        f"predicted {cell['predicted_fold_step']}"
        for cell in cells
        if cell["declared_fold_step"] != cell["predicted_fold_step"]
    ]
    if drifted:
        raise ValueError(f"summarize-input-cap cells no longer fold where declared: {drifted}")
    return cells
