"""Is the compact crossover a char threshold, or the boundary between two fold steps?

The boundary surface published an interpolated guard, and the trigger-collapse study then showed why
that number cannot be taken literally: the trigger reaches the transcript only by choosing WHICH step
folds, so the cost delta is a step function and every guard inside one step's interval costs the
same. This study measures the mechanism directly -- a grid whose cells sit one fold step apart around
the published crossover, two guards per step spanning that step's whole interval and straddling each
step change by a few chars -- and states the routing rule as the LAST fold step at which compact is
still cheaper plus the trigger interval that selects it.
"""

from pathlib import Path
from typing import cast

from llb.bench.agentic_design_fields import as_float, as_mapping
from llb.bench.agentic_memory_boundary_surface_cells import surface_cell_row
from llb.bench.agentic_memory_fold_step_design import (
    declared_cells,
    fold_step_cap_peaks,
    fold_step_prompt_sequences,
)
from llb.bench.agentic_memory_fold_step_reading import (
    READING_CONFIRMED,
    READING_PARTIAL,
    REPORTING_CONFIDENCE,
    fold_step_reading,
    routing_rule,
)
from llb.bench.agentic_memory_fold_step_rows import depth_fold_row, step_rows
from llb.bench.agentic_memory_transfer_cells import held_summary_input_cap, run_transfer_cell
from llb.bench.agentic_memory_trigger_collapse_reading import annotate_fold_steps
from llb.bench.common import LLMComplete


def run_fold_step_ladders(
    design: dict[str, object], *, model: str, backend: str, complete: LLMComplete, data_dir: Path
) -> list[dict[str, object]]:
    """Run every declared cell in declared order, tagged with the fold step it must land on."""
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
                summary_input_cap=held_summary_input_cap(held),
                compact_share=float(cast(float, held["compact_share"])),
                cell_id=cast(str, cell["cell_id"]),
                depth=int(cast(int, cell["depth"])),
                max_prompt_chars=int(cast(int, cell["max_prompt_chars"])),
                require_cap_fits=True,
            ),
            "declared_fold_step": cell["fold_step"],
        }
        for cell in declared_cells(design)
    ]


def _annotated_cells(
    design: dict[str, object],
    grid_rows: list[dict[str, object]],
    sequences: dict[int, list[int]],
) -> list[dict[str, object]]:
    """Score every declared cell and annotate the fold step it actually lands on."""
    declared = declared_cells(design)
    if [row["cell_id"] for row in grid_rows] != [cell["cell_id"] for cell in declared]:
        raise ValueError("fold-step rows do not match the exact declared cell order")
    held = as_mapping(design, "held_fixed")
    cells = annotate_fold_steps(
        [
            surface_cell_row(row, cell, held_fixed=held, confidence=REPORTING_CONFIDENCE)
            for row, cell in zip(grid_rows, declared, strict=True)
        ],
        sequences,
    )
    _refuse_mispredicted_steps(cells)
    return cells


def _depth_ladders(
    design: dict[str, object],
    cells: list[dict[str, object]],
    sequences: dict[int, list[int]],
    peaks: dict[int, int],
) -> list[dict[str, object]]:
    """One step ladder per depth: which guards fold at which step, and what each step costs."""
    held = as_mapping(design, "held_fixed")
    share = as_float(held, "compact_share")
    fraction = as_float(as_mapping(design, "step_rule"), "within_step_cap_cost_fraction")
    return [
        depth_fold_row(
            depth,
            step_rows(
                [cell for cell in cells if cell["depth"] == depth],
                prompt_sequence=sequences[depth],
                compact_share=share,
                cap_cost_fraction=fraction,
            ),
            prompt_sequence=sequences[depth],
            compact_share=share,
            cap_peak_prompt_chars=peaks[depth],
            reference_guard=_reference_guard(design, depth),
        )
        for depth in sorted(sequences)
    ]


def analyze_fold_steps(
    design: dict[str, object],
    control_row: dict[str, object] | None,
    grid_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Gate the cells, group them by fold step, and read each depth's step ladder."""
    sequences = fold_step_prompt_sequences(design)
    peaks = fold_step_cap_peaks(design)
    eligible = bool(control_row is not None and control_row["eligible"])
    cells = _annotated_cells(design, grid_rows, sequences) if eligible else []
    depth_rows = _depth_ladders(design, cells, sequences, peaks) if eligible else []
    reading, reason = fold_step_reading(eligible, cells, depth_rows)
    return {
        "study_id": design["study_id"],
        "held_fixed": as_mapping(design, "held_fixed"),
        "step_rule": as_mapping(design, "step_rule"),
        "cap_peak_prompt_chars": {str(depth): peak for depth, peak in peaks.items()},
        "cap_prompt_sequence": {str(depth): sequence for depth, sequence in sequences.items()},
        "control_recheck": control_row,
        "cells": cells,
        "depth_ladders": depth_rows,
        "expectation_matches": sum(bool(cell["matches_expectation"]) for cell in cells),
        "fold_step_routing_rule": (
            routing_rule(depth_rows) if reading in (READING_CONFIRMED, READING_PARTIAL) else []
        ),
        "fold_step_reading": reading,
        "reason": reason,
        "changes_shipped_default": False,
    }


def _refuse_mispredicted_steps(cells: list[dict[str, object]]) -> None:
    """The declared step is the cell's identity; a probe that disagrees invalidates the grid."""
    drifted = [
        f"{cell['cell_id']}: declared {cell['declared_fold_step']}, "
        f"predicted {cell['predicted_fold_step']}"
        for cell in cells
        if cell["declared_fold_step"] != cell["predicted_fold_step"]
    ]
    if drifted:
        raise ValueError(f"fold-step cells no longer fold where they were declared: {drifted}")


def _reference_guard(design: dict[str, object], depth: int) -> int | None:
    """The interpolated guard the boundary surface published for this depth, if it published one."""
    reference = cast(dict[str, object], design.get("reference", {}))
    for row in cast(list[dict[str, object]], reference.get("measured_crossovers", [])):
        if int(cast(int, row["depth"])) == depth:
            return int(cast(int, row["crossover_max_prompt_chars"]))
    return None
