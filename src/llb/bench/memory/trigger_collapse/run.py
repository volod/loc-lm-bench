"""Does the compact routing rule live on the trigger axis alone, or on share AND guard?

The cap-fitting boundary surface swept the prompt guard at ONE `compact_share`, so the guard it
reported is a stand-in for what the policy actually reacts to: the compaction trigger
(`int(guard * compact_share)`). If the paired cost delta depends only on that product, the surface
collapses to one axis and an operator can convert a measured crossover to any other share without
re-running a grid. This study measures that directly -- families of cells that hold the trigger
constant while moving share and guard inversely, plus a contrast family that holds the guard and
moves the trigger, which is the positive control that the measurement can see a trigger change at
all.
"""

from pathlib import Path
from typing import cast

from llb.bench.memory.boundary.surface_cells import surface_cell_row
from llb.bench.memory.transfer.cells import (
    held_summary_input_cap,
    run_transfer_cell,
)
from llb.bench.memory.trigger_collapse.design import (
    collapse_cap_peaks,
    collapse_prompt_sequences,
)
from llb.bench.memory.trigger_collapse.reading import (
    READING_COLLAPSES,
    REPORTING_CONFIDENCE,
    annotate_fold_steps,
    collapse_reading,
    family_rows,
    portable_trigger_rule,
)
from llb.bench.common import LLMComplete


def run_collapse_families(
    design: dict[str, object], *, model: str, backend: str, complete: LLMComplete, data_dir: Path
) -> list[dict[str, object]]:
    """Run every declared cell in declared order, tagged with the family it belongs to."""
    held = cast(dict[str, object], design["held_fixed"])
    rows: list[dict[str, object]] = []
    for family in cast(list[dict[str, object]], design["families"]):
        for cell in cast(list[dict[str, object]], family["cells"]):
            row = run_transfer_cell(
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
                compact_share=float(cast(float, cell["compact_share"])),
                cell_id=cast(str, cell["cell_id"]),
                depth=int(cast(int, family["depth"])),
                max_prompt_chars=int(cast(int, cell["max_prompt_chars"])),
                require_cap_fits=True,
            )
            rows.append({**row, "family_id": family["family_id"]})
    return rows


def analyze_collapse(
    design: dict[str, object],
    control_row: dict[str, object] | None,
    grid_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Gate the cells, measure each family's spread against its band, and decide the collapse."""
    held = cast(dict[str, object], design["held_fixed"])
    declared = [
        cell
        for family in cast(list[dict[str, object]], design["families"])
        for cell in cast(list[dict[str, object]], family["cells"])
    ]
    peaks = collapse_cap_peaks(design)
    eligible = bool(control_row is not None and control_row["eligible"])
    cells: list[dict[str, object]] = []
    families: list[dict[str, object]] = []
    if eligible:
        if [row["cell_id"] for row in grid_rows] != [cell["cell_id"] for cell in declared]:
            raise ValueError("trigger-collapse rows do not match the exact declared cell order")
        cells = annotate_fold_steps(
            [
                surface_cell_row(row, cell, held_fixed=held, confidence=REPORTING_CONFIDENCE)
                for row, cell in zip(grid_rows, declared, strict=True)
            ],
            collapse_prompt_sequences(design),
        )
        families = family_rows(design, cells)
    reading, reason = collapse_reading(design, eligible, cells, families)
    return {
        "study_id": design["study_id"],
        "held_fixed": held,
        "cap_peak_prompt_chars": {str(depth): peak for depth, peak in peaks.items()},
        "control_recheck": control_row,
        "cells": cells,
        "families": families,
        "equivalence": design["equivalence"],
        "portable_trigger_rule": (
            portable_trigger_rule(design, peaks) if reading == READING_COLLAPSES else []
        ),
        "collapse_reading": reading,
        "reason": reason,
        "changes_shipped_default": False,
    }
