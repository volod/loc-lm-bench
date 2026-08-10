"""Restate the published compact crossovers under the summarize-input bound the runtime ships.

The retired bound trimmed the summarize call's input wherever the folded transcript outgrew the
compaction trigger, and a trimmed input is a smaller prompt -- so every cell where it actually
trimmed carries a one-sided discount on compact's own cost. The audit
(`agentic_memory_cap_audit`) says with no model which published cells those are; this module
re-measures ONLY those, substitutes them into the published cell set, re-runs each study's own
published rule, and checks the restated crossover still names the fold step it named before.

A cell the audit calls bound-invariant is not re-run: its two bounds send bit-identical prompts, so
re-measuring it would spend a GPU to reproduce a number that cannot have changed.
"""

from pathlib import Path
from typing import cast

from llb.bench.agentic.context_policy import SUMMARY_INPUT_CAP_WINDOW
from llb.bench.agentic_memory_cap_audit import VERDICT_SENSITIVE, audit_design, audit_summary
from llb.bench.agentic_policy_change_audit import KIND_SURFACE
from llb.bench.agentic_memory_crossover_restatement_design import (
    audited_designs,
    published_crossovers,
)
from llb.bench.agentic_memory_crossover_restatement_reading import (
    operator_lines,
    restatement_reading,
)
from llb.bench.agentic_memory_crossover_restatement_forms import crossover_row
from llb.bench.agentic_memory_crossover_restatement_rows import (
    cap_peak_rows,
    restated_cells,
    restated_surfaces,
)
from llb.bench.agentic_memory_transfer_cells import run_transfer_cell
from llb.bench.common import LLMComplete


def audit_published_cells(design: dict[str, object], *, root: Path) -> dict[str, object]:
    """The model-free audit of every audited study: which published cells the bound can move."""
    audits = {
        kind: audit_design(audited, study_kind=kind)
        for kind, audited in audited_designs(design, root=root).items()
    }
    return {"per_study": audits, "summary": audit_summary(audits)}


def run_sensitive_surface_cells(
    design: dict[str, object],
    audit: dict[str, object],
    *,
    root: Path,
    model: str,
    backend: str,
    complete: LLMComplete,
    data_dir: Path,
) -> list[dict[str, object]]:
    """Re-measure only the surface's bound-sensitive cells, under the shipped bound."""
    held = cast(dict[str, object], audited_designs(design, root=root)[KIND_SURFACE]["held_fixed"])
    sensitive = [
        row
        for row in cast(dict[str, list[dict[str, object]]], audit["per_study"])[KIND_SURFACE]
        if row["verdict"] == VERDICT_SENSITIVE
    ]
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
                compact_share=float(cast(float, cell["compact_share"])),
                summary_input_cap=SUMMARY_INPUT_CAP_WINDOW,
                cell_id=cast(str, cell["cell_id"]),
                depth=int(cast(int, cell["depth"])),
                max_prompt_chars=int(cast(int, cell["max_prompt_chars"])),
                require_cap_fits=True,
            ),
            "restated_cell_id": cell["cell_id"],
        }
        for cell in sensitive
    ]


def analyze_restatement(
    design: dict[str, object],
    audit: dict[str, object],
    control_row: dict[str, object] | None,
    published_surface: dict[str, object] | None,
    restated_rows: list[dict[str, object]],
    *,
    root: Path,
) -> dict[str, object]:
    """Substitute the re-measured cells, re-run the published rule, and read every crossover."""
    designs = audited_designs(design, root=root)
    eligible = bool(control_row is not None and control_row["eligible"])
    summary = cast(dict[str, object], audit["summary"])
    surfaces: list[dict[str, object]] = []
    cells: list[dict[str, object]] = []
    cap_peaks: list[dict[str, object]] = []
    if eligible and published_surface is not None and restated_rows:
        cells = restated_cells(designs, restated_rows)
        surfaces = restated_surfaces(designs, published_surface, cells)
        cap_peaks = cap_peak_rows(published_surface, surfaces)
    crossovers = [
        crossover_row(row, designs, audit, surfaces) for row in published_crossovers(design)
    ]
    reading, reason = restatement_reading(
        eligible, crossovers, int(cast(int, summary["n_bound_sensitive"]))
    )
    shipped = cast(str, design["shipped_summary_input_cap"])
    return {
        "study_id": design["study_id"],
        "shipped_summary_input_cap": shipped,
        "restatement_rule": design["restatement_rule"],
        "audit": audit,
        "control_recheck": control_row,
        "restated_cells": cells,
        "restated_depth_surface": surfaces,
        "restated_cap_peaks": cap_peaks,
        "crossovers": crossovers,
        "restatement_reading": reading,
        "reason": reason,
        "operator_lines": operator_lines(crossovers, cap_peaks, reading, shipped),
        "changes_shipped_default": False,
    }
