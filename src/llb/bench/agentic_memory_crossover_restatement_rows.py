"""Row builders for the crossover restatement: re-measured cells, surfaces, and crossover rows.

Substituting a re-measured cell into a published grid and re-running that grid's own rule is the
whole restatement, so these builders reuse the published machinery rather than reimplementing it:
the surface's precondition gate (`surface_cell_row`) and its predeclared interpolation
(`depth_surface_row`). What is added is the check the restatement exists for -- whether the restated
number still names the fold step the published one named.
"""

from typing import cast

from llb.bench.agentic_memory_boundary_crossover import READING_BRACKETED, depth_surface_row
from llb.bench.agentic_memory_boundary_probe import (
    cap_prompt_sequence,
    compaction_trigger_chars,
    first_fold_step,
    fold_step_guard_interval,
)
from llb.bench.agentic_memory_boundary_surface_cells import surface_cell_row
from llb.bench.agentic_memory_cap_audit import KIND_SURFACE, VERDICT_SENSITIVE
from llb.bench.agentic_memory_crossover_restatement_reading import (
    BASIS_ALREADY_MEASURED,
    BASIS_INVARIANT,
    BASIS_RESTATED,
    FORM_INTERPOLATED,
    REPORTING_CONFIDENCE,
)


def restated_cells(
    designs: dict[str, dict[str, object]], restated_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Gate the re-measured cells with the surface's own precondition contract."""
    held = cast(dict[str, object], designs[KIND_SURFACE]["held_fixed"])
    declared = {
        cast(str, cell["cell_id"]): cell
        for cell in cast(
            list[dict[str, object]],
            cast(dict[str, object], designs[KIND_SURFACE]["surface"])["cells"],
        )
    }
    return [
        surface_cell_row(
            row,
            declared[cast(str, row["restated_cell_id"])],
            held_fixed=held,
            confidence=REPORTING_CONFIDENCE,
        )
        for row in restated_rows
    ]


def restated_surfaces(
    published_surface: dict[str, object], cells: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Re-run the surface's published interpolation over its cells with the new ones substituted."""
    replacements = {cast(str, cell["cell_id"]): cell for cell in cells}
    substituted = [
        replacements.get(cast(str, cell["cell_id"]), cell)
        for cell in cast(list[dict[str, object]], published_surface["cells"])
    ]
    peaks = cast(dict[str, object], published_surface["cap_peak_prompt_chars"])
    return [
        {
            **depth_surface_row(
                depth,
                [cell for cell in substituted if int(cast(int, cell["depth"])) == depth],
                cap_peak_prompt_chars=int(cast(int, peaks[str(depth)])),
            ),
            "restated_cell_ids": sorted(
                cast(str, cell["cell_id"])
                for cell in cells
                if int(cast(int, cell["depth"])) == depth
            ),
        }
        for depth in sorted({int(cast(int, cell["depth"])) for cell in substituted})
    ]


def crossover_row(
    published: dict[str, object],
    designs: dict[str, dict[str, object]],
    audit: dict[str, object],
    surfaces: list[dict[str, object]],
) -> dict[str, object]:
    """One published crossover, restated where a run was needed and re-checked where it was not."""
    kind = cast(str, published["study_kind"])
    depth = int(cast(int, published["depth"]))
    sensitive = [
        row
        for row in cast(dict[str, list[dict[str, object]]], audit["per_study"])[kind]
        if row["verdict"] == VERDICT_SENSITIVE and int(cast(int, row["depth"])) == depth
    ]
    row: dict[str, object] = {
        **published,
        "n_bound_sensitive_cells": len(sensitive),
        "bound_sensitive_cell_ids": [cast(str, cell["cell_id"]) for cell in sensitive],
        "published_value": published.get("value"),
        "published_fold_step": published["fold_step"],
        "restated_value": None,
        "restated_fold_step": published["fold_step"],
        "names_same_fold_step": True,
        "basis": BASIS_INVARIANT if not sensitive else None,
    }
    if not sensitive:
        return row
    if published["form"] != FORM_INTERPOLATED:
        # A fold-step boundary is a property of the deterministic ladder, not of a measured cost: its
        # sensitive cells were re-measured by the summarize-input-cap study, which confirmed the
        # boundary directly. Nothing here can restate it more strongly.
        row["basis"] = BASIS_ALREADY_MEASURED
        return row
    restated = next(
        (surface for surface in surfaces if int(cast(int, surface["depth"])) == depth), None
    )
    if restated is None or restated["reading"] != READING_BRACKETED:
        return row
    return _interpolated_row(row, published, restated, designs[kind], depth)


def _interpolated_row(
    row: dict[str, object],
    published: dict[str, object],
    restated: dict[str, object],
    design: dict[str, object],
    depth: int,
) -> dict[str, object]:
    """Place the re-interpolated guard on the deterministic step ladder it has to stay inside."""
    guard = float(cast(float, restated["crossover_max_prompt_chars"]))
    share = float(cast(float, published["compact_share"]))
    sequence = _prompt_sequence(design, depth)
    row.update(
        {
            "restated_value": guard,
            "restated_bracket": restated["bracket"],
            "restated_guard_ratio": restated["crossover_guard_ratio"],
            "restated_fold_step": first_fold_step(
                sequence, compaction_trigger_chars(int(guard), share)
            ),
            "basis": BASIS_RESTATED,
            "fold_step_guard_interval": list(
                fold_step_guard_interval(sequence, int(cast(int, published["fold_step"])), share)
            ),
        }
    )
    row["names_same_fold_step"] = row["restated_fold_step"] == published["fold_step"]
    return row


def _prompt_sequence(design: dict[str, object], depth: int) -> list[int]:
    held = cast(dict[str, object], design["held_fixed"])
    return cap_prompt_sequence(
        depth=depth,
        n_tasks=int(cast(int, held["n_tasks"])),
        pad_chars=int(cast(int, held["pad_chars"])),
        max_steps_margin=int(cast(int, held["max_steps_margin"])),
        observation_cap_chars=int(cast(int, held["observation_cap_chars"])),
        observation_head_share=float(cast(float, held["observation_head_share"])),
    )
