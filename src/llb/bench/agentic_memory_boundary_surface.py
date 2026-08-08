"""Map the cap-fitting compact cost crossover instead of trusting one usable geometry.

The replication measured ONE cap-fitting cell: at depth 6 with a 12,000-char guard, compact ties
completion and more than repays its summary call. That single cell cannot say whether the saving is
a property of memory-dependent work or of that geometry. This study holds the eligible family,
typed memory tasks, observation cap, activation floor, and summarizer-inclusive accounting fixed
and varies ONLY transcript depth and prompt guard over a predeclared grid of cap-fitting cells,
then interpolates the guard at which the paired cost delta changes sign.
"""

from pathlib import Path
from typing import cast

from llb.bench.agentic_memory_boundary_crossover import (
    READING_BRACKETED,
    RULE_LINEAR_ZERO_CROSSING,
    depth_surface_row,
    routing_rule,
)
from llb.bench.agentic_memory_boundary_probe import cap_peak_prompt_chars
from llb.bench.agentic_design_fields import as_mapping, as_rows
from llb.bench.agentic_memory_boundary_surface_cells import (
    surface_cell_row,
    validate_surface_cells,
)
from llb.bench.agentic_memory_transfer import load_transfer_design
from llb.bench.agentic_memory_transfer_cells import (
    held_summary_input_cap,
    run_transfer_cell,
)
from llb.bench.common import LLMComplete
from llb.rag.fusion_evidence.evidence_gate import minimum_discordant_pairs

STUDY_KIND = "compact_memory_boundary_surface"
METHOD = "agentic-compact-memory-boundary-surface"
REPORTING_CONFIDENCE = 0.975

READING_INELIGIBLE = "pinned_family_control_ineligible"
READING_INVALID = "surface_cells_invalid"
READING_MAPPED = "surface_mapped"
READING_PARTIAL = "surface_partially_mapped"
READING_NOT_BRACKETED = "surface_not_bracketed"


def load_surface_design(path: Path | str) -> dict[str, object]:
    """Reuse the strict JSON loader; surface-specific validation runs separately."""
    return load_transfer_design(path)


def validate_surface_design(design: dict[str, object]) -> None:
    """Refuse a grid that is not cap-fitting, not bracketing, or not held fixed."""
    if design.get("schema_version") != 1 or design.get("study_kind") != STUDY_KIND:
        raise ValueError("compact-memory boundary-surface design schema or study kind is invalid")
    confidence = float(cast(float, design.get("reporting_confidence", 0.0)))
    if confidence != REPORTING_CONFIDENCE:
        raise ValueError(f"boundary-surface reporting confidence must be {REPORTING_CONFIDENCE}")

    held = cast(dict[str, object], design.get("held_fixed", {}))
    share = float(cast(float, held.get("compact_share", 0.0)))
    completion_floor = float(cast(float, held.get("minimum_cell_completion", 0.0)))
    activation_floor = float(cast(float, held.get("minimum_compaction_rate", -1.0)))
    if not held.get("model") or not held.get("model_family") or held.get("backend") != "ollama":
        raise ValueError("the boundary surface must pin one local Ollama model family")
    if not 0.0 < share <= 1.0 or not 0.0 <= activation_floor <= 1.0:
        raise ValueError("boundary-surface trigger share or activation floor is invalid")
    if not 0.0 < completion_floor <= 1.0:
        raise ValueError("boundary-surface cells need a positive completion floor")
    if int(cast(int, held.get("n_tasks", 0))) < minimum_discordant_pairs(confidence):
        raise ValueError("boundary-surface cells cannot reach the 97.5% exact-sign-test floor")

    control = cast(dict[str, object], design.get("control_recheck", {}))
    threshold = float(cast(float, control.get("minimum_completion_rate", 0.0)))
    if (
        int(cast(int, control.get("n_tasks", 0))) < 1
        or int(cast(int, control.get("depth", 0))) < 3
        or not 0.0 < threshold <= 1.0
    ):
        raise ValueError("boundary-surface control recheck contract is invalid")

    surface = cast(dict[str, object], design.get("surface", {}))
    rule = cast(dict[str, object], surface.get("interpolation", {}))
    if (
        rule.get("rule") != RULE_LINEAR_ZERO_CROSSING
        or rule.get("axis") != "max_prompt_chars"
        or rule.get("metric") != "compact_minus_cap_total_model_input_tokens"
        or rule.get("requires_sign_change") is not True
        or rule.get("requires_cost_separation_at_both_endpoints") is not True
        or rule.get("outside_grid") != "report_bounded_direction"
    ):
        raise ValueError("the boundary surface must predeclare the linear zero-crossing rule")
    validate_surface_cells(
        cast(list[dict[str, object]], surface.get("cells", [])),
        reference=cast(dict[str, object], design.get("reference", {})),
        held_fixed=held,
    )


def surface_cap_peaks(design: dict[str, object]) -> dict[int, int]:
    """The deterministic cap peak prompt behind every tested depth (no model, no GPU)."""
    held = cast(dict[str, object], design["held_fixed"])
    cells = cast(list[dict[str, object]], cast(dict[str, object], design["surface"])["cells"])
    return {
        depth: cap_peak_prompt_chars(
            depth=depth,
            n_tasks=int(cast(int, held["n_tasks"])),
            pad_chars=int(cast(int, held["pad_chars"])),
            max_steps_margin=int(cast(int, held["max_steps_margin"])),
            observation_cap_chars=int(cast(int, held["observation_cap_chars"])),
            observation_head_share=float(cast(float, held["observation_head_share"])),
        )
        for depth in sorted({int(cast(int, cell["depth"])) for cell in cells})
    }


def run_surface_grid(
    design: dict[str, object], *, model: str, backend: str, complete: LLMComplete, data_dir: Path
) -> list[dict[str, object]]:
    """Run every predeclared cap-fitting cell in declared order."""
    held = cast(dict[str, object], design["held_fixed"])
    return [
        run_transfer_cell(
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
        )
        for cell in cast(
            list[dict[str, object]], cast(dict[str, object], design["surface"])["cells"]
        )
    ]


def _surface_cells(
    design: dict[str, object], grid_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Score every declared cell, refusing rows that are not the declared grid in declared order."""
    declared = as_rows(as_mapping(design, "surface"), "cells")
    if [row["cell_id"] for row in grid_rows] != [cell["cell_id"] for cell in declared]:
        raise ValueError("boundary-surface rows do not match the exact declared cell order")
    held = as_mapping(design, "held_fixed")
    return [
        surface_cell_row(row, cell, held_fixed=held, confidence=REPORTING_CONFIDENCE)
        for row, cell in zip(grid_rows, declared, strict=True)
    ]


def analyze_surface(
    design: dict[str, object],
    control_row: dict[str, object] | None,
    grid_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Gate every cell, interpolate one crossover per depth, and state the routing surface."""
    peaks = surface_cap_peaks(design)
    eligible = bool(control_row is not None and control_row["eligible"])
    cells = _surface_cells(design, grid_rows) if eligible else []
    depth_rows = (
        [
            depth_surface_row(
                depth,
                [cell for cell in cells if cell["depth"] == depth],
                cap_peak_prompt_chars=peaks[depth],
            )
            for depth in sorted(peaks)
        ]
        if eligible
        else []
    )
    reading, reason = _surface_reading(eligible, cells, depth_rows)
    return {
        "study_id": design["study_id"],
        "held_fixed": as_mapping(design, "held_fixed"),
        "cap_peak_prompt_chars": {str(depth): peak for depth, peak in peaks.items()},
        "control_recheck": control_row,
        "cells": cells,
        "depth_surface": depth_rows,
        "expectation_matches": sum(bool(cell["matches_expectation"]) for cell in cells),
        "routing_rule": routing_rule(depth_rows),
        "surface_reading": reading,
        "reason": reason,
        "changes_shipped_default": False,
    }


def _surface_reading(
    eligible: bool, cells: list[dict[str, object]], depth_rows: list[dict[str, object]]
) -> tuple[str, str]:
    """Eligibility and cell validity gate the surface before any crossover counts."""
    invalid = [cell for cell in cells if not cell["valid"]]
    bracketed = [row for row in depth_rows if row["reading"] == READING_BRACKETED]
    if not eligible:
        return (
            READING_INELIGIBLE,
            "the pinned family no longer passes the unchanged token-chain control",
        )
    if invalid:
        named = "; ".join(f"{cell['cell_id']}: {cell['invalid_reason']}" for cell in invalid)
        return (
            READING_INVALID,
            f"a predeclared cap-fitting cell did not hold its preconditions: {named}",
        )
    if len(bracketed) == len(depth_rows):
        return READING_MAPPED, "every tested depth brackets a measured cost crossover"
    if bracketed:
        return (
            READING_PARTIAL,
            f"{len(bracketed)}/{len(depth_rows)} tested depths bracket a cost crossover",
        )
    return (
        READING_NOT_BRACKETED,
        "no tested depth changes cost sign; the crossover lies outside the grid",
    )
