"""Design contract for completion under repeated compact-memory folds."""

from pathlib import Path
from typing import cast

from llb.bench.agentic.context_policy import POLICY_COMPACT
from llb.bench.agentic.design_fields import as_mapping, as_rows
from llb.bench.memory.boundary.probe import cap_peak_prompt_chars, compact_fold_input_probe
from llb.bench.policy_change.geometry import load_audited_design

DESIGN_PATH = "samples/benchmarks/agentic_compact_repeated_fold_completion_design.json"
STUDY_KIND = "compact_repeated_fold_completion"
MECHANISM_ARMS = ("typed_marker", "model_summary_only")


def load_repeated_fold_design(path: Path | str | None = None) -> dict[str, object]:
    """Load the committed completion design through the shared strict JSON loader."""
    from llb.core.paths import PROJECT_ROOT

    return load_audited_design(PROJECT_ROOT / DESIGN_PATH if path is None else path)


def completion_cells(design: dict[str, object]) -> list[dict[str, object]]:
    """Cells in increasing declared fold-count order."""
    return as_rows(design, "cells")


def cell_geometry(cell: dict[str, object], held: dict[str, object]) -> dict[str, int | float | str]:
    """Translate one declared cell to the compact runner's held geometry."""
    return {
        "depth": int(cast(int, cell["depth"])),
        "n_tasks": int(cast(int, held["n_tasks"])),
        "pad_chars": int(cast(int, held["pad_chars"])),
        "max_steps_margin": int(cast(int, held["max_steps_margin"])),
        "observation_cap_chars": int(cast(int, held["observation_cap_chars"])),
        "observation_head_share": float(cast(float, held["observation_head_share"])),
        "max_prompt_chars": int(cast(int, cell["max_prompt_chars"])),
        "compact_share": float(cast(float, held["compact_share"])),
        "summary_input_cap": str(cast(str, held["summary_input_cap"])),
    }


def probe_completion_cell(cell: dict[str, object], held: dict[str, object]) -> dict[str, object]:
    """Model-free oracle fold count and cap-fitting status for one completion cell."""
    geometry = cell_geometry(cell, held)
    depth = int(cast(int, geometry["depth"]))
    n_tasks = int(cast(int, geometry["n_tasks"]))
    pad_chars = int(cast(int, geometry["pad_chars"]))
    max_steps_margin = int(cast(int, geometry["max_steps_margin"]))
    observation_cap_chars = int(cast(int, geometry["observation_cap_chars"]))
    observation_head_share = float(cast(float, geometry["observation_head_share"]))
    probe = compact_fold_input_probe(
        depth=depth,
        n_tasks=n_tasks,
        pad_chars=pad_chars,
        max_steps_margin=max_steps_margin,
        observation_cap_chars=observation_cap_chars,
        observation_head_share=observation_head_share,
        max_prompt_chars=int(cast(int, geometry["max_prompt_chars"])),
        compact_share=float(cast(float, geometry["compact_share"])),
        summary_input_cap=str(cast(str, geometry["summary_input_cap"])),
    )
    peak = cap_peak_prompt_chars(
        depth=depth,
        n_tasks=n_tasks,
        pad_chars=pad_chars,
        max_steps_margin=max_steps_margin,
        observation_cap_chars=observation_cap_chars,
        observation_head_share=observation_head_share,
    )
    guard = int(cast(int, cell["max_prompt_chars"]))
    return {
        "oracle_folds": int(cast(int, probe["n_compactions"])),
        "oracle_fold_input_chars": probe["summary_fold_input_chars"],
        "cap_peak_prompt_chars": peak,
        "cap_fitting": guard >= peak,
    }


def validate_repeated_fold_design(design: dict[str, object]) -> None:
    """Refuse drift in the held model, source cells, ordering, or fold geometry."""
    if design.get("study_kind") != STUDY_KIND:
        raise ValueError(f"repeated-fold study_kind must be {STUDY_KIND!r}")
    if int(cast(int, design.get("seed", 0))) < 1:
        raise ValueError("repeated-fold completion needs a positive deterministic seed")
    held = as_mapping(design, "held_fixed")
    minimum_control = float(cast(float, held.get("minimum_control_completion", 0.0)))
    if not 0.0 < minimum_control <= 1.0:
        raise ValueError("minimum control completion must be in (0, 1]")
    if cast(list[str], design.get("policies", [])) != [POLICY_COMPACT]:
        raise ValueError("repeated-fold completion must run the compact policy alone")
    if tuple(cast(list[str], design.get("mechanism_arms", []))) != MECHANISM_ARMS:
        raise ValueError(f"mechanism arms must be {MECHANISM_ARMS!r}")
    cells = completion_cells(design)
    declared_folds = [int(cast(int, cell.get("expected_oracle_folds", 0))) for cell in cells]
    if declared_folds != sorted(set(declared_folds)) or declared_folds[0] != 1:
        raise ValueError("completion cells must declare unique increasing folds starting at one")
    digests = set()
    for cell in cells:
        measured = probe_completion_cell(cell, held)
        expected = int(cast(int, cell["expected_oracle_folds"]))
        if measured["oracle_folds"] != expected:
            raise ValueError(
                f"cell {cell['cell_id']!r} declares {expected} oracle folds but measures "
                f"{measured['oracle_folds']}"
            )
        if bool(measured["cap_fitting"]) != bool(cell["cap_fitting_control"]):
            raise ValueError(f"cell {cell['cell_id']!r} cap-fitting declaration drifted")
        digests.add((cell["depth"], held["n_tasks"], held["pad_chars"]))
    if len(digests) != 1:
        raise ValueError("every fold-count cell must run the identical memory task set")
