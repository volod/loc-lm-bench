"""Predeclared geometry for unavoidable elision under the window summary-input bound."""

from pathlib import Path
from typing import cast

from llb.bench.agentic.context_policy import POLICY_COMPACT, SUMMARY_INPUT_CAP_WINDOW
from llb.bench.agentic.design_fields import as_mapping, as_rows
from llb.bench.memory.boundary.probe import compact_fold_input_probe
from llb.bench.memory.fold_step.ladder import compaction_trigger_chars
from llb.bench.policy_change.geometry import load_audited_design

DESIGN_PATH = "samples/benchmarks/agentic_compact_window_elision_design.json"
STUDY_KIND = "compact_window_elision"
ROLE_FIT = "transcript_fits"
ROLE_ELIDED = "transcript_elided"
ARM_ROLES = (ROLE_FIT, ROLE_ELIDED)


def load_window_elision_design(path: Path | str | None = None) -> dict[str, object]:
    """Load the committed design through the shared strict JSON loader."""
    from llb.core.paths import PROJECT_ROOT

    return load_audited_design(PROJECT_ROOT / DESIGN_PATH if path is None else path)


def elision_cells(design: dict[str, object]) -> list[dict[str, object]]:
    """Return the fit control first so a failed control can stop a live run early."""
    cells = as_rows(design, "cells")
    return sorted(cells, key=lambda cell: cell["role"] != ROLE_FIT)


def cell_geometry(cell: dict[str, object], held: dict[str, object]) -> dict[str, object]:
    """Translate one arm to the deterministic probe and compact runner geometry."""
    return {
        "depth": int(cast(int, held["depth"])),
        "n_tasks": int(cast(int, held["n_tasks"])),
        "pad_chars": int(cast(int, held["pad_chars"])),
        "max_steps_margin": int(cast(int, held["max_steps_margin"])),
        "observation_cap_chars": int(cast(int, held["observation_cap_chars"])),
        "observation_head_share": float(cast(float, held["observation_head_share"])),
        "max_prompt_chars": int(cast(int, cell["max_prompt_chars"])),
        "compact_share": float(cast(float, cell["compact_share"])),
        "summary_input_cap": SUMMARY_INPUT_CAP_WINDOW,
    }


def probe_elision_cell(cell: dict[str, object], held: dict[str, object]) -> dict[str, object]:
    """Measure the folded input and elision without warming a model."""
    geometry = cell_geometry(cell, held)
    probe = compact_fold_input_probe(**geometry)  # type: ignore[arg-type]
    guard = int(cast(int, geometry["max_prompt_chars"]))
    share = float(cast(float, geometry["compact_share"]))
    return {**probe, "compaction_trigger_chars": compaction_trigger_chars(guard, share)}


def validate_window_elision_design(design: dict[str, object]) -> None:
    """Refuse a comparison that does not isolate elision at one shared fold."""
    held, cells = _validate_header(design)
    measured = [(cell, probe_elision_cell(cell, held)) for cell in cells]
    _validate_shared_geometry(measured)
    for cell, probe in measured:
        _validate_cell(cell, probe)


def _validate_header(
    design: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Validate the study vocabulary and return its typed geometry containers."""
    if design.get("study_kind") != STUDY_KIND:
        raise ValueError(f"window-elision study_kind must be {STUDY_KIND!r}")
    if int(cast(int, design.get("seed", 0))) < 1:
        raise ValueError("window-elision study needs a positive deterministic seed")
    if cast(list[str], design.get("policies", [])) != [POLICY_COMPACT]:
        raise ValueError("window-elision study must run the compact policy alone")
    held = as_mapping(design, "held_fixed")
    if held.get("summary_input_cap") != SUMMARY_INPUT_CAP_WINDOW:
        raise ValueError("window-elision study must pin the shipped window summary-input bound")
    if held.get("preserve_memory_markers") is not True:
        raise ValueError("window-elision study must measure the shipped typed-memory behavior")
    minimum = float(cast(float, held.get("minimum_control_completion", 0.0)))
    if not 0.0 < minimum <= 1.0:
        raise ValueError("minimum control completion must be in (0, 1]")
    cells = elision_cells(design)
    if len(cells) != 2 or {cell.get("role") for cell in cells} != set(ARM_ROLES):
        raise ValueError(f"window-elision cells must contain exactly the roles {ARM_ROLES!r}")
    return held, cells


def _validate_shared_geometry(
    measured: list[tuple[dict[str, object], dict[str, object]]],
) -> None:
    """Require a single trigger, fold count, and offered transcript across arms."""
    triggers = {probe["compaction_trigger_chars"] for _cell, probe in measured}
    inputs = {cast(list[int], probe["summary_fold_input_chars"])[0] for _cell, probe in measured}
    folds = {probe["n_compactions"] for _cell, probe in measured}
    if len(triggers) != 1:
        raise ValueError("window-elision arms must hold the compaction trigger fixed")
    if folds != {1} or len(inputs) != 1:
        raise ValueError("window-elision arms must fold the identical transcript exactly once")


def _validate_cell(cell: dict[str, object], probe: dict[str, object]) -> None:
    """Check one arm against its predeclaration and required side of the elision split."""
    expected = as_mapping(cell, "expected")
    fields = (
        "compaction_trigger_chars",
        "summary_input_chars",
        "summary_input_elided_chars",
        "n_compactions",
    )
    drifted = [field for field in fields if probe[field] != expected[field]]
    if drifted:
        field = drifted[0]
        raise ValueError(
            f"cell {cell['cell_id']!r} {field} drifted: "
            f"declared {expected[field]!r}, measured {probe[field]!r}"
        )
    elided = int(cast(int, probe["summary_input_elided_chars"]))
    if cell["role"] == ROLE_FIT and elided != 0:
        raise ValueError("the transcript-fitting control unexpectedly elides input")
    if cell["role"] == ROLE_ELIDED and elided <= 0:
        raise ValueError("the elided arm must exceed the window summary-input bound")
