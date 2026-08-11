"""Two-family, stratum-controlled transfer design for unavoidable window elision."""

from pathlib import Path
from typing import Any, cast

from llb.bench.agentic.context_summary import (
    SUMMARY_TRIM_PER_ENTRY_HEAD,
    summary_prompt_overhead_chars,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_design_fields import as_mapping, as_rows
from llb.bench.agentic_memory_boundary_probe import compact_tasks_fold_input_probe
from llb.bench.agentic_memory_fold_step_ladder import compaction_trigger_chars
from llb.bench.agentic_memory_window_elision_design import (
    ARM_ROLES,
    ROLE_ELIDED,
    ROLE_FIT,
    cell_geometry,
)
from llb.bench.agentic_memory_window_elision_tasks import (
    STRATA,
    answer_fact_placement,
    build_window_elision_stratum_tasks,
)
from llb.bench.agentic_policy_change_geometry import load_audited_design

DESIGN_PATH = "samples/benchmarks/agentic_compact_window_elision_transfer_design.json"
STUDY_KIND = "compact_window_elision_transfer"


def load_window_elision_transfer_design(
    path: Path | str | None = None,
) -> dict[str, object]:
    """Load the committed transfer design through the strict shared JSON loader."""
    from llb.core.paths import PROJECT_ROOT

    return load_audited_design(PROJECT_ROOT / DESIGN_PATH if path is None else path)


def transfer_cells(design: dict[str, object]) -> list[dict[str, object]]:
    """Return the fitting control first, then the elided treatment."""
    return sorted(as_rows(design, "cells"), key=lambda cell: cell["role"] != ROLE_FIT)


def transfer_task_records(design: dict[str, object]) -> list[dict[str, Any]]:
    """Build the immutable head/middle/tail task set declared by the design."""
    held = as_mapping(design, "held_fixed")
    stages = {
        key: int(cast(int, value)) for key, value in as_mapping(design, "fact_stages").items()
    }
    return build_window_elision_stratum_tasks(
        n_tasks_per_stratum=int(cast(int, held["n_tasks_per_stratum"])),
        fact_stages=stages,
        depth=int(cast(int, held["depth"])),
        pad_chars=int(cast(int, held["pad_chars"])),
    )


def transfer_tasks(design: dict[str, object]) -> list[AgenticTask]:
    """Typed tasks consumed by the agent loop."""
    return [AgenticTask.from_record(record) for record in transfer_task_records(design)]


def transfer_case_metadata(design: dict[str, object]) -> dict[str, dict[str, object]]:
    """Per-case stratum and stage fields projected into persisted outcomes."""
    return {
        str(record["id"]): {
            "evidence_stratum": record["elision_stratum"],
            "fact_stage": record["fact_stage"],
        }
        for record in transfer_task_records(design)
    }


def probe_transfer_cell(design: dict[str, object], cell: dict[str, object]) -> dict[str, object]:
    """Probe one transfer cell over all stratum tasks without a live model."""
    return _probe_tasks(design, cell, transfer_tasks(design))


def transfer_placements(design: dict[str, object]) -> list[dict[str, object]]:
    """Measure every answer fact against the elided arm's exact trim boundaries."""
    elided = next(cell for cell in transfer_cells(design) if cell["role"] == ROLE_ELIDED)
    probe = probe_transfer_cell(design, elided)
    guard = int(cast(int, elided["max_prompt_chars"]))
    cap = guard - summary_prompt_overhead_chars()
    offered = int(cast(int, probe["summary_input_chars"]))
    return [
        answer_fact_placement(record, offered_chars=offered, transcript_cap_chars=cap)
        for record in transfer_task_records(design)
    ]


def validate_window_elision_transfer_design(design: dict[str, object]) -> None:
    """Require two families, exact paired geometry, and independently checked strata."""
    _validate_header(design)
    probes = [(cell, probe_transfer_cell(design, cell)) for cell in transfer_cells(design)]
    _validate_probes(probes)
    _validate_placements(design)


def _validate_header(design: dict[str, object]) -> None:
    if design.get("study_kind") != STUDY_KIND or int(cast(int, design.get("seed", 0))) < 1:
        raise ValueError("window-elision transfer identity or seed is invalid")
    _validate_roster(design)
    _validate_task_contract(design)
    _validate_cell_contract(design)


def _validate_roster(design: dict[str, object]) -> None:
    required = int(cast(int, design.get("required_qualified_families", 0)))
    roster = as_rows(design, "candidate_roster")
    families = [str(row.get("model_family", "")) for row in roster]
    models = [str(row.get("model", "")) for row in roster]
    if required != 2 or len(roster) < required:
        raise ValueError("window-elision transfer requires two qualified model families")
    if (
        not all(families)
        or len(families) != len(set(families))
        or not all(models)
        or len(models) != len(set(models))
    ):
        raise ValueError("window-elision transfer roster families and models must be unique")
    if any(row.get("backend") != "ollama" for row in roster):
        raise ValueError("window-elision transfer roster must use local Ollama models")


def _validate_task_contract(design: dict[str, object]) -> None:
    held = as_mapping(design, "held_fixed")
    if int(cast(int, held.get("n_tasks_per_stratum", 0))) < 2:
        raise ValueError("window-elision transfer needs at least two tasks per stratum")
    if float(cast(float, held.get("minimum_control_completion", 0.0))) != 1.0:
        raise ValueError("every qualified family must complete every fitting-control task")
    if held.get("preserve_memory_markers") is not True:
        raise ValueError("the transfer must keep shipped typed-marker preservation enabled")
    stages = as_mapping(design, "fact_stages")
    if set(stages) != set(STRATA):
        raise ValueError(f"fact stages must name exactly {STRATA!r}")


def _validate_cell_contract(design: dict[str, object]) -> None:
    cells = transfer_cells(design)
    if len(cells) != 2 or {cell.get("role") for cell in cells} != set(ARM_ROLES):
        raise ValueError(f"transfer cells must contain exactly the roles {ARM_ROLES!r}")
    prototype = as_mapping(design, "prototype")
    if prototype.get("summary_trim_strategy") != SUMMARY_TRIM_PER_ENTRY_HEAD:
        raise ValueError("the conditional prototype must be the entry-aware head strategy")


def _validate_probes(
    probes: list[tuple[dict[str, object], dict[str, object]]],
) -> None:
    triggers = {probe["compaction_trigger_chars"] for _cell, probe in probes}
    inputs = {probe["summary_input_chars"] for _cell, probe in probes}
    folds = {probe["n_compactions"] for _cell, probe in probes}
    if len(triggers) != 1 or len(inputs) != 1 or folds != {1}:
        raise ValueError("transfer arms must hold trigger, offered input, and one fold fixed")
    for cell, probe in probes:
        expected = as_mapping(cell, "expected")
        fields = (
            "compaction_trigger_chars",
            "summary_input_chars",
            "summary_input_elided_chars",
            "n_compactions",
        )
        drifted = [field for field in fields if probe[field] != expected[field]]
        if drifted:
            raise ValueError(f"transfer cell {cell['cell_id']!r} drifted at {drifted[0]}")


def _validate_placements(design: dict[str, object]) -> None:
    placements = transfer_placements(design)
    if any(row["declared_stratum"] != row["measured_stratum"] for row in placements):
        raise ValueError("an answer fact does not occupy its declared elision stratum")
    elided = next(cell for cell in transfer_cells(design) if cell["role"] == ROLE_ELIDED)
    expected = as_mapping(elided, "expected")
    records = transfer_task_records(design)
    for record, task in zip(records, transfer_tasks(design), strict=True):
        probe = _probe_tasks(design, elided, [task])
        if probe["summary_input_chars"] != expected["summary_input_chars"]:
            raise ValueError(f"task {record['id']!r} does not fold the declared transcript bytes")
        if probe["summary_input_elided_chars"] != expected["summary_input_elided_chars"]:
            raise ValueError(f"task {record['id']!r} does not receive the declared elision")


def _probe_tasks(
    design: dict[str, object],
    cell: dict[str, object],
    tasks: list[AgenticTask],
) -> dict[str, object]:
    held = as_mapping(design, "held_fixed")
    geometry = cell_geometry(cell, held)
    guard = int(cast(int, geometry["max_prompt_chars"]))
    share = float(cast(float, geometry["compact_share"]))
    probe = compact_tasks_fold_input_probe(
        tasks,
        max_steps=int(cast(int, held["depth"])) + int(cast(int, held["max_steps_margin"])),
        max_prompt_chars=guard,
        compact_share=share,
        summary_input_cap=str(cast(str, geometry["summary_input_cap"])),
        observation_cap_chars=int(cast(int, geometry["observation_cap_chars"])),
        observation_head_share=float(cast(float, geometry["observation_head_share"])),
    )
    return {**probe, "compaction_trigger_chars": compaction_trigger_chars(guard, share)}
