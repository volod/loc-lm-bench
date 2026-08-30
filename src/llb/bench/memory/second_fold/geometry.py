"""Where a second-fold cell sits: its task world, its guard, and what perfect play does with it.

Split from the contract that CHECKS those facts because the two cost wildly different things to
call. Every name here RUNS episodes -- one deterministic workflow walk per task per call -- while
`design.py` calls them a handful of times to decide whether a declared cell is in the regime at all,
and the runner and the report read them again per cell. Keeping them apart is what lets an import
line say which cost it is paying.
"""

from pathlib import Path
from typing import cast

from llb.bench.agentic.design_fields import as_float, as_int, as_mapping, as_rows, as_str
from llb.bench.memory.boundary.probe import (
    cap_peak_prompt_chars,
    cap_prompt_sequence,
    compact_fold_input_probe,
)
from llb.bench.memory.fold_step.ladder import compaction_trigger_chars, first_fold_step
from llb.bench.memory.transfer.run import load_transfer_design

DESIGN_PATH = "samples/benchmarks/agentic_compact_second_fold_trigger_design.json"


def load_second_fold_design(path: Path | str | None = None) -> dict[str, object]:
    """Load the committed design through the studies' shared strict JSON loader."""
    from llb.core.paths import PROJECT_ROOT

    return load_transfer_design(PROJECT_ROOT / DESIGN_PATH if path is None else path)


def second_fold_cells(design: dict[str, object]) -> list[dict[str, object]]:
    """Every declared cell, tagged with its family, in declared run order."""
    return [
        {**cell, "family_id": family["family_id"], "depth": family["depth"]}
        for family in as_rows(design, "families")
        for cell in as_rows(family, "cells")
    ]


def cell_geometry(cell: dict[str, object], held: dict[str, object]) -> dict[str, object]:
    """One cell's task world and guard, in the keywords the probes and the runner take."""
    return {
        "depth": as_int(cell, "depth"),
        "n_tasks": as_int(held, "n_tasks"),
        "pad_chars": as_int(held, "pad_chars"),
        "max_steps_margin": as_int(held, "max_steps_margin"),
        "observation_cap_chars": as_int(held, "observation_cap_chars"),
        "observation_head_share": as_float(held, "observation_head_share"),
        "max_prompt_chars": as_int(cell, "max_prompt_chars"),
        "compact_share": as_float(cell, "compact_share"),
        "summary_input_cap": as_str(held, "summary_input_cap"),
    }


def probe_second_fold_cell(cell: dict[str, object], held: dict[str, object]) -> dict[str, object]:
    """Model-free fold count, per-fold summarize input, cost, and cap-peak position."""
    geometry = cell_geometry(cell, held)
    probe = compact_fold_input_probe(**geometry)  # type: ignore[arg-type]
    peak = _cap_peak(as_int(cell, "depth"), held)
    guard = as_int(cell, "max_prompt_chars")
    trigger = compaction_trigger_chars(guard, as_float(cell, "compact_share"))
    return {
        "compaction_trigger_chars": trigger,
        "first_fold_step": first_fold_step(_prompt_sequence(as_int(cell, "depth"), held), trigger),
        "oracle_folds": int(cast(int, probe["n_compactions"])),
        "oracle_fold_input_chars": probe["summary_fold_input_chars"],
        "oracle_model_input_chars": int(cast(int, probe["model_input_prompt_chars"])),
        "cap_peak_prompt_chars": peak,
        "below_cap_peak": guard < peak,
    }


def second_fold_cap_peaks(design: dict[str, object]) -> dict[int, int]:
    """The deterministic cap peak behind every tested depth, for the report's regime line."""
    held = as_mapping(design, "held_fixed")
    return {
        depth: _cap_peak(depth, held)
        for depth in sorted({as_int(family, "depth") for family in as_rows(design, "families")})
    }


def _cap_peak(depth: int, held: dict[str, object]) -> int:
    return cap_peak_prompt_chars(
        depth=depth,
        n_tasks=as_int(held, "n_tasks"),
        pad_chars=as_int(held, "pad_chars"),
        max_steps_margin=as_int(held, "max_steps_margin"),
        observation_cap_chars=as_int(held, "observation_cap_chars"),
        observation_head_share=as_float(held, "observation_head_share"),
    )


def _prompt_sequence(depth: int, held: dict[str, object]) -> list[int]:
    return cap_prompt_sequence(
        depth=depth,
        n_tasks=as_int(held, "n_tasks"),
        pad_chars=as_int(held, "pad_chars"),
        max_steps_margin=as_int(held, "max_steps_margin"),
        observation_cap_chars=as_int(held, "observation_cap_chars"),
        observation_head_share=as_float(held, "observation_head_share"),
    )
