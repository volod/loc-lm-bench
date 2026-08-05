"""Which published compact evidence the summarize-input bound can move -- decided with no model.

Every cap-fitting study before the step-aligned bound was measured with the summarize call's input
capped at the compaction trigger, and a trimmed summarize input is a SMALLER prompt: the retired
bound discounted compact's own measured cost, one-sidedly in compact's favor, at exactly the cells a
crossover is read from. Re-running everything to find out where that mattered would be the expensive
answer. The cheap one is exact: the tool world is deterministic, so
`compact_fold_input_probe` says per cell what each bound offers the summarizer and how much it
elides, and a cell that elides NOTHING under either bound sends bit-identical prompts under both --
its published cost stands with no run at all.

This module extracts cell geometry from the committed designs (each study nests its cells
differently), probes both bounds, and splits the published evidence into bound-invariant cells and
the few that must be re-measured. It is pure arithmetic over the deterministic probe: no backend, no
GPU, and it runs in CI.
"""

from pathlib import Path
from typing import cast

from llb.bench.agentic.context import SUMMARY_INPUT_CAP_TRIGGER, SUMMARY_INPUT_CAP_WINDOW
from llb.bench.agentic_memory_boundary_probe import compact_fold_input_probe

# How each committed study nests its cells under its own design root.
KIND_SURFACE = "compact_memory_boundary_surface"
KIND_COLLAPSE = "compact_trigger_guard_collapse"
KIND_FOLD_STEP = "compact_fold_step_crossover"
AUDITED_KINDS = (KIND_SURFACE, KIND_COLLAPSE, KIND_FOLD_STEP)

VERDICT_INVARIANT = "bound_invariant"
VERDICT_SENSITIVE = "bound_sensitive"


def audit_design(design: dict[str, object], *, study_kind: str) -> list[dict[str, object]]:
    """One audit row per declared cell: what each bound offers the summarizer, and the verdict."""
    held = _held_fixed(design, study_kind)
    return [_audit_cell(cell, held) for cell in declared_geometry(design, study_kind)]


def declared_geometry(design: dict[str, object], study_kind: str) -> list[dict[str, object]]:
    """Every declared cell's `(cell_id, depth, compact_share, max_prompt_chars)`, in design order.

    The three studies nest cells differently -- a flat grid, families, and per-depth step ladders --
    so the shape is read per kind rather than guessed at.
    """
    held = _held_fixed(design, study_kind)
    # The collapse study SWEEPS the share, so it states one per cell and holds none fixed.
    default_share = held.get("compact_share")
    # `(depth, group)`: the surface states depth per cell, the others state it on the group.
    groups: list[tuple[int | None, dict[str, object]]]
    if study_kind == KIND_SURFACE:
        groups = [(None, cast(dict[str, object], design["surface"]))]
    elif study_kind == KIND_COLLAPSE:
        groups = [
            (int(cast(int, family["depth"])), family)
            for family in cast(list[dict[str, object]], design["families"])
        ]
    elif study_kind == KIND_FOLD_STEP:
        groups = [
            (int(cast(int, ladder["depth"])), step)
            for ladder in cast(list[dict[str, object]], design["ladders"])
            for step in cast(list[dict[str, object]], ladder["steps"])
        ]
    else:
        raise ValueError(f"no cell geometry is known for study kind {study_kind!r}")
    return [
        {
            "cell_id": cast(str, cell["cell_id"]),
            "depth": int(cast(int, cell.get("depth", depth))),
            "compact_share": _share(cell, default_share),
            "max_prompt_chars": int(cast(int, cell["max_prompt_chars"])),
        }
        for depth, group in groups
        for cell in cast(list[dict[str, object]], group["cells"])
    ]


def sensitive_cell_ids(rows: list[dict[str, object]]) -> list[str]:
    """The cells whose published cost was measured under a bound that changed their prompts."""
    return [cast(str, row["cell_id"]) for row in rows if row["verdict"] == VERDICT_SENSITIVE]


def audit_summary(audits: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    """Roll every audited study into counts plus the named cells that still need a run."""
    rows = [row for study_rows in audits.values() for row in study_rows]
    return {
        "n_cells": len(rows),
        "n_bound_invariant": sum(row["verdict"] == VERDICT_INVARIANT for row in rows),
        "n_bound_sensitive": sum(row["verdict"] == VERDICT_SENSITIVE for row in rows),
        "sensitive": [
            {"study_kind": kind, "cell_id": row["cell_id"], **_geometry(row)}
            for kind, study_rows in audits.items()
            for row in study_rows
            if row["verdict"] == VERDICT_SENSITIVE
        ],
    }


def load_audited_design(path: Path | str) -> dict[str, object]:
    """Load one committed study design for auditing (the studies' own strict JSON loader)."""
    from llb.bench.agentic_memory_transfer import load_transfer_design

    return load_transfer_design(path)


def _audit_cell(cell: dict[str, object], held: dict[str, object]) -> dict[str, object]:
    probes = {
        bound: compact_fold_input_probe(
            depth=int(cast(int, cell["depth"])),
            n_tasks=int(cast(int, held["n_tasks"])),
            max_prompt_chars=int(cast(int, cell["max_prompt_chars"])),
            compact_share=float(cast(float, cell["compact_share"])),
            summary_input_cap=bound,
            pad_chars=int(cast(int, held["pad_chars"])),
            max_steps_margin=int(cast(int, held["max_steps_margin"])),
            observation_cap_chars=int(cast(int, held["observation_cap_chars"])),
            observation_head_share=float(cast(float, held["observation_head_share"])),
        )
        for bound in (SUMMARY_INPUT_CAP_TRIGGER, SUMMARY_INPUT_CAP_WINDOW)
    }
    trigger, window = probes[SUMMARY_INPUT_CAP_TRIGGER], probes[SUMMARY_INPUT_CAP_WINDOW]
    # The bound reaches the run ONLY through the summarize prompt. Identical elision means identical
    # summarize input, which means identical summary, which means identical later prompts.
    invariant = (
        trigger["summary_input_elided_chars"] == window["summary_input_elided_chars"]
        and trigger["summary_input_chars"] == window["summary_input_chars"]
    )
    return {
        **cell,
        "summary_input_chars": trigger["summary_input_chars"],
        "trigger_elided_chars": trigger["summary_input_elided_chars"],
        "window_elided_chars": window["summary_input_elided_chars"],
        "n_compactions": trigger["n_compactions"],
        "verdict": VERDICT_INVARIANT if invariant else VERDICT_SENSITIVE,
    }


def _share(cell: dict[str, object], default: object) -> float:
    share = cell.get("compact_share", default)
    if share is None:
        raise ValueError(f"cell {cell.get('cell_id')!r} states no compact_share and none is held")
    return float(cast(float, share))


def _geometry(row: dict[str, object]) -> dict[str, object]:
    return {
        key: row[key]
        for key in ("depth", "compact_share", "max_prompt_chars", "trigger_elided_chars")
    }


def _held_fixed(design: dict[str, object], study_kind: str) -> dict[str, object]:
    if study_kind not in AUDITED_KINDS:
        raise ValueError(
            f"{study_kind!r} is not an audited study kind; choose from {AUDITED_KINDS}"
        )
    return cast(dict[str, object], design["held_fixed"])
