"""Which published compact evidence the summarize-input bound can move -- the bound-specific view.

This is ONE use of the general policy-change audit in `agentic_policy_change_audit`: the field is
`summary_input_cap`, the two values are the retired trigger bound and the shipped window bound, and
the verdict is that module's byte-level prompt-sequence comparison. What this adds is the diagnostic
the restatement reports -- how many chars each bound actually elided from the summarizer's input,
which is WHY the prompts differ where they do, and the size of the cost discount the retired bound
was applying.

Pure replay over the deterministic tool world: no backend, no GPU, and it runs in CI.
"""

from typing import cast

from llb.bench.agentic.context import SUMMARY_INPUT_CAP_TRIGGER, SUMMARY_INPUT_CAP_WINDOW
from llb.bench.agentic_memory_boundary_probe import compact_fold_input_probe
from llb.bench.agentic_policy_change_audit import (
    VERDICT_CHANGED,
    VERDICT_INVARIANT as PROMPT_INVARIANT,
    PolicyChange,
    audit_cell_prompts,
    declared_geometry,
)

SUMMARY_INPUT_CAP_FIELD = "summary_input_cap"

VERDICT_INVARIANT = "bound_invariant"
VERDICT_SENSITIVE = "bound_sensitive"
_VERDICTS = {PROMPT_INVARIANT: VERDICT_INVARIANT, VERDICT_CHANGED: VERDICT_SENSITIVE}


def audit_design(design: dict[str, object], *, study_kind: str) -> list[dict[str, object]]:
    """One audit row per declared cell: what each bound offers the summarizer, and the verdict."""
    held = cast(dict[str, object], design["held_fixed"])
    return [
        _elision_columns(
            audit_cell_prompts(
                cell,
                held,
                PolicyChange.of(
                    SUMMARY_INPUT_CAP_FIELD, SUMMARY_INPUT_CAP_TRIGGER, SUMMARY_INPUT_CAP_WINDOW
                ),
            ),
            held,
        )
        for cell in declared_geometry(design, study_kind)
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


def _elision_columns(row: dict[str, object], held: dict[str, object]) -> dict[str, object]:
    """Name what the two bounds offered the summarizer, and relabel the verdict for this field."""
    probes = {
        bound: compact_fold_input_probe(
            depth=int(cast(int, row["depth"])),
            n_tasks=int(cast(int, held["n_tasks"])),
            max_prompt_chars=int(cast(int, row["max_prompt_chars"])),
            compact_share=float(cast(float, row["compact_share"])),
            summary_input_cap=bound,
            pad_chars=int(cast(int, held["pad_chars"])),
            max_steps_margin=int(cast(int, held["max_steps_margin"])),
            observation_cap_chars=int(cast(int, held["observation_cap_chars"])),
            observation_head_share=float(cast(float, held["observation_head_share"])),
        )
        for bound in (SUMMARY_INPUT_CAP_TRIGGER, SUMMARY_INPUT_CAP_WINDOW)
    }
    trigger, window = probes[SUMMARY_INPUT_CAP_TRIGGER], probes[SUMMARY_INPUT_CAP_WINDOW]
    return {
        **row,
        "summary_input_chars": trigger["summary_input_chars"],
        "trigger_elided_chars": trigger["summary_input_elided_chars"],
        "window_elided_chars": window["summary_input_elided_chars"],
        "n_compactions": trigger["n_compactions"],
        "verdict": _VERDICTS[cast(str, row["verdict"])],
    }


def _geometry(row: dict[str, object]) -> dict[str, object]:
    return {
        key: row[key]
        for key in ("depth", "compact_share", "max_prompt_chars", "trigger_elided_chars")
    }
