"""What the repeatedly folding geometry says about the reach of a bound-invariance verdict.

The fixture (`agentic_memory_two_fold_fixture`) states the geometry; this reads it. Each cell is
audited TWICE over the same summarize-bound change -- once under the study oracle, once under the
controller that spends the whole step budget -- and the reading is about the cells where those two
verdicts DISAGREE. A cell the oracle calls bound-invariant and imperfect play calls bound-sensitive
is the whole finding: on that transcript the published invariance statement is true of the walk it
was read on and false of the walk a real controller can produce.

The margin is re-read here too, on the axis the one-fold evidence could not separate. The
imperfect-play peak margin looked like a constant because every study that measured it held
`max_steps_margin` at 4; reading it at several step budgets says whether it is a constant or a
per-wasted-step price, and the per-fold summarize margin says the same thing on the summarizer's
input rather than on the step prompt.
"""

from typing import cast

from llb.bench.agentic.design_fields import as_mapping, as_rows
from llb.bench.memory.cap_audit import VERDICT_INVARIANT, VERDICT_SENSITIVE
from llb.bench.memory.two_fold.fixture import (
    probe_two_fold_cell,
    two_fold_cells,
    two_fold_change,
)
from llb.bench.memory.worst_case_probe import cap_peak_margin
from llb.bench.policy_change.audit import (
    VERDICT_CHANGED,
    VERDICT_INVARIANT as PROMPT_INVARIANT,
    PolicyChange,
    audit_cell_prompts,
)
from llb.bench.policy_change.tasks import worst_case_replay_controller

_VERDICTS = {PROMPT_INVARIANT: VERDICT_INVARIANT, VERDICT_CHANGED: VERDICT_SENSITIVE}

READING_ONE_FOLD_ONLY = "invariance_holds_for_one_fold_only"
READING_EXTENDS = "invariance_extends_to_repeated_compaction"
READING_DRIFTED = "the_declared_geometry_no_longer_measures_what_it_declares"


def two_fold_row(
    cell: dict[str, object], held: dict[str, object], change: PolicyChange
) -> dict[str, object]:
    """One cell: both verdicts, the geometry behind them, and whether they separate."""
    oracle = audit_cell_prompts(cell, held, change)
    worst = audit_cell_prompts(cell, held, change, controller=worst_case_replay_controller)
    oracle_verdict = _VERDICTS[cast(str, oracle["verdict"])]
    worst_verdict = _VERDICTS[cast(str, worst["verdict"])]
    return {
        "cell_id": cell["cell_id"],
        "depth": cell["depth"],
        "compact_share": cell["compact_share"],
        "max_prompt_chars": cell["max_prompt_chars"],
        **probe_two_fold_cell(cell, held, change),
        "oracle_verdict": oracle_verdict,
        "worst_case_verdict": worst_verdict,
        "worst_case_first_divergent_step": worst["first_divergent_step"],
        "separates": oracle_verdict == VERDICT_INVARIANT and worst_verdict == VERDICT_SENSITIVE,
    }


def declaration_drift(row: dict[str, object], declared: dict[str, object]) -> list[str]:
    """Every predeclared field the measurement no longer agrees with, named one by one.

    Predeclared rather than read off the run: the fixture is the design of a regime, so a cell whose
    measured geometry moves is a finding about the runtime and not a new expectation.
    """
    predeclared = as_mapping(declared, "predeclared")
    expected = as_mapping(declared, "expected")
    return [
        f"{field}: declared {value!r}, measured {row[field]!r}"
        for field, value in (*predeclared.items(), *expected.items())
        if field in row and row[field] != value
    ]


def margin_scaling(design: dict[str, object]) -> list[dict[str, object]]:
    """The imperfect-play peak margin at each declared step budget, and its per-step price.

    One row per budget, because the question is whether the margin is a number or a rate. It is a
    rate: each budgeted extra step buys the controller one more wasted transcript entry, so a study
    that widens its step budget widens the head-room its guards must carry by the same arithmetic.
    """
    scaling = as_mapping(design, "margin_scaling")
    held = as_mapping(design, "held_fixed")
    depth = int(cast(int, scaling["depth"]))
    rows = []
    for budget in cast(list[int], scaling["max_steps_margins"]):
        margin = cap_peak_margin(
            depth=depth,
            n_tasks=int(cast(int, held["n_tasks"])),
            pad_chars=int(cast(int, held["pad_chars"])),
            max_steps_margin=int(budget),
            observation_cap_chars=int(cast(int, held["observation_cap_chars"])),
            observation_head_share=float(cast(float, held["observation_head_share"])),
        )
        steps = int(cast(int, margin["budgeted_extra_steps"]))
        rows.append(
            {
                "max_steps_margin": int(budget),
                **margin,
                "margin_chars_per_extra_step": int(cast(int, margin["margin_chars"])) / steps,
            }
        )
    return rows


def analyze_two_fold(design: dict[str, object]) -> dict[str, object]:
    """Audit every declared cell under both walks, read the margin, and state the validity limit."""
    change = two_fold_change(design)
    held = as_mapping(design, "held_fixed")
    declared = {cast(str, row["cell_id"]): row for row in as_rows(design, "cells")}
    rows = [
        {**row, "declaration_drift": declaration_drift(row, declared[cast(str, row["cell_id"])])}
        for row in (two_fold_row(cell, held, change) for cell in two_fold_cells(design))
    ]
    scaling = margin_scaling(design)
    reading, reason = two_fold_reading(rows, scaling)
    return {
        "study_id": design["study_id"],
        "study_kind": design["study_kind"],
        "held_fixed": held,
        "change": {"baseline": dict(change.baseline), "candidate": dict(change.candidate)},
        "cells": rows,
        "separating_cell_ids": [
            cast(str, row["cell_id"]) for row in rows if bool(row["separates"])
        ],
        "margin_scaling": scaling,
        "reading": reading,
        "reason": reason,
        "changes_shipped_default": False,
    }


def two_fold_reading(
    rows: list[dict[str, object]], scaling: list[dict[str, object]]
) -> tuple[str, str]:
    """The validity statement the fixture exists to produce."""
    drifted = [row for row in rows if row["declaration_drift"]]
    if drifted:
        named = "; ".join(
            f"{row['cell_id']}: {', '.join(cast(list[str], row['declaration_drift']))}"
            for row in drifted
        )
        return READING_DRIFTED, f"a declared cell no longer measures what it declares -- {named}"
    separating = [cast(str, row["cell_id"]) for row in rows if bool(row["separates"])]
    rates = sorted({float(cast(float, row["margin_chars_per_extra_step"])) for row in scaling})
    rate = (
        f"the peak margin is {rates[0]:.0f} chars per budgeted extra step across "
        f"{len(scaling)} step budgets, so it scales with the budget rather than being one number"
        if len(rates) == 1
        else f"the peak margin costs {rates} chars per extra step across the read budgets"
    )
    if not separating:
        return (
            READING_EXTENDS,
            "no repeatedly folding cell separates the two walks, so the bound-invariance verdict "
            f"read under perfect play also holds for the worst case the step budget allows; {rate}",
        )
    return (
        READING_ONE_FOLD_ONLY,
        f"{len(separating)}/{len(rows)} repeatedly folding cells ({', '.join(separating)}) are "
        "bound-invariant under perfect play and bound-sensitive once the controller spends its "
        "whole step budget, so a bound-invariance verdict read on a one-fold transcript states "
        f"nothing about a repeatedly folding one; {rate}",
    )
