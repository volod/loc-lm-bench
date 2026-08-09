"""Reading one policy change TWO ways, so the compound guarantee is a tested property.

The policy-change audit replays two WHOLE policies, so a commit that moves several constants gets
one verdict computed between the two configurations that actually existed
(`agentic_policy_change_audit`). That is right by construction -- and on the 22 published cells
nothing tests it, because there no constructible change makes the compound answer differ from the
union of the per-field ones. A guarantee nothing exercises is a guarantee a refactor can delete:
collapse the compound replay back to a field-at-a-time loop and every test stays green.

So this module reads the committed interaction geometry
(`agentic_policy_change_interaction_fixture`) both ways and reports where they DISAGREE:

  - the compound reading -- one `PolicyChange` carrying every moved field, which is what the audit
    and the CI pin gate do today;
  - the per-field union -- each field audited alone with the others left at the cell's geometry and
    the shipped defaults, then the strongest verdict and earliest step taken across them, which is
    what an implementation without `PolicyChange` reports.

On the fixture's separating cells the union says "invalidates nothing" and the compound reading says
"invalidated", and the compound one is the true statement about the commit: replaying the whole
candidate policy really does send different prompts there.
`tests/llb/bench/test_agentic_policy_change_interaction.py` runs this in `make ci` and fails the
moment the two readings agree everywhere again.
"""

from typing import cast

from llb.bench.agentic_policy_change_audit import (
    VERDICT_CHANGED,
    VERDICT_INVARIANT,
    VERDICT_NOT_APPLICABLE,
    PolicyChange,
    audit_cell_prompts,
)
from llb.bench.agentic_policy_change_interaction_fixture import (
    interaction_cells,
    interaction_change,
    validate_interaction_design,
)

# The two readings the fixture holds apart: the change replayed as ONE change, and the union of the
# per-field audits a collapsed implementation would report instead.
READING_COMPOUND = "compound"
READING_PER_FIELD = "per_field_union"
# Why a cell separates: the two readings disagree about WHETHER the cell is invalidated, or about
# WHERE the change first bites. Either one is a fact the per-field reading gets wrong.
SEPARATES_ON_VERDICT = "verdict"
SEPARATES_ON_FIRST_DIVERGENT_STEP = "first_divergent_step"


def per_field_union(
    cell: dict[str, object], held: dict[str, object], change: PolicyChange
) -> dict[str, object]:
    """What a field-at-a-time audit would report for one cell -- the reading being ruled out.

    Each field is replayed ALONE, which is what an implementation without `PolicyChange` does: the
    untouched constants keep the cell's declared geometry and the shipped defaults on BOTH sides, so
    the two arms differ in one field only. The union then takes the strongest verdict any field
    reports and the earliest step any of them names -- the most generous reading of a per-field
    audit, so a separation cannot be an artifact of how its rows were combined.
    """
    fields = {
        field: audit_cell_prompts(cell, held, PolicyChange.of(field, base, change.candidate[field]))
        for field, base in change.baseline.items()
    }
    rows = list(fields.values())
    return {
        "reading": READING_PER_FIELD,
        "verdict": _union_verdict({cast(str, row["verdict"]) for row in rows}),
        "changed_arms": sorted(
            {arm for row in rows for arm in cast(list[str], row["changed_arms"])}
        ),
        "first_divergent_step": min(
            (
                cast(int, row["first_divergent_step"])
                for row in rows
                if row["first_divergent_step"] is not None
            ),
            default=None,
        ),
        "per_field": {field: _reading(row) for field, row in fields.items()},
    }


def audit_interaction_cell(
    cell: dict[str, object], held: dict[str, object], change: PolicyChange
) -> dict[str, object]:
    """Both readings of one cell, and whether they disagree."""
    compound = audit_cell_prompts(cell, held, change)
    union = per_field_union(cell, held, change)
    reasons = []
    if compound["verdict"] != union["verdict"]:
        reasons.append(SEPARATES_ON_VERDICT)
    if compound["first_divergent_step"] != union["first_divergent_step"]:
        reasons.append(SEPARATES_ON_FIRST_DIVERGENT_STEP)
    return {
        "cell_id": cell["cell_id"],
        "depth": cell["depth"],
        "max_prompt_chars": cell["max_prompt_chars"],
        READING_COMPOUND: {"reading": READING_COMPOUND, **_reading(compound)},
        READING_PER_FIELD: union,
        "separates": bool(reasons),
        "separates_on": reasons,
    }


def audit_interaction_design(design: dict[str, object]) -> list[dict[str, object]]:
    """Read every declared cell both ways, in design order."""
    validate_interaction_design(design)
    change = interaction_change(design)
    held = cast(dict[str, object], design["held_fixed"])
    return [audit_interaction_cell(cell, held, change) for cell in interaction_cells(design)]


def separation_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """The one-line answer: does the committed geometry still hold the two readings apart?"""
    separating = [cast(str, row["cell_id"]) for row in rows if row["separates"]]
    return {
        "n_cells": len(rows),
        "n_separating": len(separating),
        "separating_cells": separating,
        "separates": bool(separating),
        "reasons": sorted(
            {reason for row in rows for reason in cast(list[str], row["separates_on"])}
        ),
    }


def _reading(row: dict[str, object]) -> dict[str, object]:
    """The comparable part of an audit row -- what the two readings must agree on to be the same."""
    return {
        "verdict": row["verdict"],
        "changed_arms": row["changed_arms"],
        "first_divergent_step": row["first_divergent_step"],
    }


def _union_verdict(verdicts: set[str]) -> str:
    """The strongest verdict any single field reports: changed beats invariant beats pinned."""
    if VERDICT_CHANGED in verdicts:
        return VERDICT_CHANGED
    if verdicts == {VERDICT_NOT_APPLICABLE}:
        return VERDICT_NOT_APPLICABLE
    return VERDICT_INVARIANT
