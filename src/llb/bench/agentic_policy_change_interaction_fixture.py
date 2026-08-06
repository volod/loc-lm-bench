"""The committed geometry a compound policy change and a per-field reading disagree on.

This module owns the FIXTURE: what the design file must contain to be a counterexample at all, and
the no-model measurement of the geometry it predeclares. The two readings themselves, and the
verdict about whether they separate, live in `agentic_policy_change_interaction`.

The separation rests on an interaction between two constants, not on the size of a change. Two
constants interact when one decides what the other MEANS, and `compact_share` x `summary_input_cap`
is that pair: under the `trigger` bound the summarize call's input cap IS
`compact_share * max_prompt_chars` (`agentic.episode.summary_input_cap_chars`), while under the
`window` bound it does not depend on the share at all. So a guard whose folded transcript sits
BETWEEN the two shares' triggers reads three ways -- the share alone changes nothing, the bound
alone changes nothing, and moving both elides the summarize input.

Three numbers decide that, and the fixture states all three per cell so a drift names the claim that
moved rather than only the digest that changed: the fold step (which must be the SAME at both
shares, or the share alone would already read as changed), the transcript the fold offers the
summarizer, and each share's trigger. `probe_cell_geometry` measures them with an oracle controller
and no model, the same way the cap-fitting designs predeclare their own bounds.

The fixture publishes NO number: it is a geometry design, deliberately absent from
`AUDITED_DESIGN_PATHS`, so nothing here is evidence a future constant change has to re-run.
"""

from pathlib import Path
from typing import Any, cast

from llb.bench.agentic.context import SUMMARY_INPUT_CAP_TRIGGER
from llb.bench.agentic_memory_boundary_probe import (
    cap_prompt_sequence,
    compact_fold_input_probe,
    compaction_trigger_chars,
    first_fold_step,
)
from llb.bench.agentic_policy_change_audit import (
    KIND_INTERACTION,
    PolicyChange,
    declared_geometry,
    load_audited_design,
)
from llb.bench.agentic_policy_change_interaction_terms import FIELD_BOUND, FIELD_SHARE

INTERACTION_DESIGN_PATH = "samples/benchmarks/agentic_policy_change_interaction_design.json"

# The interacting pair: the bound's own value is a function of the share under `trigger`. It is the
# one pair of `AUDITABLE_FIELDS` that separates the two readings at all -- see the enumeration in
# `agentic_policy_change_interaction_couplings` for the other fourteen and what answers them.
INTERACTING_FIELDS = (FIELD_SHARE, FIELD_BOUND)
# What every declared cell owes the reader: the reading it predicts, the geometry that produces it,
# and the sentence saying why. A cell missing any of them is a claim nobody can check.
_REQUIRED_CELL_KEYS = ("expected", "predeclared", "why")


def load_interaction_design(path: Path | str | None = None) -> dict[str, object]:
    """Load the committed interaction fixture (the studies' own strict JSON loader)."""
    from llb.core.paths import PROJECT_ROOT

    return load_audited_design(PROJECT_ROOT / INTERACTION_DESIGN_PATH if path is None else path)


def interaction_change(design: dict[str, object]) -> PolicyChange:
    """The compound change the fixture is built around, as the one change it is."""
    change = cast(dict[str, dict[str, Any]], design["change"])
    return PolicyChange(baseline=change["baseline"], candidate=change["candidate"])


def interaction_cells(design: dict[str, object]) -> list[dict[str, object]]:
    """Every declared cell's replayable geometry, in design order."""
    return declared_geometry(design, KIND_INTERACTION)


def validate_interaction_design(design: dict[str, object]) -> None:
    """Refuse a fixture that could go quiet for a reason other than the seam it guards.

    A cell that declares one of the moved fields is not described by the change at all, and a
    `held_fixed` that disagrees with the baseline would replay a configuration neither reading is
    about -- both would leave the separation untested while the file still looks like a fixture.
    """
    if design.get("study_kind") != KIND_INTERACTION:
        raise ValueError(f"interaction fixture study_kind must be {KIND_INTERACTION!r}")
    change = interaction_change(design)
    if change.fields != INTERACTING_FIELDS:
        raise ValueError(
            f"the interaction fixture must move exactly {INTERACTING_FIELDS}, got {change.fields}"
        )
    held = cast(dict[str, object], design["held_fixed"])
    for field in change.fields:
        if field in held and held[field] != change.baseline[field]:
            raise ValueError(
                f"held_fixed states {field}={held[field]!r} but the change baselines it at "
                f"{change.baseline[field]!r}"
            )
    cells = interaction_cells(design)
    if not cells:
        raise ValueError("the interaction fixture must declare at least one cell")
    for cell in cells:
        own = cast(list[str], cell["pinned_fields"])
        pinned = [field for field in change.fields if field in own]
        if pinned:
            raise ValueError(
                f"cell {cell['cell_id']!r} declares {pinned} itself, so the change cannot audit it"
            )
    for declared in cast(list[dict[str, object]], design["cells"]):
        missing = [key for key in _REQUIRED_CELL_KEYS if key not in declared]
        if missing:
            raise ValueError(f"cell {declared['cell_id']!r} declares no {missing}")
    if not any(row["separates"] for row in _expectations(design)):
        raise ValueError("the interaction fixture must declare at least one SEPARATING cell")


def declared_expectations(design: dict[str, object]) -> dict[str, dict[str, object]]:
    """What the fixture PREDECLARES each cell will show, keyed by cell id.

    Predeclared rather than read off a run: the fixture is the design of an interaction, so a cell
    whose measured reading drifts from its declaration is a finding, not a new expectation.
    """
    return {cast(str, row["cell_id"]): row for row in _expectations(design)}


def probe_cell_geometry(
    cell: dict[str, object], held: dict[str, object], change: PolicyChange
) -> dict[str, int | None]:
    """Measure the geometry the fixture predeclares -- with no model, and no digest.

    Measuring through the probe rather than the replay means a drifted fixture says WHICH claim
    moved (the fold step, the offered transcript, a trigger) instead of only that some prompt did.
    """
    guard = int(cast(int, cell["max_prompt_chars"]))
    shares = (
        float(cast(float, change.baseline[FIELD_SHARE])),
        float(cast(float, change.candidate[FIELD_SHARE])),
    )
    geometry = geometry_kwargs(int(cast(int, cell["depth"])), held)
    sequence = cap_prompt_sequence(**geometry)
    folds = [first_fold_step(sequence, compaction_trigger_chars(guard, share)) for share in shares]
    # Always under the TRIGGER bound: the window bound elides nothing at either share here by
    # construction, so (offered, elided-under-trigger) is what decides how the two shares read.
    baseline, candidate = (
        compact_fold_input_probe(
            max_prompt_chars=guard,
            compact_share=share,
            summary_input_cap=SUMMARY_INPUT_CAP_TRIGGER,
            **geometry,
        )
        for share in shares
    )
    return {
        # None when the two shares fold at DIFFERENT steps, which is a different geometry entirely.
        "fold_step": folds[0] if len(set(folds)) == 1 else None,
        "summary_input_chars": baseline["summary_input_chars"],
        "trigger_chars_at_baseline_share": compaction_trigger_chars(guard, shares[0]),
        "trigger_chars_at_candidate_share": compaction_trigger_chars(guard, shares[1]),
        "elided_chars_at_baseline_share": baseline["summary_input_elided_chars"],
        "elided_chars_at_candidate_share": candidate["summary_input_elided_chars"],
    }


def geometry_kwargs(depth: int, held: dict[str, object]) -> dict[str, Any]:
    """The task world one depth states, in the keywords every probe here takes."""
    return {
        "depth": depth,
        "n_tasks": int(cast(int, held["n_tasks"])),
        "pad_chars": int(cast(int, held["pad_chars"])),
        "max_steps_margin": int(cast(int, held["max_steps_margin"])),
        "observation_cap_chars": int(cast(int, held["observation_cap_chars"])),
        "observation_head_share": float(cast(float, held["observation_head_share"])),
    }


def _expectations(design: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"cell_id": cell["cell_id"], **cast(dict[str, object], cell["expected"])}
        for cell in cast(list[dict[str, object]], design["cells"])
    ]
