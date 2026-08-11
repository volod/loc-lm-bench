"""The repeatedly folding geometry a one-fold invariance verdict does not cover.

Every cap-fitting cell this repo has measured folds EXACTLY ONCE, and that is not an accident of
the tested geometries -- it is a property of the cap-fitting band itself. Trigger hysteresis raises
the trigger to the FULL guard after the first summary
(`episode_prompt.step_prompt`), so a second fold needs the post-fold prompt to cross the whole
guard. A fold replaces at least one transcript entry with a strictly shorter summary line, so the
post-fold prompt sits BELOW the walk's own peak prompt -- and a cap-fitting guard sits above that
peak by construction, imperfect-play margin included
([the imperfect-play safety margin](../../../docs/impl/current/extended-workflows/imperfect-play-margin.md)).
A repeatedly folding cell therefore lives below the cap peak, where the `observation_cap` arm
overflows and no cost delta is measurable.

That is exactly why the regime needs a FIXTURE rather than a study. The published bound-invariance
verdicts were all read on transcripts that fold once, where the first fold sits inside the prefix
the perfect-play and worst-case walks share, so the two walks cannot disagree. This design states
the geometry where they CAN: a cell that folds twice under perfect play and three times when the
controller spends its whole step budget, so the extra entries reach a later summarize input.

The fixture publishes NO number. It is deliberately absent from `AUDITED_DESIGN_PATHS`, so nothing
it declares is evidence a future constant change has to re-run; what it carries is the validity
limit on a statement made elsewhere.
"""

from pathlib import Path
from typing import Any, cast

from llb.bench.agentic.context_policy import POLICY_COMPACT
from llb.bench.agentic_design_fields import as_mapping, as_rows
from llb.bench.agentic_memory_boundary_probe import cap_peak_prompt_chars, compact_fold_input_probe
from llb.bench.agentic_memory_worst_case_probe import worst_case_fold_input_probe
from llb.bench.agentic_policy_change_audit import KIND_TWO_FOLD, PolicyChange
from llb.bench.agentic_policy_change_geometry import declared_geometry, load_audited_design

TWO_FOLD_DESIGN_PATH = "samples/benchmarks/agentic_compact_two_fold_geometry_design.json"

# What makes a cell part of this fixture at all: perfect play must ALREADY fold more than once, or
# the cell is back in the one-fold regime the published verdicts already cover.
MIN_DECLARED_FOLDS = 2
# What every declared cell owes the reader: the measurement it predicts, the reading it predicts,
# and the sentence saying why. A cell missing any of them is a claim nobody can check.
_REQUIRED_CELL_KEYS = ("expected", "predeclared", "why")
_REQUIRED_EXPECTED_KEYS = ("oracle_verdict", "worst_case_verdict", "separates")


def load_two_fold_design(path: Path | str | None = None) -> dict[str, object]:
    """Load the committed two-fold fixture (the studies' own strict JSON loader)."""
    from llb.core.paths import PROJECT_ROOT

    return load_audited_design(PROJECT_ROOT / TWO_FOLD_DESIGN_PATH if path is None else path)


def two_fold_change(design: dict[str, object]) -> PolicyChange:
    """The summarize-bound change the fixture reads its two verdicts over."""
    change = cast(dict[str, dict[str, Any]], design["change"])
    return PolicyChange(baseline=change["baseline"], candidate=change["candidate"])


def two_fold_cells(design: dict[str, object]) -> list[dict[str, object]]:
    """Every declared cell's replayable geometry, in design order."""
    return declared_geometry(design, KIND_TWO_FOLD)


def validate_two_fold_design(design: dict[str, object]) -> None:
    """Refuse a fixture that has drifted out of the regime it exists to cover.

    The two refusals that matter are the two halves of the claim: a cell that folds once is back in
    the regime the published verdicts already hold on, and a cell whose guard clears its cap peak is
    claiming a cap-fitting cell folds repeatedly -- which the hysteresis argument says cannot happen
    and which a real measurement here would therefore be evidence AGAINST, not a fixture to keep.
    """
    if design.get("study_kind") != KIND_TWO_FOLD:
        raise ValueError(f"two-fold fixture study_kind must be {KIND_TWO_FOLD!r}")
    if design.get("publishes_numbers") is not False:
        raise ValueError("the two-fold fixture must declare that it publishes no number")
    change = two_fold_change(design)
    held = as_mapping(design, "held_fixed")
    _check_margin_scaling(design)
    declared = _checked_declarations(design)
    for cell in two_fold_cells(design):
        _check_cell_regime(cell, held, change)
    if not any(bool(as_mapping(row, "expected")["separates"]) for row in declared):
        raise ValueError(
            "the two-fold fixture must declare at least one SEPARATING cell, or it tests nothing "
            "the one-fold regime does not already cover"
        )


def _checked_declarations(design: dict[str, object]) -> list[dict[str, object]]:
    """Every cell states the measurement it predicts, the reading it predicts, and why."""
    declared = as_rows(design, "cells")
    if not declared:
        raise ValueError("the two-fold fixture must declare at least one cell")
    for row in declared:
        missing = [key for key in _REQUIRED_CELL_KEYS if key not in row]
        if missing:
            raise ValueError(f"cell {row.get('cell_id')!r} declares no {missing}")
        absent = [key for key in _REQUIRED_EXPECTED_KEYS if key not in as_mapping(row, "expected")]
        if absent:
            raise ValueError(f"cell {row.get('cell_id')!r} predicts no {absent}")
    return declared


def probe_two_fold_cell(
    cell: dict[str, object], held: dict[str, object], change: PolicyChange
) -> dict[str, object]:
    """Measure both walks over one cell -- fold counts, per-fold offered transcripts, elision.

    Measured through the probe rather than read off the replay digests, so a drifted fixture says
    WHICH claim moved -- a fold count, an offered transcript, an elision -- instead of only that
    some prompt did.
    """
    bound = cast(str, change.baseline[BOUND_FIELD])
    geometry = two_fold_geometry(cell, held)
    oracle = compact_fold_input_probe(summary_input_cap=bound, **geometry)
    worst = worst_case_fold_input_probe(summary_input_cap=bound, **geometry)
    oracle_folds = cast(list[int], oracle["summary_fold_input_chars"])
    worst_folds = cast(list[int], worst["summary_fold_input_chars"])
    return {
        "cap_peak_prompt_chars": cap_peak_prompt_chars(
            **{key: value for key, value in geometry.items() if key not in _PROBE_ONLY}
        ),
        "oracle_folds": int(cast(int, oracle["n_compactions"])),
        "worst_case_folds": int(cast(int, worst["n_compactions"])),
        "oracle_fold_input_chars": oracle_folds,
        "worst_case_fold_input_chars": worst_folds,
        "oracle_elided_chars_under_baseline": int(cast(int, oracle["summary_input_elided_chars"])),
        "worst_case_elided_chars_under_baseline": int(
            cast(int, worst["summary_input_elided_chars"])
        ),
        # Per-fold head-room the stalling walk added, ordinal by ordinal, plus the folds only it
        # reached: this is the margin read on the SUMMARIZE input rather than on the step prompt.
        "fold_input_margin_chars": [
            worst_folds[index] - oracle_folds[index] for index in range(len(oracle_folds))
        ],
        "worst_case_only_folds": max(0, len(worst_folds) - len(oracle_folds)),
    }


def two_fold_geometry(cell: dict[str, object], held: dict[str, object]) -> dict[str, Any]:
    """One cell's task world and guard, in the keywords every probe here takes."""
    return {
        "depth": int(cast(int, cell["depth"])),
        "n_tasks": int(cast(int, held["n_tasks"])),
        "pad_chars": int(cast(int, held["pad_chars"])),
        "max_steps_margin": int(cast(int, held["max_steps_margin"])),
        "observation_cap_chars": int(cast(int, held["observation_cap_chars"])),
        "observation_head_share": float(cast(float, held["observation_head_share"])),
        "max_prompt_chars": int(cast(int, cell["max_prompt_chars"])),
        "compact_share": float(cast(float, cell["compact_share"])),
    }


BOUND_FIELD = "summary_input_cap"
# Keys `compact_fold_input_probe` takes that the cap-peak walk does not: the cap walk has no guard
# and no trigger, which is exactly why its peak is a property of the task world alone.
_PROBE_ONLY = frozenset({"max_prompt_chars", "compact_share"})


def _check_margin_scaling(design: dict[str, object]) -> None:
    scaling = as_mapping(design, "margin_scaling")
    budgets = cast(list[int], scaling.get("max_steps_margins", []))
    if len(budgets) < 2 or sorted(budgets) != budgets or len(set(budgets)) != len(budgets):
        raise ValueError(
            "the margin-scaling read needs at least two distinct step budgets in increasing order, "
            "or it cannot say whether the margin is a constant or a budget"
        )
    if int(cast(int, scaling.get("depth", 0))) < 3 or not scaling.get("why"):
        raise ValueError("the margin-scaling read must name a runnable depth and why it is read")


def _check_cell_regime(
    cell: dict[str, object], held: dict[str, object], change: PolicyChange
) -> None:
    """One cell is in the regime: compact-only, repeatedly folding, and below its cap peak."""
    cell_id = cell["cell_id"]
    if cast(list[str], cell["policies"]) != [POLICY_COMPACT]:
        raise ValueError(
            f"cell {cell_id!r} must declare the compact arm alone: its guard is below the cap peak, "
            "so an observation_cap arm there measures overflow rather than cost"
        )
    if BOUND_FIELD in cast(list[str], cell["pinned_fields"]):
        raise ValueError(f"cell {cell_id!r} pins {BOUND_FIELD!r}, so the change cannot audit it")
    measured = probe_two_fold_cell(cell, held, change)
    peak = int(cast(int, measured["cap_peak_prompt_chars"]))
    guard = int(cast(int, cell["max_prompt_chars"]))
    if guard >= peak:
        raise ValueError(
            f"cell {cell_id!r} guard {guard} clears its {peak}-char cap peak, so it is in the "
            "cap-fitting band -- where hysteresis makes a second fold impossible and this fixture "
            "has nothing to measure"
        )
    if int(cast(int, measured["oracle_folds"])) < MIN_DECLARED_FOLDS:
        raise ValueError(
            f"cell {cell_id!r} folds {measured['oracle_folds']} time(s) under perfect play, below "
            f"the {MIN_DECLARED_FOLDS} this fixture exists to reach"
        )
