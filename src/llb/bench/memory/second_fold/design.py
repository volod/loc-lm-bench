"""Design contract for the second-fold trigger restatement: what a family must be before it runs.

Two rules replace the collapse study's cap-fitting cell gate, which by construction refuses every
cell in this regime. First, every cell must sit BELOW its depth's cap peak -- a guard above it is
cap-fitting, and a cap-fitting cell cannot fold twice, so declaring one here would measure the
one-fold claim again under a new name. Second, every cell must fold at least twice under perfect
play, which is the regime the study exists in. Both are decided by the deterministic probe with no
model, so a design that could not answer the question is refused before a GPU is warmed.
"""

from typing import cast

from llb.bench.agentic.context_policy import POLICY_COMPACT
from llb.bench.agentic.design_fields import as_float, as_int, as_mapping, as_rows, as_str
from llb.bench.memory.fold_step.ladder import compaction_trigger_chars
from llb.bench.memory.second_fold.geometry import probe_second_fold_cell
from llb.bench.memory.second_fold.reading import (
    EQUIVALENCE_METRIC,
    EQUIVALENCE_RULE,
    KIND_EQUAL_GUARD,
    KIND_EQUAL_TRIGGER,
    MIN_REPEATED_FOLDS,
    REPORTING_CONFIDENCE,
    STUDY_KIND,
)
from llb.rag.fusion_evidence.evidence_gate import minimum_discordant_pairs


def validate_second_fold_design(design: dict[str, object]) -> None:
    """Refuse a grid that is out of the regime, underpowered, or cannot see a trigger change."""
    if design.get("schema_version") != 1 or design.get("study_kind") != STUDY_KIND:
        raise ValueError("second-fold design schema or study kind is invalid")
    if as_float(design, "reporting_confidence") != REPORTING_CONFIDENCE:
        raise ValueError(f"second-fold reporting confidence must be {REPORTING_CONFIDENCE}")
    if cast(list[str], design.get("policies", [])) != [POLICY_COMPACT]:
        raise ValueError(
            "the second-fold study must declare the compact arm alone: every cell sits below its "
            "cap peak, where an observation_cap arm measures overflow rescue rather than cost"
        )
    if as_int(design, "seed") < 1:
        raise ValueError("the second-fold study needs a positive deterministic seed")
    held = as_mapping(design, "held_fixed")
    _validate_held(held)
    _validate_control(as_mapping(design, "control_recheck"))
    families = as_rows(design, "families")
    _validate_families(families, held)
    _validate_equivalence(as_mapping(design, "equivalence"), families)


def _validate_held(held: dict[str, object]) -> None:
    if not held.get("model") or not held.get("model_family") or held.get("backend") != "ollama":
        raise ValueError("the second-fold study must pin one local Ollama model family")
    if as_int(held, "n_tasks") < minimum_discordant_pairs(REPORTING_CONFIDENCE):
        raise ValueError("second-fold cells cannot reach the 97.5% exact-sign-test floor")
    if not 0.0 <= as_float(held, "minimum_cell_completion", -1.0) <= 1.0:
        raise ValueError("second-fold cell completion floor is invalid")
    if as_int(held, "minimum_measured_folds") < MIN_REPEATED_FOLDS:
        raise ValueError(
            f"a measured-fold floor below {MIN_REPEATED_FOLDS} admits cells back into the one-fold "
            "regime the trigger collapse already covers"
        )


def _validate_control(control: dict[str, object]) -> None:
    threshold = as_float(control, "minimum_completion_rate")
    if as_int(control, "n_tasks") < 1 or as_int(control, "depth") < 3 or not 0.0 < threshold <= 1.0:
        raise ValueError("second-fold control recheck contract is invalid")


def _validate_families(families: list[dict[str, object]], held: dict[str, object]) -> None:
    ids = [as_str(family, "family_id") for family in families]
    kinds = [as_str(family, "kind") for family in families]
    if len(families) < 2 or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("the second-fold study needs at least two uniquely named families")
    if kinds.count(KIND_EQUAL_TRIGGER) < 1 or kinds.count(KIND_EQUAL_GUARD) != 1:
        raise ValueError("the second-fold study needs an equal-trigger family and one contrast")
    seen: set[str] = set()
    for family in families:
        seen |= _validate_family(family, held, seen)
    _validate_repeat_geometry(families)
    _validate_window(held, families)


def _validate_family(
    family: dict[str, object], held: dict[str, object], seen: set[str]
) -> set[str]:
    """One family's cells: uniquely named, inside the regime, and on the family's own axis."""
    cells = [{**cell, "depth": family["depth"]} for cell in as_rows(family, "cells")]
    cell_ids = [as_str(cell, "cell_id") for cell in cells]
    if len(cells) < 2 or not all(cell_ids) or seen & set(cell_ids):
        raise ValueError(f"family {family.get('family_id')} needs two or more unique cells")
    for cell in cells:
        _validate_cell_regime(cell, held)
    _validate_family_axes(family, cells, held)
    return set(cell_ids)


def _validate_cell_regime(cell: dict[str, object], held: dict[str, object]) -> None:
    """One cell is below its cap peak, folds repeatedly, and measures what it predeclared."""
    cell_id = as_str(cell, "cell_id")
    measured = probe_second_fold_cell(cell, held)
    if not measured["below_cap_peak"]:
        raise ValueError(
            f"cell {cell_id!r} guard {cell['max_prompt_chars']} clears its "
            f"{measured['cap_peak_prompt_chars']}-char cap peak, so it is cap-fitting -- where "
            "hysteresis makes a second fold impossible and this study has nothing to restate"
        )
    if int(cast(int, measured["oracle_folds"])) < MIN_REPEATED_FOLDS:
        raise ValueError(
            f"cell {cell_id!r} folds {measured['oracle_folds']} time(s) under perfect play, below "
            f"the {MIN_REPEATED_FOLDS} this study exists to measure"
        )
    declared = as_mapping(cell, "predeclared")
    if not declared:
        raise ValueError(f"cell {cell_id!r} predeclares no measurable geometry")
    unknown = sorted(key for key in declared if key not in measured)
    if unknown:
        raise ValueError(
            f"cell {cell_id!r} predeclares {unknown}, which the probe does not measure, so nothing "
            f"would ever check it; it measures {sorted(measured)}"
        )
    drifted = {key: measured[key] for key, value in declared.items() if measured[key] != value}
    if drifted:
        raise ValueError(
            f"cell {cell_id!r} predeclares {dict(declared)} but the geometry now measures {drifted}"
        )
    if "separates_from_anchor" not in as_mapping(cell, "expected"):
        raise ValueError(f"cell {cell_id!r} predicts no separation from its family anchor")


def _validate_family_axes(
    family: dict[str, object], cells: list[dict[str, object]], held: dict[str, object]
) -> None:
    """An equal-trigger family holds the trigger and moves the guard; the contrast does neither."""
    family_id = family.get("family_id")
    shares = [as_float(cell, "compact_share") for cell in cells]
    guards = [as_int(cell, "max_prompt_chars") for cell in cells]
    if as_int(family, "depth") < 3 or any(
        not 0.0 < share <= 1.0 or guard <= 0 for share, guard in zip(shares, guards, strict=True)
    ):
        raise ValueError(f"family {family_id} geometry is invalid")
    if len(set(shares)) != len(shares):
        raise ValueError(f"family {family_id} must move compact_share")
    triggers = [
        compaction_trigger_chars(guard, share) for share, guard in zip(shares, guards, strict=True)
    ]
    if as_str(family, "kind") == KIND_EQUAL_TRIGGER:
        if len(set(triggers)) != 1 or len(set(guards)) != len(guards):
            raise ValueError(f"family {family_id} must hold ONE trigger while moving the guard")
        return
    if len(set(guards)) != 1:
        raise ValueError(f"family {family_id} must hold the guard while moving the trigger")
    _validate_contrast_steps(family_id, cells, held)


def _validate_contrast_steps(
    family_id: object, cells: list[dict[str, object]], held: dict[str, object]
) -> None:
    """The contrast must move the FIRST FOLD STEP, not merely the trigger.

    Two triggers inside one step's interval fold at the same step and produce the identical
    transcript, so a contrast built from them would report "no resolving power" about the study
    rather than about the measurement -- the mechanism the collapse established is that the trigger
    reaches the transcript only by choosing which step folds.
    """
    steps = [probe_second_fold_cell(cell, held)["first_fold_step"] for cell in cells[1:]]
    anchor_step = probe_second_fold_cell(cells[0], held)["first_fold_step"]
    if len(set(steps)) != len(steps) or anchor_step in steps:
        raise ValueError(
            f"family {family_id} members fold at first steps {[anchor_step, *steps]}, so at least "
            "two of them produce the identical transcript and the contrast tests nothing"
        )


def _validate_repeat_geometry(families: list[dict[str, object]]) -> None:
    """Exactly one cell re-runs an anchor's geometry, which is the study's own noise floor."""
    cells = [cell for family in families for cell in as_rows(family, "cells")]
    anchors = {as_str(as_rows(family, "cells")[0], "cell_id") for family in families}
    repeats = [cell for cell in cells if cell.get("repeats_anchor")]
    if len(repeats) != 1:
        raise ValueError(
            "the second-fold study needs exactly one cell repeating an anchor geometry, or a "
            "measured spread has no noise floor to be read against"
        )
    repeat = repeats[0]
    target = as_str(repeat, "repeats_anchor")
    anchor = next((cell for cell in cells if as_str(cell, "cell_id") == target), None)
    if anchor is None or target not in anchors:
        raise ValueError(
            f"cell {repeat['cell_id']!r} repeats {target!r}, which is no family anchor"
        )
    if (repeat["compact_share"], repeat["max_prompt_chars"]) != (
        anchor["compact_share"],
        anchor["max_prompt_chars"],
    ):
        raise ValueError(
            f"cell {repeat['cell_id']!r} claims to repeat {target!r} but declares another geometry"
        )


def _validate_window(held: dict[str, object], families: list[dict[str, object]]) -> None:
    from llb.backends.context_fit import CHARS_PER_TOKEN, PROMPT_HEADROOM_TOKENS

    widest = max(
        as_int(cell, "max_prompt_chars") for family in families for cell in as_rows(family, "cells")
    )
    required = int(widest / CHARS_PER_TOKEN) + PROMPT_HEADROOM_TOKENS
    if as_int(held, "max_model_len") < required:
        raise ValueError(
            f"declared max_model_len cannot carry the widest guard ({widest} chars needs about "
            f"{required} tokens)"
        )


def _validate_equivalence(rule: dict[str, object], families: list[dict[str, object]]) -> None:
    fraction = as_float(rule, "anchor_cost_fraction")
    contrasts = {
        as_str(family, "family_id")
        for family in families
        if as_str(family, "kind") == KIND_EQUAL_GUARD
    }
    if (
        rule.get("rule") != EQUIVALENCE_RULE
        or rule.get("metric") != EQUIVALENCE_METRIC
        or rule.get("requires_same_anchor_geometry") is not True
        or not 0.0 < fraction <= 0.1
        or rule.get("contrast_family") not in contrasts
    ):
        raise ValueError("the second-fold study must predeclare its band and its contrast family")
