"""Study identity, vocabulary, and family-level reading for the trigger/guard collapse.

The equivalence question is not "do two intervals overlap": the paired intervals here are far
tighter than the difference an operator would act on, so overlap would reject a practical
equivalence over a few tokens. It is stated instead on the scale the operator pays -- a family's
SPREAD of compact-minus-cap total model-input tokens against a predeclared fraction of what the cap
baseline costs at that depth. The equal-guard family is the positive control: if MOVING the trigger
does not clear the same band, an equal-trigger agreement measures nothing and the study says so.
"""

from typing import cast

from llb.bench.agentic.context_budget import fixed_budget
from llb.bench.agentic_memory_boundary_probe import first_fold_step

STUDY_KIND = "compact_trigger_guard_collapse"
METHOD = "agentic-compact-trigger-guard-collapse"
REPORTING_CONFIDENCE = 0.975

KIND_EQUAL_TRIGGER = "equal_trigger"
KIND_EQUAL_GUARD = "equal_guard"
EQUIVALENCE_RULE = "spread_within_cap_cost_fraction"
EQUIVALENCE_METRIC = "compact_minus_cap_total_model_input_tokens"

READING_INELIGIBLE = "pinned_family_control_ineligible"
READING_INVALID = "collapse_cells_invalid"
READING_NO_POWER = "no_resolving_power"
READING_COLLAPSES = "trigger_ratio_collapses"
READING_INTERACTS = "share_and_guard_interact"


def compaction_trigger_chars(max_prompt_chars: int, compact_share: float) -> int:
    """The trigger the runtime will compute, taken from the runtime's own arithmetic."""
    return fixed_budget(max_prompt_chars).compaction_trigger_chars(compact_share)


def annotate_fold_steps(
    cells: list[dict[str, object]], prompt_sequences: dict[int, list[int]]
) -> list[dict[str, object]]:
    """Attach the trigger and the deterministic step it folds at to every measured cell.

    The fold step is the mechanism the collapse claim rests on: the trigger reaches the transcript
    only by deciding WHICH step compacts, so two pairs with the same trigger are expected to be
    bit-identical, and two triggers inside the same step interval are too.
    """
    annotated: list[dict[str, object]] = []
    for cell in cells:
        trigger = _trigger(cell)
        sequence = prompt_sequences[int(cast(int, cell["depth"]))]
        annotated.append(
            {
                **cell,
                "compaction_trigger_chars": trigger,
                "predicted_fold_step": first_fold_step(sequence, trigger),
            }
        )
    return annotated


def family_rows(
    design: dict[str, object], cells: list[dict[str, object]]
) -> list[dict[str, object]]:
    """One row per declared family: its geometry, its spread, and its equivalence band."""
    fraction = float(
        cast(float, cast(dict[str, object], design["equivalence"])["cap_cost_fraction"])
    )
    return [
        _family_row(
            family,
            [cell for cell in cells if cell["family_id"] == family["family_id"]],
            fraction,
        )
        for family in cast(list[dict[str, object]], design["families"])
    ]


def _family_row(
    family: dict[str, object], members: list[dict[str, object]], fraction: float
) -> dict[str, object]:
    """One family's geometry, its measured spread, and the band that spread is read against."""
    deltas = [_delta(cell) for cell in members]
    cap_costs = [float(cast(float, cell["cap_mean_total_model_input_tokens"])) for cell in members]
    sides = sorted({str(cell["measured_side"]) for cell in members})
    baseline = sum(cap_costs) / len(cap_costs) if cap_costs else 0.0
    spread = max(deltas) - min(deltas) if deltas else 0.0
    return {
        "family_id": family["family_id"],
        "kind": family["kind"],
        "depth": family["depth"],
        "triggers": sorted({_trigger(cell) for cell in members}),
        "fold_steps": sorted(
            {cast(int, cell["predicted_fold_step"]) for cell in members},
            key=lambda step: (step is None, step),
        ),
        "compact_shares": [cast(float, cell["compact_share"]) for cell in members],
        "guards": [cast(int, cell["max_prompt_chars"]) for cell in members],
        "cost_deltas": deltas,
        "spread": spread,
        "cap_baseline_total_model_input_tokens": baseline,
        "cap_baseline_spread": max(cap_costs) - min(cap_costs) if cap_costs else 0.0,
        "equivalence_band": fraction * baseline,
        "measured_sides": sides,
        "within_band": bool(spread <= fraction * baseline),
        "same_side": len(sides) == 1,
    }


def collapse_reading(
    design: dict[str, object],
    eligible: bool,
    cells: list[dict[str, object]],
    families: list[dict[str, object]],
) -> tuple[str, str]:
    """Eligibility, cell validity, and the positive control all gate the collapse claim."""
    invalid = [cell for cell in cells if not cell["valid"]]
    if not eligible:
        return (
            READING_INELIGIBLE,
            "the pinned family no longer passes the unchanged token-chain control",
        )
    if invalid:
        named = "; ".join(f"{cell['cell_id']}: {cell['invalid_reason']}" for cell in invalid)
        return READING_INVALID, f"a declared cell did not hold its preconditions: {named}"
    contrast_id = cast(dict[str, object], design["equivalence"])["contrast_family"]
    contrast = next(row for row in families if row["family_id"] == contrast_id)
    if bool(contrast["within_band"]):
        return (
            READING_NO_POWER,
            f"the contrast family {contrast_id} moved the trigger over {contrast['triggers']} "
            f"chars and still stayed inside its own "
            f"{cast(float, contrast['equivalence_band']):.1f}-token band, so an equal-trigger "
            "agreement would say nothing",
        )
    moved = [
        row
        for row in families
        if row["kind"] == KIND_EQUAL_TRIGGER and not (row["within_band"] and row["same_side"])
    ]
    if moved:
        named = ", ".join(
            f"{row['family_id']} spread {cast(float, row['spread']):.1f} tok vs band "
            f"{cast(float, row['equivalence_band']):.1f} sides={row['measured_sides']}"
            for row in moved
        )
        return READING_INTERACTS, f"an equal-trigger family moved: {named}"
    widest = max(
        cast(float, row["spread"]) for row in families if row["kind"] == KIND_EQUAL_TRIGGER
    )
    return (
        READING_COLLAPSES,
        f"every equal-trigger family held within its band (widest spread {widest:.1f} tokens) "
        f"while the contrast family moved {cast(float, contrast['spread']):.1f} tokens",
    )


def portable_trigger_rule(design: dict[str, object], cap_peaks: dict[int, int]) -> list[str]:
    """Re-express the prior guard crossovers on the trigger axis, which now carries the rule."""
    lines: list[str] = []
    ratios: list[float] = []
    reference = cast(dict[str, object], design["reference"])
    for row in cast(list[dict[str, object]], reference["measured_crossovers"]):
        depth = int(cast(int, row["depth"]))
        peak = cap_peaks.get(depth)
        if peak is None:
            continue
        share = float(cast(float, row["compact_share"]))
        guard = int(cast(int, row["crossover_max_prompt_chars"]))
        trigger = compaction_trigger_chars(guard, share)
        ratios.append(trigger / peak)
        lines.append(
            f"depth {depth}: the {guard}-char crossover guard measured at compact_share={share:g} "
            f"is a {trigger}-char trigger ({trigger / peak:.2f}x the {peak}-char cap peak prompt); "
            f"at any other share s the crossover guard is {trigger}/s"
        )
    if ratios:
        lines.append(
            "portable form: use compact while compact_share * guard stays below "
            f"{min(ratios):.2f}-{max(ratios):.2f}x the transcript's cap peak prompt"
        )
    return lines


def _delta(cell: dict[str, object]) -> float:
    evidence = cast(dict[str, object], cell["cost_evidence"])
    return cast(dict[str, float], evidence[EQUIVALENCE_METRIC])["mean"]


def _trigger(cell: dict[str, object]) -> int:
    return compaction_trigger_chars(
        int(cast(int, cell["max_prompt_chars"])), float(cast(float, cell["compact_share"]))
    )
