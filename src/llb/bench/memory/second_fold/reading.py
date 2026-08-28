"""Study identity, vocabulary, and family reading for the second-fold trigger restatement.

The trigger collapse was measured where a cap arm exists, and every cell that has one folds exactly
once. This study restates the same claim one regime over, so the comparison has to survive without
that arm: each family names an ANCHOR cell, every other member is paired against it on total
model-input tokens, and the family's spread is read against a predeclared fraction of what the
anchor itself costs. Nothing here differences two policies -- the two arms are two compact cells.

The band is stated on the anchor's cost for the same reason the collapse stated it on the cap
baseline: the paired intervals are far tighter than a difference an operator would act on, so an
overlap test would reject a practical equivalence over a few tokens. What replaces the cap arm as
the sanity check is the REPEAT GEOMETRY -- one cell that re-runs the anchor's exact geometry under
another id -- because a spread is only readable as a geometry effect above the noise floor of
running the identical geometry twice.
"""

from typing import cast

from llb.rag.fusion_evidence.evidence_gate import minimum_discordant_pairs
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison
from llb.rag.fusion_evidence.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_index_sets

STUDY_KIND = "compact_second_fold_trigger_collapse"
REPORTING_CONFIDENCE = 0.975

KIND_EQUAL_TRIGGER = "equal_trigger"
KIND_EQUAL_GUARD = "equal_guard"
EQUIVALENCE_RULE = "spread_within_anchor_cost_fraction"
EQUIVALENCE_METRIC = "compact_total_model_input_tokens"
# What makes a cell part of this regime at all: a cell that folds once is back inside the claim the
# collapse already established, and measures nothing this study exists to measure.
MIN_REPEATED_FOLDS = 2

READING_INELIGIBLE = "pinned_family_control_ineligible"
READING_INVALID = "second_fold_cells_invalid"
READING_UNSTABLE = "repeat_geometry_did_not_reproduce"
READING_NO_POWER = "no_resolving_power"
READING_COLLAPSES = "trigger_ratio_collapses_through_the_second_fold"
READING_GUARD_REENTERS = "guard_re_enters_after_the_first_fold"


def member_comparison(
    member: dict[str, object], anchor: dict[str, object], confidence: float
) -> PairedComparison:
    """One member's paired cost delta against its family anchor, item by item.

    The two cells ran the identical task set at the identical seed, so the pairing is by task
    position and needs no baseline policy: what varies between them is the geometry alone.
    """
    candidate = _costs(member)
    baseline = _costs(anchor)
    return paired_comparison(
        candidate,
        baseline,
        bootstrap_index_sets(len(baseline), DEFAULT_RESAMPLES, DEFAULT_SEED),
        confidence,
    )


def separates(comparison: PairedComparison, band: float, confidence: float) -> bool:
    """A delta both larger than the equivalence band and readable at the reporting level."""
    discordant = comparison["wins"] + comparison["losses"]
    return bool(
        abs(cast(dict[str, float], comparison["delta"])["mean"]) > band
        and float(comparison["sign_test_p"]) <= 1.0 - confidence
        and discordant >= minimum_discordant_pairs(confidence)
    )


def family_rows(
    design: dict[str, object], cells: list[dict[str, object]], confidence: float
) -> list[dict[str, object]]:
    """One row per declared family: its anchor, every member's paired delta, and its band."""
    fraction = float(
        cast(float, cast(dict[str, object], design["equivalence"])["anchor_cost_fraction"])
    )
    return [
        _family_row(
            family,
            [cell for cell in cells if cell["family_id"] == family["family_id"]],
            fraction,
            confidence,
        )
        for family in cast(list[dict[str, object]], design["families"])
    ]


def _family_row(
    family: dict[str, object],
    members: list[dict[str, object]],
    fraction: float,
    confidence: float,
) -> dict[str, object]:
    anchor = members[0]
    anchor_cost = float(cast(float, anchor["mean_total_model_input_tokens"]))
    band = fraction * anchor_cost
    comparisons = [member_comparison(member, anchor, confidence) for member in members[1:]]
    moved = [
        {
            "cell_id": member["cell_id"],
            "compact_share": member["compact_share"],
            "max_prompt_chars": member["max_prompt_chars"],
            "compaction_trigger_chars": member["compaction_trigger_chars"],
            "expected_separation": bool(member["expected_separation"]),
            "paired_delta": comparison["delta"],
            "discordant_pairs": comparison["wins"] + comparison["losses"],
            "two_sided_sign_test_p": comparison["sign_test_p"],
            "separates": separates(comparison, band, confidence),
        }
        for member, comparison in zip(members[1:], comparisons, strict=True)
    ]
    costs = [float(cast(float, member["mean_total_model_input_tokens"])) for member in members]
    spread = max(costs) - min(costs)
    return {
        "family_id": family["family_id"],
        "kind": family["kind"],
        "depth": family["depth"],
        "anchor_cell_id": anchor["cell_id"],
        "anchor_total_model_input_tokens": anchor_cost,
        "triggers": sorted({cast(int, member["compaction_trigger_chars"]) for member in members}),
        "guards": [cast(int, member["max_prompt_chars"]) for member in members],
        "compact_shares": [cast(float, member["compact_share"]) for member in members],
        "first_fold_steps": sorted({cast(int, member["first_fold_step"]) for member in members}),
        "measured_fold_counts": [
            cast(list[int], member["measured_fold_counts"]) for member in members
        ],
        "member_deltas": moved,
        "spread": spread,
        "equivalence_band": band,
        "within_band": bool(spread <= band),
        "separated_members": [row["cell_id"] for row in moved if row["separates"]],
    }


def repeat_geometry_row(
    cells: list[dict[str, object]], families: list[dict[str, object]], confidence: float
) -> dict[str, object] | None:
    """The noise floor: one cell re-running another's exact geometry under a second id."""
    repeat = next((cell for cell in cells if cell.get("repeats_anchor")), None)
    if repeat is None:
        return None
    anchor = next(cell for cell in cells if cell["cell_id"] == repeat["repeats_anchor"])
    band = next(
        float(cast(float, row["equivalence_band"]))
        for row in families
        if row["anchor_cell_id"] == anchor["cell_id"]
    )
    comparison = member_comparison(repeat, anchor, confidence)
    return {
        "cell_id": repeat["cell_id"],
        "anchor_cell_id": anchor["cell_id"],
        "paired_delta": comparison["delta"],
        "two_sided_sign_test_p": comparison["sign_test_p"],
        "equivalence_band": band,
        "reproduces": not separates(comparison, band, confidence),
    }


def second_fold_reading(
    design: dict[str, object],
    eligible: bool,
    cells: list[dict[str, object]],
    families: list[dict[str, object]],
    repeat: dict[str, object] | None,
) -> tuple[str, str]:
    """Eligibility, cell validity, the repeat pair, and the contrast all gate the claim."""
    if not eligible:
        return (
            READING_INELIGIBLE,
            "the pinned family no longer passes the unchanged token-chain control",
        )
    contrast_id = cast(dict[str, object], design["equivalence"])["contrast_family"]
    contrast = next((row for row in families if row["family_id"] == contrast_id), None)
    for refusal in (
        _invalid_reading(cells),
        _repeat_reading(repeat),
        _power_reading(contrast, contrast_id),
    ):
        if refusal is not None:
            return refusal
    return _trigger_reading(families, cast(dict[str, object], contrast))


def _invalid_reading(cells: list[dict[str, object]]) -> tuple[str, str] | None:
    """A declared cell that lost its preconditions cannot contribute a comparable cost."""
    invalid = [cell for cell in cells if not cell["valid"]]
    if not invalid:
        return None
    named = "; ".join(f"{cell['cell_id']}: {cell['invalid_reason']}" for cell in invalid)
    return READING_INVALID, f"a declared cell did not hold its preconditions: {named}"


def _repeat_reading(repeat: dict[str, object] | None) -> tuple[str, str] | None:
    """The noise floor: one geometry run twice must land inside the band it is read against."""
    if repeat is None or repeat["reproduces"]:
        return None
    delta = cast(dict[str, float], repeat["paired_delta"])["mean"]
    return (
        READING_UNSTABLE,
        f"{repeat['cell_id']} re-ran the geometry of {repeat['anchor_cell_id']} and moved "
        f"{delta:+.1f} tokens, past its own "
        f"{cast(float, repeat['equivalence_band']):.1f}-token band, so no spread here is "
        "readable as a geometry effect",
    )


def _power_reading(
    contrast: dict[str, object] | None, contrast_id: object
) -> tuple[str, str] | None:
    """The positive control: moving the trigger onto another fold step must move the cost."""
    if contrast is None:
        return READING_NO_POWER, f"the design names no contrast family {contrast_id!r} to read"
    if contrast["separated_members"]:
        return None
    return (
        READING_NO_POWER,
        f"the contrast family {contrast_id} moved the trigger across fold steps "
        f"{contrast['first_fold_steps']} and no member cleared its "
        f"{cast(float, contrast['equivalence_band']):.1f}-token band",
    )


def _trigger_reading(
    families: list[dict[str, object]], contrast: dict[str, object]
) -> tuple[str, str]:
    """Whether every equal-trigger family held, now that the run is readable at all."""
    moved = [
        row
        for row in families
        if row["kind"] == KIND_EQUAL_TRIGGER
        and (row["separated_members"] or not row["within_band"])
    ]
    if moved:
        named = ", ".join(
            f"{row['family_id']} spread {cast(float, row['spread']):.1f} tok vs band "
            f"{cast(float, row['equivalence_band']):.1f}, separated={row['separated_members']}"
            for row in moved
        )
        return READING_GUARD_REENTERS, f"an equal-trigger family moved: {named}"
    widest = max(
        cast(float, row["spread"]) for row in families if row["kind"] == KIND_EQUAL_TRIGGER
    )
    return (
        READING_COLLAPSES,
        f"every equal-trigger family held within its band (widest spread {widest:.1f} tokens) "
        f"while the contrast family separated {contrast['separated_members']}",
    )


def _costs(cell: dict[str, object]) -> list[float]:
    return [float(value) for value in cast(list[float], cell["case_total_model_input_tokens"])]
