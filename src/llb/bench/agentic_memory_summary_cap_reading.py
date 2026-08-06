"""Study identity, vocabulary, and the two readings the summarize-input cap study cuts.

The fold step fixes the transcript the controller sees, so the compact cost is a step function of
the trigger -- except for the summarize call, whose input cap was the trigger itself. Two guards
that fold the SAME step therefore fed the summarizer different amounts of the same transcript, which
is the whole within-step residual the fold-step study isolated, and it also means the shipped cap
elided a folded transcript that would have fit the window.

Two independent questions follow, and this study reads both:

  - the RESIDUAL reading -- does pinning the cap to a step-aligned quantity drive the within-step
    movement to zero without moving the fold-step boundary the routing rule is stated on;
  - the ELISION reading -- does the span the shipped cap cut out of the summarizer's input cost
    completion, i.e. was it trimming evidence the summary needed.

The per-arm step and ladder rows are the fold-step study's own (`agentic_memory_fold_step_rows`),
so a residual measured here is on exactly the scale that study published.
"""

from typing import cast

from llb.rag.fusion_evidence.evidence_gate import (
    READING_INSUFFICIENT_EVIDENCE,
    READING_SEPARATED,
)

STUDY_KIND = "compact_summary_input_cap"
METHOD = "agentic-compact-summary-input-cap"
REPORTING_CONFIDENCE = 0.975

CAP_RULE = "step_aligned_cap_zeroes_the_within_step_residual"
CAP_METRIC = "compact_minus_cap_total_model_input_tokens"
COMPLETION_METRIC = "compact_completion"

ROLE_STEP_ALIGNED = "step_aligned"
ROLE_REFERENCE = "shipped_reference"
ARM_ROLES = (ROLE_REFERENCE, ROLE_STEP_ALIGNED)

READING_INELIGIBLE = "pinned_family_control_ineligible"
READING_INVALID = "summary_cap_cells_invalid"
READING_EXACT = "step_aligned_cap_is_an_exact_step_function"
READING_RESIDUAL_SURVIVES = "within_step_residual_survives_step_alignment"
READING_BOUNDARY_MOVED = "step_aligned_cap_moves_the_fold_step_boundary"
READING_LADDER_UNREADABLE = "an_arm_ladder_is_unreadable"

ELISION_FREE = "elided_span_costs_no_completion"
ELISION_COSTS = "elided_span_costs_completion"
ELISION_NONE_TO_PRICE = "reference_arm_elided_nothing"
ELISION_UNREADABLE = "elision_completion_delta_is_unreadable"


def residual_reading(
    eligible: bool,
    cells: list[dict[str, object]],
    arms: list[dict[str, object]],
    *,
    tolerance_tokens: float,
) -> tuple[str, str]:
    """Whether the step-aligned arm turned the compact cost into a pure step function."""
    if not eligible:
        return (
            READING_INELIGIBLE,
            "the pinned family no longer passes the unchanged token-chain control",
        )
    invalid = [cell for cell in cells if not cell["valid"]]
    if invalid:
        named = "; ".join(f"{cell['cell_id']}: {cell['invalid_reason']}" for cell in invalid)
        return READING_INVALID, f"a declared cell did not hold its preconditions: {named}"
    unreadable = [arm for arm in arms if not arm["ladder_confirms_boundary"]]
    if unreadable:
        named = ", ".join(f"{arm['arm_id']}: {arm['ladder_reading']}" for arm in unreadable)
        return READING_LADDER_UNREADABLE, f"an arm did not resolve its own ladder: {named}"
    aligned = _arm(arms, ROLE_STEP_ALIGNED)
    reference = _arm(arms, ROLE_REFERENCE)
    if aligned["last_compact_cheaper_fold_step"] != reference["last_compact_cheaper_fold_step"]:
        return (
            READING_BOUNDARY_MOVED,
            f"the step-aligned cap moves the last compact-cheaper fold step from "
            f"{reference['last_compact_cheaper_fold_step']} to "
            f"{aligned['last_compact_cheaper_fold_step']}, so the published routing rule does not "
            "carry over unchanged",
        )
    residual = cast(float, aligned["within_step_residual_tokens"])
    if residual > tolerance_tokens:
        return (
            READING_RESIDUAL_SURVIVES,
            f"{residual:.1f} tokens still move inside a fold step under the step-aligned cap, over "
            f"the predeclared {tolerance_tokens:.1f}-token tolerance",
        )
    return (
        READING_EXACT,
        f"the step-aligned cap leaves {residual:.1f} tokens moving inside a fold step (against "
        f"{cast(float, reference['within_step_residual_tokens']):.1f} under the trigger cap) while "
        f"the boundary stays at fold step {aligned['last_compact_cheaper_fold_step']}",
    )


def elision_reading(
    elided_chars: float, completion_delta_reading: str, completion_delta_mean: float
) -> tuple[str, str]:
    """What the span the shipped cap cut out of the summarizer input cost in completion.

    Direction matters: the delta is step-aligned MINUS reference, so a separated POSITIVE delta is
    completion the elision was destroying and a separated negative one would mean the trimmed input
    summarized better -- both are findings, and only a flat reading says the elision was free.
    """
    if elided_chars <= 0.0:
        return (
            ELISION_NONE_TO_PRICE,
            "the reference arm elided nothing on this ladder, so there is no trimmed span to price",
        )
    if completion_delta_reading == READING_SEPARATED:
        return (
            ELISION_COSTS,
            f"removing the {elided_chars:.0f}-char elision moves compact completion by "
            f"{completion_delta_mean:+.3f}, so the shipped cap was trimming evidence the summary "
            "needed",
        )
    if completion_delta_reading == READING_INSUFFICIENT_EVIDENCE:
        return (
            ELISION_UNREADABLE,
            "too few tasks differ for the completion delta to be readable at this reporting level",
        )
    return (
        ELISION_FREE,
        f"the reference arm elided {elided_chars:.0f} chars of summarizer input and completion "
        f"stayed flat ({completion_delta_mean:+.3f}), so the trimmed span carried no evidence the "
        "summary needed",
    )


def operator_lines(
    arms: list[dict[str, object]], residual: str, elision: str, shipped_cap: str
) -> list[str]:
    """What an operator takes away: what the cap is now, and what changed by pinning it.

    Only a reading that survived every gate earns the "pure step function" line. A moved boundary or
    an unreadable ladder gets no line at all, because the arms did not establish one.
    """
    if residual in (READING_INELIGIBLE, READING_INVALID, READING_LADDER_UNREADABLE):
        return [f"[{residual}] no summarize-input-cap line is supported"]
    aligned = _arm(arms, ROLE_STEP_ALIGNED)
    reference = _arm(arms, ROLE_REFERENCE)
    if residual == READING_BOUNDARY_MOVED:
        return [
            f"[{residual}] the step-aligned cap moves the last compact-cheaper fold step from "
            f"{reference['last_compact_cheaper_fold_step']} to "
            f"{aligned['last_compact_cheaper_fold_step']}; re-derive the routing rule before "
            "applying it under the shipped bound"
        ]
    lines = [
        f"the summarize call's input is bounded by `{shipped_cap}` -- the resolved prompt budget "
        "minus the summary template -- so the folded transcript is summarized at its own size"
        + (
            " and the compact cost is a step function of the fold step alone"
            if residual == READING_EXACT
            else f", but {cast(float, aligned['within_step_residual_tokens']):.1f} tokens still "
            "move inside a fold step, so the cost is not yet a pure step function"
        ),
        f"depth {aligned['depth']}: within-step residual "
        f"{cast(float, reference['within_step_residual_tokens']):.1f} tokens under the trigger cap "
        f"-> {cast(float, aligned['within_step_residual_tokens']):.1f} under the step-aligned cap; "
        f"the fold-step boundary is unchanged at step {aligned['last_compact_cheaper_fold_step']}",
    ]
    if elision == ELISION_FREE:
        lines.append(
            "the trigger cap's elision was free on this shape: pin the cap for predictability, not "
            "for completion"
        )
    elif elision == ELISION_COSTS:
        lines.append(
            "the trigger cap was cutting evidence the summary needed -- the pin buys completion, "
            "not only predictability"
        )
    elif elision == ELISION_NONE_TO_PRICE:
        lines.append(
            "this ladder folds a transcript the trigger cap never trimmed, so it says nothing "
            "about what an elision costs; a deeper ladder would"
        )
    return lines


def _arm(arms: list[dict[str, object]], role: str) -> dict[str, object]:
    return next(arm for arm in arms if arm["role"] == role)
