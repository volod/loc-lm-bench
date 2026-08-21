"""Study identity, vocabulary, and the step-level reading for the fold-step crossover.

The compact cost delta is a STEP function of the compaction trigger: the trigger reaches the
transcript ONLY by selecting which step folds, so every trigger inside one step's interval produces
the identical transcript and costs the same to the token. The crossover is therefore not a point on
the guard axis but the boundary between two fold steps, and a guard value interpolated between two
cells is an artifact of fitting a continuous rule to a discrete mechanism.

This module owns the study identity, the reading vocabulary, and the operator-facing routing lines;
the per-step and per-depth rows they are read from are built in
`agentic_memory_fold_step_rows.py`.
"""

from typing import cast

STUDY_KIND = "compact_fold_step_crossover"
METHOD = "agentic-compact-fold-step-crossover"
REPORTING_CONFIDENCE = 0.975

STEP_RULE = "within_step_spread_within_cap_cost_fraction"
STEP_METRIC = "compact_minus_cap_total_model_input_tokens"

READING_INELIGIBLE = "pinned_family_control_ineligible"
READING_INVALID = "fold_step_cells_invalid"
READING_CONFIRMED = "fold_step_boundary_confirmed"
READING_PARTIAL = "fold_step_boundary_partially_resolved"
READING_WITHIN_STEP = "cost_moves_within_a_fold_step"
READING_NON_MONOTONE = "cost_side_not_monotone_in_fold_step"
READING_NO_POWER = "no_resolving_power"
READING_NO_FLIP = "no_side_change_across_tested_steps"

# The controller column is DERIVED (total minus summarizer) from two separately truncated
# char->token projections, so identical controller chars can still land one token apart. Exactness
# is read at that quantization, never below it.
CONTROLLER_TOKEN_QUANTIZATION = 1.0

# Worst-first, so one unresolved depth is named by its own failure rather than by a generic miss.
_DEPTH_PRIORITY = (READING_WITHIN_STEP, READING_NON_MONOTONE, READING_NO_POWER, READING_NO_FLIP)


def fold_step_reading(
    eligible: bool, cells: list[dict[str, object]], depth_rows: list[dict[str, object]]
) -> tuple[str, str]:
    """Eligibility, cell validity, and every depth's own ladder gate the boundary claim."""
    invalid = [cell for cell in cells if not cell["valid"]]
    if not eligible:
        return (
            READING_INELIGIBLE,
            "the pinned family no longer passes the unchanged token-chain control",
        )
    if invalid:
        named = "; ".join(f"{cell['cell_id']}: {cell['invalid_reason']}" for cell in invalid)
        return READING_INVALID, f"a declared cell did not hold its preconditions: {named}"
    readings = [cast(str, row["reading"]) for row in depth_rows]
    confirmed = readings.count(READING_CONFIRMED)
    if not depth_rows:
        return READING_INVALID, "no depth produced a readable fold-step ladder"
    if confirmed == len(readings):
        return (
            READING_CONFIRMED,
            f"every tested depth flips cost side exactly at a fold-step change ({confirmed} of "
            f"{len(readings)}); within a step, guards spanning the whole interval agree",
        )
    worst = next(name for name in _DEPTH_PRIORITY if name in readings)
    named = ", ".join(
        f"depth {row['depth']}: {row['reading']}"
        for row in depth_rows
        if row["reading"] != READING_CONFIRMED
    )
    if confirmed:
        return READING_PARTIAL, f"{confirmed}/{len(readings)} depths confirmed; {named}"
    return worst, named


def routing_rule(depth_rows: list[dict[str, object]]) -> list[str]:
    """Operator-facing lines: a fold step to stay at or below, not a char threshold to guess at."""
    lines: list[str] = []
    for row in depth_rows:
        depth = cast(int, row["depth"])
        share = cast(float, row["compact_share"])
        if row["reading"] != READING_CONFIRMED:
            lines.append(
                f"depth {depth}: [{row['reading']}] no fold-step routing line is supported"
            )
            continue
        boundary = cast(dict[str, object], row["boundary"])
        step = cast(int, row["last_compact_cheaper_fold_step"])
        trigger = cast(int, boundary["trigger_boundary_chars"])
        guard = cast(int, boundary["guard_boundary_chars"])
        lines.append(
            f"depth {depth}: fold no later than step {step} -- use compact while "
            f"compact_share * guard stays below {trigger} chars (step {step}'s own cap prompt), "
            f"which at compact_share={share:g} is a guard below {guard} chars; use "
            "observation_cap at or above it"
        )
        lines.extend(_residual_line(row, boundary))
        artifact = cast(dict[str, object] | None, row["interpolated_guard_artifact"])
        if artifact is not None:
            lines.append(
                f"depth {depth}: the interpolated {artifact['guard_chars']}-char crossover guard "
                f"lands inside step {artifact['fold_step']}'s guard interval "
                f"[{cast(list[int], artifact['guard_interval'])[0]}, "
                f"{cast(list[int], artifact['guard_interval'])[1]}), where every guard costs the "
                "same, so it names a "
                f"point at which nothing changes ({artifact['gap_to_boundary_chars']} chars below "
                "the step change that does)"
            )
    return lines


def _residual_line(row: dict[str, object], boundary: dict[str, object]) -> list[str]:
    """Name whatever cost still moves INSIDE a step, and where it comes from.

    The step claim is about the transcript the fold step fixes. What survives it is the summarize
    call, whose input cap IS the trigger: a different trigger inside one step trims that input
    differently, and the summary it returns is then carried by every later controller prompt.
    """
    residual = cast(float, row["within_step_residual_tokens"])
    if residual <= 0.0:
        return [
            f"depth {row['depth']}: inside a step the whole cost is bit-identical across the "
            "guard interval -- the summarize call never reached its trigger-sized input cap"
        ]
    summarizer = cast(float | None, row["within_step_summarizer_residual_tokens"])
    controller = cast(float | None, row["within_step_controller_residual_tokens"])
    if summarizer is None or controller is None:
        source = "unrecorded on this bundle"
    elif row["controller_cost_is_exact_step"]:
        source = (
            f"all of it the summarize call ({summarizer:.1f} tok), whose input cap IS the trigger"
        )
    else:
        source = (
            f"{summarizer:.1f} tok of it the summarize call, whose input cap IS the trigger, plus "
            f"{controller:.1f} tok of later controller prompts carrying the summary it returned"
        )
    return [
        f"depth {row['depth']}: {residual:.1f} tokens still move INSIDE a step -- {source} -- "
        f"against a {cast(float, boundary['step_change_separation']):.1f}-token step change"
    ]
