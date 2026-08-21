"""Row builders for the fold-step crossover: one row per tested step, one per depth.

The trigger reaches the transcript only by selecting which step folds, so a step is the unit the
cost is grouped on: every guard inside one step's interval sends the identical controller prompts.
These builders attach that geometry to the measured cells and read each depth's ladder into the
LAST fold step at which compact is still cheaper. Both halves of that pairing are read off a probe
result, so both are stated in cells and steps rather than left to the interval arithmetic: a group
whose step the measured sequence cannot answer for is refused here, naming the cells.
"""

from typing import cast

from llb.bench.memory.boundary.gate import SIDE_CAP_CHEAPER, SIDE_COMPACT_CHEAPER
from llb.bench.memory.fold_step.ladder import (
    compaction_trigger_chars,
    first_fold_step,
    fold_step_guard_interval,
    fold_step_trigger_interval,
    foldable_fold_steps,
    smallest_guard_reaching,
)
from llb.bench.memory.fold_step.reading import (
    CONTROLLER_TOKEN_QUANTIZATION,
    READING_CONFIRMED,
    READING_NO_FLIP,
    READING_NON_MONOTONE,
    READING_NO_POWER,
    READING_WITHIN_STEP,
    STEP_METRIC,
)


def step_rows(
    cells: list[dict[str, object]],
    *,
    prompt_sequence: list[int],
    compact_share: float,
    cap_cost_fraction: float,
) -> list[dict[str, object]]:
    """One row per tested fold step: what selects it, what it cost, and whether it held together."""
    steps = sorted(
        {cast(int | None, cell["predicted_fold_step"]) for cell in cells},
        key=lambda step: (step is None, step),
    )
    return [
        _step_row(
            step,
            [cell for cell in cells if cell["predicted_fold_step"] == step],
            prompt_sequence=prompt_sequence,
            compact_share=compact_share,
            cap_cost_fraction=cap_cost_fraction,
        )
        for step in steps
    ]


def depth_fold_row(
    depth: int,
    step_row_list: list[dict[str, object]],
    *,
    prompt_sequence: list[int],
    compact_share: float,
    cap_peak_prompt_chars: int,
    reference_guard: int | None,
) -> dict[str, object]:
    """Read one depth's step ladder into a fold-step boundary, or into a named failure."""
    row: dict[str, object] = {
        "depth": depth,
        "compact_share": compact_share,
        "cap_peak_prompt_chars": cap_peak_prompt_chars,
        "cap_prompt_sequence": prompt_sequence,
        "fold_steps": [cast(int, step["fold_step"]) for step in step_row_list],
        "steps": step_row_list,
        "last_compact_cheaper_fold_step": None,
        "boundary": None,
        "interpolated_guard_artifact": _interpolated_artifact(
            reference_guard, prompt_sequence, compact_share
        ),
        "within_step_residual_tokens": _widest(step_row_list, "spread"),
        "within_step_controller_residual_tokens": _widest(
            step_row_list, "controller_prompt_spread"
        ),
        "within_step_summarizer_residual_tokens": _widest(
            step_row_list, "summarizer_prompt_spread"
        ),
        "controller_cost_is_exact_step": _controller_is_exact(step_row_list),
        "reading": READING_NO_FLIP,
    }
    broken = [step for step in step_row_list if not (step["within_band"] and step["same_side"])]
    if broken:
        row["reading"] = READING_WITHIN_STEP
        return row
    sides = [cast(str, step["side"]) for step in step_row_list]
    flip = _first_flip(sides)
    if flip is None:
        row["reading"] = READING_NO_FLIP if len(set(sides)) == 1 else READING_NON_MONOTONE
        return row
    low, high = step_row_list[flip], step_row_list[flip + 1]
    boundary = _boundary(low, high, prompt_sequence, compact_share)
    row["last_compact_cheaper_fold_step"] = low["fold_step"]
    row["boundary"] = boundary
    row["reading"] = READING_CONFIRMED if boundary["step_change_separates"] else READING_NO_POWER
    return row


def _step_row(
    step: int | None,
    members: list[dict[str, object]],
    *,
    prompt_sequence: list[int],
    compact_share: float,
    cap_cost_fraction: float,
) -> dict[str, object]:
    step = _fold_step_on_the_sequence(step, members, prompt_sequence)
    deltas = [_delta(cell) for cell in members]
    cap_costs = [float(cast(float, cell["cap_mean_total_model_input_tokens"])) for cell in members]
    baseline = sum(cap_costs) / len(cap_costs)
    spread = max(deltas) - min(deltas)
    sides = sorted({cast(str, cell["measured_side"]) for cell in members})
    trigger_interval = fold_step_trigger_interval(prompt_sequence, step)
    guard_interval = fold_step_guard_interval(prompt_sequence, step, compact_share)
    guards = sorted(cast(int, cell["max_prompt_chars"]) for cell in members)
    return {
        "fold_step": step,
        "controller_prompt_spread": _spread(members, "compact_mean_controller_prompt_tokens"),
        "summarizer_prompt_spread": _spread(members, "compact_mean_compaction_prompt_tokens"),
        "bit_identical": spread == 0.0,
        "cell_ids": [cast(str, cell["cell_id"]) for cell in members],
        "trigger_interval": list(trigger_interval),
        "guard_interval": list(guard_interval),
        "guards": guards,
        "triggers": sorted(cast(int, cell["compaction_trigger_chars"]) for cell in members),
        "guard_span_chars": max(guards) - min(guards),
        "guard_interval_width_chars": guard_interval[1] - guard_interval[0],
        "cost_deltas": deltas,
        "mean_cost_delta": sum(deltas) / len(deltas),
        "spread": spread,
        "cap_baseline_total_model_input_tokens": baseline,
        "equivalence_band": cap_cost_fraction * baseline,
        "within_band": bool(spread <= cap_cost_fraction * baseline),
        "measured_sides": sides,
        "same_side": len(sides) == 1,
        "side": sides[0] if len(sides) == 1 else None,
    }


def _fold_step_on_the_sequence(
    step: int | None, members: list[dict[str, object]], prompt_sequence: list[int]
) -> int:
    """The grouped step, or a refusal naming the CELLS the ladder was asked to read.

    The step is a measured cell property (`predicted_fold_step`) and the sequence is the depth's own
    oracle walk, so a step the sequence cannot answer for means the two describe different geometries
    -- a probe that measured nothing, or rows grouped against another depth's ladder. The interval
    arithmetic says that as a bare "outside an N-step sequence", two layers under the declared grid.
    """
    if step is not None and 1 <= step <= len(prompt_sequence):
        return step
    claim = (
        "fold at no step of it -- no prompt exceeds their compact trigger"
        if step is None
        else f"sit at fold step {step}, off its foldable ladder "
        f"{foldable_fold_steps(prompt_sequence)}"
    )
    raise ValueError(
        f"cells {[cast(str, cell['cell_id']) for cell in members]} cannot be read against the "
        f"measured {len(prompt_sequence)}-step prompt sequence: they {claim}"
    )


def _boundary(
    low: dict[str, object],
    high: dict[str, object],
    prompt_sequence: list[int],
    compact_share: float,
) -> dict[str, object]:
    """The step change itself: the trigger that ends the cheap step and the guards that straddle it."""
    trigger_boundary = cast(list[int], low["trigger_interval"])[1]
    guard_boundary = smallest_guard_reaching(trigger_boundary, compact_share)
    gap = abs(min(cast(list[int], high["guards"])) - max(cast(list[int], low["guards"])))
    band = max(cast(float, low["equivalence_band"]), cast(float, high["equivalence_band"]))
    separation = abs(cast(float, high["mean_cost_delta"]) - cast(float, low["mean_cost_delta"]))
    return {
        "from_fold_step": low["fold_step"],
        "to_fold_step": high["fold_step"],
        "trigger_boundary_chars": trigger_boundary,
        "guard_boundary_chars": guard_boundary,
        "compact_cheaper_trigger_interval": low["trigger_interval"],
        "compact_cheaper_guard_interval": low["guard_interval"],
        "straddling_guard_gap_chars": gap,
        "step_change_separation": separation,
        "step_change_band": band,
        "step_change_separates": bool(separation > band),
    }


def _interpolated_artifact(
    reference_guard: int | None, prompt_sequence: list[int], compact_share: float
) -> dict[str, object] | None:
    """Where a previously published interpolated guard falls on the discrete step ladder."""
    if reference_guard is None:
        return None
    step = first_fold_step(
        prompt_sequence, compaction_trigger_chars(reference_guard, compact_share)
    )
    if step is None:
        return None
    interval = fold_step_guard_interval(prompt_sequence, step, compact_share)
    return {
        "guard_chars": reference_guard,
        "fold_step": step,
        "guard_interval": list(interval),
        "gap_to_boundary_chars": interval[1] - reference_guard,
    }


def _widest(step_row_list: list[dict[str, object]], metric: str) -> float | None:
    """The largest per-step value of one spread, or None when any step did not record it."""
    values = [step[metric] for step in step_row_list]
    if any(value is None for value in values):
        return None
    return max((cast(float, value) for value in values), default=0.0)


def _controller_is_exact(step_row_list: list[dict[str, object]]) -> bool | None:
    """Whether the CONTROLLER prompts were bit-identical inside every tested step.

    The sharpest form of the step claim: the fold step fixes the transcript the controller sees,
    so any within-step movement that survives this check comes from the summarizer call instead.
    """
    spreads = [step["controller_prompt_spread"] for step in step_row_list]
    if any(spread is None for spread in spreads):
        return None
    return all(cast(float, spread) <= CONTROLLER_TOKEN_QUANTIZATION for spread in spreads)


def _spread(members: list[dict[str, object]], metric: str) -> float | None:
    """One metric's spread across the guards inside a step, or None when it was not recorded."""
    values = [cell.get(metric) for cell in members]
    if any(value is None for value in values):
        return None
    numbers = [float(cast(float, value)) for value in values]
    return max(numbers) - min(numbers)


def _first_flip(sides: list[str]) -> int | None:
    for index, (low, high) in enumerate(zip(sides, sides[1:])):
        if low == SIDE_COMPACT_CHEAPER and high == SIDE_CAP_CHEAPER:
            return index
    return None


def _delta(cell: dict[str, object]) -> float:
    evidence = cast(dict[str, object], cell["cost_evidence"])
    return cast(dict[str, float], evidence[STEP_METRIC])["mean"]
