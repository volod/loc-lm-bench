"""Cell-level contract for the cap-fitting boundary surface: what a cell must be, and what it says.

A surface cell is only informative when BOTH policies stayed usable: cap never overflowed, compact
actually compacted, completion stayed paired, and neither policy was measured on a geometry it
could not finish. Those preconditions are checked here rather than inside the crossover reading, so
a failed precondition surfaces as an invalid cell with a named reason instead of quietly bending
the interpolated crossover.
"""

from typing import cast

from llb.bench.agentic_design_fields import as_int, as_str
from llb.bench.agentic_memory_boundary_gate import (
    SIDE_CAP_CHEAPER,
    SIDE_COMPACT_CHEAPER,
    boundary_cost_evidence,
)
from llb.bench.agentic_memory_fold_step_ladder import (
    guard_is_cap_fitting_under_imperfect_play,
    imperfect_play_guard_band,
)
from llb.bench.agentic_memory_worst_case_probe import cap_peak_margin, margin_peaks

EXPECTED_SIDES = (SIDE_COMPACT_CHEAPER, SIDE_CAP_CHEAPER)


def _cell_geometries(cells: list[dict[str, object]]) -> list[tuple[int, int]]:
    """Each cell's (depth, guard) pair -- the identity a surface cell is placed by."""
    return [(as_int(cell, "depth"), as_int(cell, "max_prompt_chars")) for cell in cells]


def _check_cell_identity(cells: list[dict[str, object]], geometries: list[tuple[int, int]]) -> None:
    """Uniquely named, uniquely placed, runnable, and predeclaring which side it expects."""
    ids = [as_str(cell, "cell_id") for cell in cells]
    if len(cells) < 3 or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("the boundary surface needs at least three uniquely named cells")
    if len(set(geometries)) != len(geometries):
        raise ValueError("boundary-surface cells must name unique depth/guard geometries")
    if any(depth < 3 or guard <= 0 for depth, guard in geometries):
        raise ValueError("boundary-surface cell geometry is invalid")
    if any(cell.get("expected_side") not in EXPECTED_SIDES for cell in cells):
        raise ValueError(f"every boundary-surface cell must predeclare one of {EXPECTED_SIDES}")


def validate_surface_cells(
    cells: list[dict[str, object]],
    *,
    reference: dict[str, object],
    held_fixed: dict[str, object],
) -> None:
    """Refuse a grid that cannot bracket a crossover or cannot keep both policies usable."""
    geometries = _cell_geometries(cells)
    _check_cell_identity(cells, geometries)
    anchor = (as_int(reference, "depth"), as_int(reference, "max_prompt_chars"))
    if anchor not in geometries:
        raise ValueError("the boundary surface must re-run the reference cap-fitting geometry")
    depths = sorted({depth for depth, _guard in geometries})
    if len(depths) < 2:
        raise ValueError("the boundary surface must vary depth as well as prompt guard")
    for depth in depths:
        _validate_depth(depth, [cell for cell in cells if cell["depth"] == depth], held_fixed)
    _validate_window(held_fixed, max(guard for _depth, guard in geometries))


def _validate_depth(
    depth: int, cells: list[dict[str, object]], held_fixed: dict[str, object]
) -> None:
    """One depth must predeclare both sides in guard order, all inside the usable band."""
    rows = sorted(cells, key=lambda cell: int(cast(int, cell["max_prompt_chars"])))
    sides = [str(cell["expected_side"]) for cell in rows]
    if len(rows) < 2 or set(sides) != set(EXPECTED_SIDES):
        raise ValueError(
            f"depth {depth} needs cells predeclared on both sides of the cost crossover"
        )
    if sides != sorted(sides, key=EXPECTED_SIDES.index):
        raise ValueError(
            f"depth {depth} must predeclare compact-cheaper cells below cap-cheaper ones"
        )
    _validate_band(
        depth,
        [int(cast(int, cell["max_prompt_chars"])) for cell in rows],
        held_fixed,
        float(cast(float, held_fixed["compact_share"])),
    )


def surface_cell_row(
    row: dict[str, object],
    cell: dict[str, object],
    *,
    held_fixed: dict[str, object],
    confidence: float,
) -> dict[str, object]:
    """Attach the predeclared side, the direction-aware cost reading, and the validity gate."""
    evidence = boundary_cost_evidence(row, confidence)
    reason = _invalid_reason(row, evidence, held_fixed)
    measured_side = cast(str, evidence["cost_side"])
    return {
        **row,
        "expected_side": cell["expected_side"],
        "measured_side": measured_side,
        "matches_expectation": bool(reason is None and measured_side == cell["expected_side"]),
        "cost_evidence": evidence,
        "valid": reason is None,
        "invalid_reason": reason,
    }


def _invalid_reason(
    row: dict[str, object], evidence: dict[str, object], held_fixed: dict[str, object]
) -> str | None:
    activation_floor = float(cast(float, held_fixed["minimum_compaction_rate"]))
    completion_floor = float(cast(float, held_fixed["minimum_cell_completion"]))
    completions = (
        float(cast(float, row["cap_completion"])),
        float(cast(float, row["compact_completion"])),
    )
    if int(cast(int, row["cap_context_overflows"])) > 0:
        return "observation_cap overflowed, so the guard is not cap-fitting"
    if int(cast(int, row["compact_context_overflows"])) > 0:
        return "compact overflowed at this guard"
    if float(cast(float, row["compaction_activation_rate"])) < activation_floor:
        return "compaction did not reach the predeclared activation floor"
    if not evidence["completion_tied"]:
        return "completion is not paired, so the cell measures rescue rather than cost"
    if min(completions) < completion_floor:
        return "both policies stayed below the cell completion floor"
    return None


def _validate_band(
    depth: int, guards: list[int], held_fixed: dict[str, object], share: float
) -> None:
    """Certify every guard against the band the SAFETY MARGIN leaves, not the perfect-play one.

    A guard above the perfect-play peak is cap-fitting for a controller that never wastes a step.
    The cell it places is run by a model, so the certification is made against the worst case the
    study's own step budget allows -- the margin is measured from the same held geometry the cells
    declare, so raising `max_steps_margin` tightens this check by exactly what it loosens the run.
    """
    worst_peak, peak = margin_peaks(depth_cap_peak_margin(depth, held_fixed))
    low, high = imperfect_play_guard_band(worst_peak, peak, share)
    outside = [
        guard
        for guard in guards
        if not guard_is_cap_fitting_under_imperfect_play(guard, worst_peak, peak, share)
    ]
    if outside:
        raise ValueError(
            f"depth {depth} guards {outside} fall outside the usable band ({low}, {high}) where "
            f"cap fits imperfect play ({worst_peak - peak} chars above the {peak}-char perfect-play "
            "peak) and compact still activates"
        )


def depth_cap_peak_margin(depth: int, held_fixed: dict[str, object]) -> dict[str, object]:
    """This depth's two cap peaks and the margin between them, over the study's held geometry.

    The surface is where a depth the probe measured nothing over can be NAMED, so the walks and the
    checked read are paired here rather than left to the band arithmetic two layers down. The share
    is NOT translated -- it is a `held_fixed` value the design states verbatim, and the ladder's own
    "compact share must be in (0, 1]" already names it.
    """
    return cap_peak_margin(
        depth=depth,
        n_tasks=int(cast(int, held_fixed["n_tasks"])),
        pad_chars=int(cast(int, held_fixed["pad_chars"])),
        max_steps_margin=int(cast(int, held_fixed["max_steps_margin"])),
        observation_cap_chars=int(cast(int, held_fixed["observation_cap_chars"])),
        observation_head_share=float(cast(float, held_fixed["observation_head_share"])),
    )


def _validate_window(held_fixed: dict[str, object], max_guard: int) -> None:
    from llb.optimize.tuning_space import CHARS_PER_TOKEN, PROMPT_HEADROOM_TOKENS

    declared = int(cast(int, held_fixed["max_model_len"]))
    required = int(max_guard / CHARS_PER_TOKEN) + PROMPT_HEADROOM_TOKENS
    if declared < required:
        raise ValueError(
            f"declared max_model_len {declared} cannot carry the widest guard "
            f"({max_guard} chars needs about {required} tokens)"
        )
