"""The predeclared interpolation that turns cap-fitting cells into a routing surface.

One usable cap-fitting geometry says only that compact repaid its summary THERE. A routing surface
needs the guard at which the paired cost delta changes sign for a given transcript depth, so the
rule is fixed before the run: read the compact-minus-cap total model-input delta on the guard axis,
take the FIRST adjacent pair of cost-separated cells whose means have opposite signs, and
interpolate linearly to the zero crossing. A grid with no sign change reports a BOUND (the crossing
lies outside the tested guards) instead of extrapolating past the evidence.
"""

from typing import cast

from llb.bench.memory.boundary.gate import (
    SIDE_CAP_CHEAPER,
    SIDE_COMPACT_CHEAPER,
    SIDE_TIED,
)

RULE_LINEAR_ZERO_CROSSING = "linear_zero_crossing"

READING_BRACKETED = "crossover_bracketed"
READING_COMPACT_ACROSS_GRID = "compact_cheaper_across_grid"
READING_CAP_ACROSS_GRID = "cap_cheaper_across_grid"
READING_INSUFFICIENT_CELLS = "insufficient_valid_cells"
READING_UNREADABLE = "cost_side_unreadable"


def interpolate_crossover(
    low_guard: int, low_delta: float, high_guard: int, high_delta: float
) -> float:
    """Linear zero crossing of the cost delta between two bracketing prompt guards."""
    if high_guard <= low_guard:
        raise ValueError("crossover interpolation needs strictly increasing prompt guards")
    if low_delta == high_delta:
        raise ValueError("crossover interpolation needs two different cost deltas")
    return low_guard + (high_guard - low_guard) * (-low_delta) / (high_delta - low_delta)


def depth_surface_row(
    depth: int, cells: list[dict[str, object]], *, cap_peak_prompt_chars: int
) -> dict[str, object]:
    """Read one depth's guard sweep into a crossover, or into a bounded direction."""
    valid = sorted(
        (cell for cell in cells if cell["valid"]),
        key=lambda cell: cast(int, cell["max_prompt_chars"]),
    )
    row: dict[str, object] = {
        "depth": depth,
        "cap_peak_prompt_chars": cap_peak_prompt_chars,
        "guards": [cast(int, cell["max_prompt_chars"]) for cell in valid],
        "measured_sides": [cast(str, cell["measured_side"]) for cell in valid],
        "crossover_max_prompt_chars": None,
        "crossover_guard_ratio": None,
        "bracket": None,
        "reading": READING_INSUFFICIENT_CELLS,
    }
    if len(valid) < 2:
        return row
    # The rule requires cost separation at BOTH endpoints, so a cell whose cost side is not
    # readable is never an endpoint -- it is skipped rather than blocking a bracket around it.
    readable = [cell for cell in valid if cell["measured_side"] != SIDE_TIED]
    for low, high in zip(readable, readable[1:]):
        if {low["measured_side"], high["measured_side"]} == {
            SIDE_COMPACT_CHEAPER,
            SIDE_CAP_CHEAPER,
        }:
            crossover = interpolate_crossover(
                cast(int, low["max_prompt_chars"]),
                _delta(low),
                cast(int, high["max_prompt_chars"]),
                _delta(high),
            )
            row["crossover_max_prompt_chars"] = crossover
            row["crossover_guard_ratio"] = crossover / cap_peak_prompt_chars
            row["bracket"] = [low["max_prompt_chars"], high["max_prompt_chars"]]
            row["reading"] = READING_BRACKETED
            return row
    sides = {cast(str, cell["measured_side"]) for cell in readable}
    if sides == {SIDE_COMPACT_CHEAPER}:
        row["reading"] = READING_COMPACT_ACROSS_GRID
    elif sides == {SIDE_CAP_CHEAPER}:
        row["reading"] = READING_CAP_ACROSS_GRID
    else:
        row["reading"] = READING_UNREADABLE
    return row


def routing_rule(depth_rows: list[dict[str, object]]) -> list[str]:
    """Operator-facing routing lines, one per tested depth plus the portable ratio when stable."""
    lines: list[str] = []
    ratios: list[float] = []
    for row in depth_rows:
        depth = cast(int, row["depth"])
        if row["reading"] == READING_BRACKETED:
            crossover = cast(float, row["crossover_max_prompt_chars"])
            ratio = cast(float, row["crossover_guard_ratio"])
            ratios.append(ratio)
            lines.append(
                f"depth {depth}: use compact below a {crossover:.0f}-char prompt guard "
                f"({ratio:.2f}x the {row['cap_peak_prompt_chars']}-char cap peak prompt); "
                "use observation_cap above it"
            )
        elif row["reading"] == READING_COMPACT_ACROSS_GRID:
            lines.append(
                f"depth {depth}: compact is cheaper across every tested guard up to "
                f"{max(cast(list[int], row['guards']))} chars; the crossover is above the grid"
            )
        elif row["reading"] == READING_CAP_ACROSS_GRID:
            lines.append(
                f"depth {depth}: observation_cap is cheaper from the tightest tested guard "
                f"({min(cast(list[int], row['guards']))} chars) upward; the crossover is below "
                "the grid"
            )
        else:
            lines.append(f"depth {depth}: [{row['reading']}] no routing line is supported")
    if len(ratios) >= 2:
        spread = max(ratios) - min(ratios)
        lines.append(
            f"portable form: the crossover sits at {min(ratios):.2f}-{max(ratios):.2f}x the "
            f"cap peak prompt (spread {spread:.2f}) across the tested depths"
        )
    return lines


def _delta(cell: dict[str, object]) -> float:
    evidence = cast(dict[str, object], cell["cost_evidence"])
    delta = cast(dict[str, float], evidence["compact_minus_cap_total_model_input_tokens"])
    return delta["mean"]
