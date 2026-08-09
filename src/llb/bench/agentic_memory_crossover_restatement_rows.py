"""Substitution rows for the crossover restatement: re-measured cells, surfaces, and cap peaks.

Substituting a re-measured cell into a published grid and re-running that grid's own rule is the
whole restatement, so these builders reuse the published machinery rather than reimplementing it:
the surface's precondition gate (`surface_cell_row`) and its predeclared interpolation
(`depth_surface_row`). What each published crossover then becomes is one level up, in
`agentic_memory_crossover_restatement_forms`.

One rule governs where every number comes from: what the restated geometry can measure is measured
here off its own prompt sequence -- the guard ratio's cap peak as much as the fold step -- and what
only the published artifact holds is stated as its own comparison row (`cap_peak_rows`) rather than
divided into a measured one, because a quotient reports no disagreement between its parts.
"""

from typing import cast

from llb.bench.agentic_memory_boundary_crossover import depth_surface_row
from llb.bench.agentic_memory_boundary_surface_cells import surface_cell_row
from llb.bench.agentic_memory_fold_step_ladder import measured_cap_peak
from llb.bench.agentic_memory_crossover_restatement_placement import prompt_sequence
from llb.bench.agentic_memory_crossover_restatement_reading import (
    PEAK_INVARIANT,
    PEAK_MOVED,
    PEAK_UNPUBLISHED,
    REPORTING_CONFIDENCE,
)
from llb.bench.agentic_policy_change_audit import KIND_SURFACE


def restated_cells(
    designs: dict[str, dict[str, object]], restated_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Gate the re-measured cells with the surface's own precondition contract."""
    held = cast(dict[str, object], designs[KIND_SURFACE]["held_fixed"])
    declared = {
        cast(str, cell["cell_id"]): cell
        for cell in cast(
            list[dict[str, object]],
            cast(dict[str, object], designs[KIND_SURFACE]["surface"])["cells"],
        )
    }
    return [
        surface_cell_row(
            row,
            declared[cast(str, row["restated_cell_id"])],
            held_fixed=held,
            confidence=REPORTING_CONFIDENCE,
        )
        for row in restated_rows
    ]


def restated_surfaces(
    designs: dict[str, dict[str, object]],
    published_surface: dict[str, object],
    cells: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Re-run the surface's published interpolation over its cells with the new ones substituted.

    The cap peak the guard ratio is divided by is MEASURED here, off the same prompt sequence the
    restated fold step is read from, rather than read out of the published aggregate. Mixing the two
    would state a freshly interpolated guard as a fraction of a peak a moved task world has retired:
    a changed `pad_chars`, observation cap, or step margin rescales the ratio with nothing in the run
    saying so, and the fold-step check notices only when the same drift also reshapes the step
    ladder, which a small move need not do.
    """
    design = designs[KIND_SURFACE]
    replacements = {cast(str, cell["cell_id"]): cell for cell in cells}
    substituted = [
        replacements.get(cast(str, cell["cell_id"]), cell)
        for cell in cast(list[dict[str, object]], published_surface["cells"])
    ]
    return [
        {
            **depth_surface_row(
                depth,
                [cell for cell in substituted if int(cast(int, cell["depth"])) == depth],
                cap_peak_prompt_chars=measured_cap_peak(
                    prompt_sequence(design, depth),
                    geometry=f"the restated depth {depth} surface",
                ),
            ),
            "restated_cell_ids": sorted(
                cast(str, cell["cell_id"])
                for cell in cells
                if int(cast(int, cell["depth"])) == depth
            ),
        }
        for depth in sorted({int(cast(int, cell["depth"])) for cell in substituted})
    ]


def cap_peak_rows(
    published_surface: dict[str, object], surfaces: list[dict[str, object]]
) -> list[dict[str, object]]:
    """State the published cap peak beside the re-measured one, per restated depth.

    As a row the two are comparable, so a depth whose peak moved NAMES the move, and the ratio beside
    it is the one the geometry that measured the guard supports.
    """
    published = cast(dict[str, object], published_surface["cap_peak_prompt_chars"])
    return [
        _cap_peak_row(
            int(cast(int, surface["depth"])),
            published.get(str(cast(int, surface["depth"]))),
            int(cast(int, surface["cap_peak_prompt_chars"])),
            cast(float | None, surface["crossover_guard_ratio"]),
        )
        for surface in surfaces
    ]


def _cap_peak_row(
    depth: int, published: object | None, measured: int, ratio: float | None
) -> dict[str, object]:
    stated = None if published is None else int(cast(int, published))
    if stated is None:
        reading = PEAK_UNPUBLISHED
    else:
        reading = PEAK_INVARIANT if stated == measured else PEAK_MOVED
    return {
        "study_kind": KIND_SURFACE,
        "depth": depth,
        "published_cap_peak_prompt_chars": stated,
        "measured_cap_peak_prompt_chars": measured,
        "cap_peak_delta_chars": None if stated is None else measured - stated,
        "restated_guard_ratio": ratio,
        "reading": reading,
    }
