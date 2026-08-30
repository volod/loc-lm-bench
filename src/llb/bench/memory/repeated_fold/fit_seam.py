"""Where a per-family guard fit plugs into a run: the cell order, and the runner's guard seam.

`guard_fit` decides one cell's guard from what a family measured. Neither half of that is much use
until a RUN can call it, and the calling has two constraints the fit itself has no opinion about:

  - the fit reads cells that must already have run, so the declared ladder order (by fold count,
    which is how a reader sees the rungs) is not the order the study can execute;
  - only ONE cell is fitted, so every other cell has to come back exactly as declared, or a design
    that adds a fit would silently move geometry it never asked to move.

Both live here so the fit stays a pure function of measurements and this module stays the only
place that knows what order a fitted study runs in.
"""

from typing import Callable

from llb.bench.agentic.design_fields import as_mapping, as_str
from llb.bench.memory.repeated_fold.design import completion_cells
from llb.bench.memory.repeated_fold.fold_span import fold_length_span_model, measured_fold_points
from llb.bench.memory.repeated_fold.guard_fit import (
    apply_fitted_guard,
    fit_fold_guard,
    guard_fit_spec,
    measured_fold_lengths,
    measured_step_entry_chars,
    span_length_source,
    step_length_source,
)


def guard_resolver(
    design: dict[str, object], *, evidence_floor: int
) -> (
    Callable[
        [dict[str, object], list[dict[str, object]]],
        tuple[dict[str, object], dict[str, object] | None],
    ]
    | None
):
    """The completion runner's guard seam, or `None` when the design declares no fit.

    Every cell but the fitted one is handed back exactly as declared, so a design that adds a fit
    changes the geometry of ONE rung and leaves the control and the deeper cell untouched.
    """
    spec = guard_fit_spec(design)
    if not spec:
        return None
    held = as_mapping(design, "held_fixed")
    fitted_cell_id = as_str(spec, "cell_id")
    source = as_str(spec, "fold_length_source")
    step_source = step_length_source(spec)
    span_source = span_length_source(spec)

    def resolve(
        cell: dict[str, object], rows: list[dict[str, object]]
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        if cell.get("cell_id") != fitted_cell_id:
            return cell, None
        lengths = measured_fold_lengths(rows, source)
        record = fit_fold_guard(
            design,
            cell,
            held,
            lengths,
            evidence_floor=evidence_floor,
            step_entry_chars=measured_step_entry_chars(rows, step_source),
            span_model=fold_length_span_model(
                measured_fold_points(rows, source),
                measured_fold_points(rows, span_source) if span_source else [],
            ),
        )
        return apply_fitted_guard(cell, record), record

    return resolve


def fitted_cell_order(design: dict[str, object]) -> list[dict[str, object]]:
    """The cells in the order a fitted study must RUN them, not the order it declares them.

    The declaration is a ladder, ordered by fold count so a reader can see the rungs. The run has
    one further constraint the declaration cannot express: every cell a fit measures against has
    to have run already. So the cap-fitting control stays first -- it is the eligibility gate, and
    a family that cannot fold once anchors nothing -- and the fitted cell moves LAST, behind the
    never-fitted cells whose folds give the replay its second measured span.

    The reordering is not free and its cost is named where the marker ablation is read: cell
    POSITION in the run no longer increases with fold count, so a drift in the stateful endpoint
    over the run is no longer aligned with the rung being measured.
    """
    cells = completion_cells(design)
    fitted = guard_fit_spec(design).get("cell_id")
    if not fitted:
        return cells
    return [cell for cell in cells if cell.get("cell_id") != fitted] + [
        cell for cell in cells if cell.get("cell_id") == fitted
    ]
