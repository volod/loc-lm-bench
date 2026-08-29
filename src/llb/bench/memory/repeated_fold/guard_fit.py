"""Fitting one fold-count cell's prompt guard to the family that will run it.

A fold-count ladder holds the GUARD fixed and reads whatever fold count each family lands on. That
works while every family writes summaries of a similar length and stops working the moment one does
not: the running summary sits in every later prompt, so a family whose summaries run long spends
more of the guard before the transcript grows at all, re-crosses the compact trigger sooner, and
folds again on the geometry another family folds twice on. The rung is then empty for that family
-- not because the fold count is hard to reach, but because one shared character constant was fit
against a summarizer nobody ran.

What this module fits instead is the guard, per family, from that family's OWN measured fold length.
The measurement is free: the one-fold control already folds exactly once per case, so its telemetry
carries one summarizer output length per case before any other cell runs. The fit itself is
model-free -- the deterministic oracle walk, replayed with a summarizer that writes exactly the
measured number of characters (`fold_length_controller`), over a predeclared band of candidate
guards. What is held equal across families is therefore the RUNG -- the measured fold count -- and
the guard is what each family needs to stand on it.

The replay takes TWO measurements off that control arm, not one. The fold length says how much of
the guard a post-fold prompt already spends; the per-step entry length says how fast the rest of it
is spent, and a replay that assumed the oracle's own minimal call there was counting folds on a
transcript that grows at nobody's measured rate. Both are free and both are the family's own, which
is what a per-guard count has to stand on before an operator can read it without a confirming run.

What the control cannot say on its own is WHICH FOLD its length came from: it folds once over the
whole transcript, while the fitted cell folds several shorter spans. A second never-fitted cell,
run before the fitted one, supplies the second (offered span, written length) point that turns the
control's single length into the length the fitted cell's own span implies -- see
`llb.bench.memory.repeated_fold.fold_span`.
"""

from typing import Callable, cast

from llb.bench.agentic.design_fields import as_int, as_mapping, as_str
from llb.bench.context_policy.guard_band import guard_grid, median_int, select_guard
from llb.bench.memory.repeated_fold.design import completion_cells
from llb.bench.memory.repeated_fold.fold_span import (
    FoldLengthModel,
    fold_length_span_model,
    measured_fold_points,
    shipped_arm_cases,
)
from llb.bench.memory.repeated_fold.guard_replay import (
    fold_count_margin_chars,
    fold_count_predictor,
    oracle_step_entry_chars,
)

GUARD_FIT_FIELD = "two_fold_guard_fit"

FIT_APPLIED = "guard_fitted_to_the_measured_fold_length"
FIT_DECLARED = "declared_guard_already_fits_the_measured_fold_length"
FIT_UNDERPOWERED = "no_guard_in_the_declared_band_reaches_the_evidence_floor"
FIT_UNMEASURED = "the_control_measured_no_fold_length_to_fit_against"

# What the control's step measurement says about the walk the probe replays. The fold length is
# only half of what decides a fold count: the other half is how fast the transcript grows between
# folds, which the replay used to take from the oracle's own minimal call rather than from the
# family. These readings state which of the two the run actually stood on.
STEP_LENGTH_AT_ORACLE = "the_family_steps_render_at_the_oracle_walk_length"
STEP_LENGTH_ABOVE_ORACLE = "the_family_steps_render_longer_than_the_oracle_walk"
STEP_LENGTH_UNMEASURED = "the_control_measured_no_step_length_to_replay"


def guard_fit_spec(design: dict[str, object]) -> dict[str, object]:
    """The predeclared fit: which cell, which fold count, and which band of guards to search."""
    return as_mapping(design, GUARD_FIT_FIELD)


def measured_fold_lengths(rows: list[dict[str, object]], source_cell_id: str) -> list[int]:
    """Every summarizer output length the shipped-policy control arm actually measured.

    One length per FOLD, not per case: the control folds once per case, so the two coincide there,
    and a source cell that folded more would contribute each fold it made.
    """
    return [
        int(chars)
        for case in shipped_arm_cases(rows, source_cell_id)
        for chars in cast(list[int], case.get("summary_output_chars", []))
    ]


def measured_step_entry_chars(rows: list[dict[str, object]], source_cell_id: str) -> list[int]:
    """Every per-step transcript entry the shipped-policy control arm actually appended.

    One length per STEP, across every case: the replay walks one synthetic episode, so what it
    needs from a case set is the length a step of it typically renders at.
    """
    return [
        int(chars)
        for case in shipped_arm_cases(rows, source_cell_id)
        for chars in cast(list[int], case.get("step_entry_chars", []))
    ]


def fit_fold_guard(
    design: dict[str, object],
    cell: dict[str, object],
    held: dict[str, object],
    fold_lengths: list[int],
    *,
    evidence_floor: int,
    step_entry_chars: list[int] | None = None,
    span_model: FoldLengthModel | None = None,
) -> dict[str, object]:
    """Choose this family's guard for one fold-count cell, or say why the declared one stands.

    Every input comes off cells the family has already run: the fold length its summarizer wrote,
    the per-step entry its calls appended, and the slope between the spans two of those cells
    offered. The first decides how much of the guard a post-fold prompt starts out spending; the
    second decides how fast the rest of it is spent; the third corrects the first for the fact
    that the cell it was measured on folds a longer span than the cell it is replayed at. A replay
    given only the first counts folds on a transcript that grows at the ORACLE's rate, out of a
    summary written against a span this cell never offers.
    """
    spec = guard_fit_spec(design)
    spans = span_model or fold_length_span_model([], [])
    declared = as_int(cell, "max_prompt_chars")
    target = as_int(spec, "target_folds")
    steps = list(step_entry_chars or [])
    replayed_step = median_int(steps)
    oracle_step = oracle_step_entry_chars(cell, held)
    record: dict[str, object] = {
        "cell_id": cell["cell_id"],
        "target_folds": target,
        "declared_max_prompt_chars": declared,
        "fitted_max_prompt_chars": declared,
        "fold_length_source": as_str(spec, "fold_length_source"),
        "fold_lengths": fold_lengths,
        "median_fold_length_chars": median_int(fold_lengths),
        "step_length_source": step_length_source(spec),
        "median_step_entry_chars": replayed_step,
        "step_entry_chars_range": [min(steps), max(steps)] if steps else [],
        "oracle_step_entry_chars": oracle_step,
        "step_length_reading": _step_length_reading(replayed_step, oracle_step),
        "span_length_source": span_length_source(spec),
        **spans.as_record(),
        "evidence_floor": evidence_floor,
    }
    if not fold_lengths:
        return {
            **record,
            "predicted_target_cases": 0,
            "declared_target_cases": 0,
            "meets_evidence_floor": False,
            "fit_reading": FIT_UNMEASURED,
            "fit_reason": (
                f"cell {record['fold_length_source']!r} measured no summarizer output, so the "
                f"declared guard {declared} stands unfitted"
            ),
        }
    predict = fold_count_predictor(cell, held, replayed_step, spans)
    counts = {
        guard: _target_cases(predict, guard, fold_lengths, target) for guard in guard_grid(spec)
    }
    fitted = select_guard(counts, declared)
    best = counts[fitted]
    median_fold = median_int(fold_lengths)
    return {
        **record,
        "fitted_max_prompt_chars": fitted,
        "predicted_target_cases": best,
        "declared_target_cases": counts.get(declared, 0),
        "fold_count_margin_chars": fold_count_margin_chars(predict, fitted, median_fold),
        "declared_fold_count_margin_chars": fold_count_margin_chars(predict, declared, median_fold),
        "meets_evidence_floor": best >= evidence_floor,
        "fit_reading": _fit_reading(fitted, declared, best, evidence_floor),
        "fit_reason": (
            f"guard {fitted} puts {best} of {len(fold_lengths)} measured fold lengths on "
            f"{target} folds (declared guard {declared} puts {counts.get(declared, 0)}); "
            f"replayed at {replayed_step}-char steps and "
            f"{spans.chars_per_offered_char:+.3f} written chars per offered char; "
            f"floor {evidence_floor}"
        ),
    }


def apply_fitted_guard(cell: dict[str, object], record: dict[str, object]) -> dict[str, object]:
    """The cell as it will actually run: declared in every field except the fitted guard."""
    return {**cell, "max_prompt_chars": int(cast(int, record["fitted_max_prompt_chars"]))}


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


def step_length_source(spec: dict[str, object]) -> str:
    """Which cell the per-step entry length is replayed from.

    Defaulted to the fold-length source, because it is the same control arm and the same free
    telemetry; a design may name it explicitly so the two are auditable apart.
    """
    return str(spec.get("step_length_source") or spec.get("fold_length_source") or "")


def span_length_source(spec: dict[str, object]) -> str:
    """Which cell supplies the SECOND (offered span, written length) point, if any.

    Optional by design: a study that declares none keeps the flat replay, named as such, rather
    than getting a slope invented for it. What a design may not do is name the fitted cell here --
    a fit that measured its own cell would be reading the answer it is predicting.
    """
    return str(spec.get("span_length_source") or "")


def _step_length_reading(measured: int, oracle: int) -> str:
    """Whether the family's own steps grew the transcript faster than the oracle walk does."""
    if measured <= 0:
        return STEP_LENGTH_UNMEASURED
    return STEP_LENGTH_ABOVE_ORACLE if measured > oracle else STEP_LENGTH_AT_ORACLE


def _fit_reading(fitted: int, declared: int, best: int, floor: int) -> str:
    if best < floor:
        return FIT_UNDERPOWERED
    return FIT_DECLARED if fitted == declared else FIT_APPLIED


def _target_cases(
    predict: Callable[[int, int], int], guard: int, fold_lengths: list[int], target: int
) -> int:
    return sum(predict(guard, length) == target for length in fold_lengths)
