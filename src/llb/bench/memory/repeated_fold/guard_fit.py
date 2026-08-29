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
"""

from typing import Callable, cast

from llb.bench.agentic.design_fields import as_int, as_mapping, as_str
from llb.bench.context_policy.guard_band import guard_grid, median_int, select_guard
from llb.bench.memory.boundary.probe import compact_fold_input_probe, fold_length_controller
from llb.bench.memory.repeated_fold.design import cell_geometry

GUARD_FIT_FIELD = "two_fold_guard_fit"
ARM_TYPED_MARKER = "typed_marker"

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

# How far the replayed fold length is scanned, either side, for the point where a guard's predicted
# fold count changes. That distance is the SLACK a per-guard count has: the replay takes its fold
# length off the control's fold, which covers a different span of transcript than the fitted cell's
# folds do, so the length handed to the probe is always a little wrong. A guard whose count survives
# this much error in it is one an operator can read without a confirming run; a guard whose count
# flips within it is one the fit can rank but not count. The scan is coarse on purpose -- the answer
# wanted is which side of the replay error the flip sits on, not the flip point to the character.
FOLD_COUNT_MARGIN_SCAN_CHARS = 200
FOLD_COUNT_MARGIN_STEP_CHARS = 10


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
        for row in rows
        if row["cell_id"] == source_cell_id and row["arm"] == ARM_TYPED_MARKER
        for case in cast(list[dict[str, object]], row["cases"])
        for chars in cast(list[int], case.get("summary_output_chars", []))
    ]


def measured_step_entry_chars(rows: list[dict[str, object]], source_cell_id: str) -> list[int]:
    """Every per-step transcript entry the shipped-policy control arm actually appended.

    One length per STEP, across every case: the replay walks one synthetic episode, so what it
    needs from a case set is the length a step of it typically renders at.
    """
    return [
        int(chars)
        for row in rows
        if row["cell_id"] == source_cell_id and row["arm"] == ARM_TYPED_MARKER
        for case in cast(list[dict[str, object]], row["cases"])
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
) -> dict[str, object]:
    """Choose this family's guard for one fold-count cell, or say why the declared one stands.

    Both inputs come off the control arm the family has already run: the fold length its
    summarizer wrote, and the per-step entry its calls appended. The first decides how much of the
    guard a post-fold prompt starts out spending; the second decides how fast the rest of it is
    spent. A replay given only the first counts folds on a transcript that grows at the ORACLE's
    rate, which is why its absolute per-guard count needed a confirming run to be safe.
    """
    spec = guard_fit_spec(design)
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
    predict = _fold_count_predictor(cell, held, replayed_step)
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
            f"replayed at {replayed_step}-char steps; floor {evidence_floor}"
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
        )
        return apply_fitted_guard(cell, record), record

    return resolve


def step_length_source(spec: dict[str, object]) -> str:
    """Which cell the per-step entry length is replayed from.

    Defaulted to the fold-length source, because it is the same control arm and the same free
    telemetry; a design may name it explicitly so the two are auditable apart.
    """
    return str(spec.get("step_length_source") or spec.get("fold_length_source") or "")


def oracle_step_entry_chars(cell: dict[str, object], held: dict[str, object]) -> int:
    """What one step of the UNPADDED oracle walk appends to this cell's transcript.

    Measured over the geometry rather than assumed, because the world decides most of it: the
    workflow token the tool hands out is what the call carries, so the length is a property of the
    task set and the depth. It is the baseline a family's measured step length is read against --
    a family at this number grows its context exactly as fast as the walk the probe counts on.
    """
    return median_int(
        cast(
            list[int],
            compact_fold_input_probe(**cell_geometry(cell, held))["step_entry_chars"],  # type: ignore[arg-type]
        )
    )


def fold_count_margin_chars(
    predict: Callable[[int, int], int], guard: int, fold_length: int
) -> int:
    """How far the replayed fold length can be wrong before this guard's fold count changes.

    Scanned both ways rather than assumed monotone: a longer running summary usually crosses the
    trigger sooner, but it also folds sooner, and what a fold leaves behind is not monotone in what
    it replaced. The answer is the largest offset at which the count still holds, capped by the
    scan -- a guard that survives the whole scan is not what an inexact replay can break.
    """
    if fold_length <= 0:
        return 0
    base = predict(guard, fold_length)
    for offset in range(
        FOLD_COUNT_MARGIN_STEP_CHARS,
        FOLD_COUNT_MARGIN_SCAN_CHARS + 1,
        FOLD_COUNT_MARGIN_STEP_CHARS,
    ):
        below = predict(guard, max(0, fold_length - offset))
        above = predict(guard, fold_length + offset)
        if below != base or above != base:
            return offset - FOLD_COUNT_MARGIN_STEP_CHARS
    return FOLD_COUNT_MARGIN_SCAN_CHARS


def _step_length_reading(measured: int, oracle: int) -> str:
    """Whether the family's own steps grew the transcript faster than the oracle walk does."""
    if measured <= 0:
        return STEP_LENGTH_UNMEASURED
    return STEP_LENGTH_ABOVE_ORACLE if measured > oracle else STEP_LENGTH_AT_ORACLE


def _fit_reading(fitted: int, declared: int, best: int, floor: int) -> str:
    if best < floor:
        return FIT_UNDERPOWERED
    return FIT_DECLARED if fitted == declared else FIT_APPLIED


def _fold_count_predictor(
    cell: dict[str, object], held: dict[str, object], step_entry_chars: int
) -> Callable[[int, int], int]:
    """Fold count as a pure function of (guard, fold length), memoized over the search.

    The walk is deterministic and every task in the set produces the same fold count for a given
    pair, so one probe answers for the whole case set and the grid costs one probe per distinct
    pair rather than one per case. The measured step length is fixed for the whole fit -- it is
    one family's walk, not one case's -- so it stays out of the cache key.
    """
    cache: dict[tuple[int, int], int] = {}

    def predict(guard: int, fold_length: int) -> int:
        key = (guard, fold_length)
        if key not in cache:
            geometry = cell_geometry({**cell, "max_prompt_chars": guard}, held)
            probe = compact_fold_input_probe(
                controller=fold_length_controller(fold_length, step_entry_chars=step_entry_chars),
                **geometry,  # type: ignore[arg-type]
            )
            cache[key] = int(cast(int, probe["n_compactions"]))
        return cache[key]

    return predict


def _target_cases(
    predict: Callable[[int, int], int], guard: int, fold_lengths: list[int], target: int
) -> int:
    return sum(predict(guard, length) == target for length in fold_lengths)
