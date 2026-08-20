"""Fold-step ladder arithmetic: the trigger interval, the guard inverse, and the foldable ladder.

Every rule here is a pure function of a prompt-size SEQUENCE plus a share or a guard -- no design
file, no episode, no model -- which is why the callers that sweep them (the placement rules, the
summarize-cap ladder, the policy-change band solver) can afford to call them in a tight loop. The
crossover study that READS these rules is tested in `test_agentic_memory_fold_step_crossover.py`.

The happy paths come first, then the EDGES the callers only reach by accident -- an out-of-range
step, an unreachable step read as a bare interval, a trigger nothing exceeds, a share outside
`(0, 1]`, a non-positive peak, and the empty sequence -- so a ladder function that mis-handles one
fails here by name instead of surfacing as a puzzling design-validation or band-solver error.
"""

import pytest

from llb.bench.memory.fold_step.ladder import (
    compaction_trigger_chars,
    first_fold_step,
    fold_step_guard_interval,
    fold_step_trigger_interval,
    foldable_fold_steps,
    live_entries_at_fold_step,
    measured_cap_peak,
    reachable_fold_steps,
    smallest_guard_reaching,
    usable_guard_band,
)

# A strictly growing sequence: every step exceeds the running maximum, so every step is reachable.
GROWING = [3000, 3904, 4792, 5680, 6568, 7456, 8374]
# A sequence whose second prompt does not exceed the first, so step 2's trigger interval is empty.
REPEATED = [3000, 3000, 4000]
# No step at all: every ladder function must answer for a geometry nothing was measured over.
EMPTY: list[int] = []
# The shipped compact share, used wherever an edge case needs a share that is itself valid.
SHARE = 0.5
# Shares outside `(0, 1]`: zero and negative make the guard inverse meaningless, above 1 asks for a
# trigger the budget cannot reach.
REFUSED_SHARES = (0.0, -0.5, 1.5)


def test_the_trigger_interval_inverts_the_fold_step_prediction():
    assert reachable_fold_steps(GROWING) == [1, 2, 3, 4, 5, 6, 7]
    low, high = fold_step_trigger_interval(GROWING, 6)
    assert (low, high) == (6568, 7456)
    # Every trigger inside the interval selects the step, and the first one outside does not.
    assert {first_fold_step(GROWING, trigger) for trigger in range(low, high)} == {6}
    assert first_fold_step(GROWING, high) == 7


def test_the_guard_interval_inverts_the_trigger_through_the_runtime_truncation():
    guard_low, guard_high = fold_step_guard_interval(GROWING, 6, 0.5)
    assert (guard_low, guard_high) == (13136, 14912)
    assert first_fold_step(GROWING, compaction_trigger_chars(guard_high - 1, 0.5)) == 6
    assert first_fold_step(GROWING, compaction_trigger_chars(guard_high, 0.5)) == 7
    # The guard is resolved against the runtime's truncating arithmetic, not a float inverse.
    assert compaction_trigger_chars(smallest_guard_reaching(7456, 0.45), 0.45) >= 7456
    assert compaction_trigger_chars(smallest_guard_reaching(7456, 0.45) - 1, 0.45) < 7456


def test_an_unreachable_step_and_an_unfoldable_step_both_leave_the_ladder():
    # A step whose prompt does not exceed the running maximum can never be selected.
    assert reachable_fold_steps(REPEATED) == [1, 3]
    # A trigger can SELECT step 1; no episode folds there, because its prompt is built from zero
    # entries and `compact_state` has nothing older to summarize.
    assert live_entries_at_fold_step(1) == 0
    assert foldable_fold_steps(GROWING) == [2, 3, 4, 5, 6, 7]
    assert foldable_fold_steps(REPEATED) == [3]


def test_a_step_outside_the_sequence_is_refused_rather_than_read():
    # Steps are 1-based, so both 0 and one past the end are off the ladder. Refusing here is what
    # keeps a caller that fabricated a step from reading a neighbouring step's interval as its own.
    for step in (0, len(GROWING) + 1):
        with pytest.raises(ValueError, match="outside a 7-step sequence"):
            fold_step_trigger_interval(GROWING, step)
    with pytest.raises(ValueError, match="outside a 7-step sequence"):
        fold_step_guard_interval(GROWING, 0, SHARE)


def test_an_unreachable_step_reads_as_an_empty_trigger_interval():
    # `reachable_fold_steps` already drops step 2 of REPEATED; read directly, the emptiness is the
    # interval itself -- `low >= high`, because step 1 already reached this step's size.
    low, high = fold_step_trigger_interval(REPEATED, 2)
    assert (low, high) == (3000, 3000)
    assert low >= high
    # No trigger selects it: everything below the interval folds sooner, everything at or above it
    # folds later.
    assert first_fold_step(REPEATED, low - 1) == 1
    assert first_fold_step(REPEATED, high) == 3
    guard_low, guard_high = fold_step_guard_interval(REPEATED, 2, SHARE)
    assert guard_low >= guard_high


def test_no_prompt_above_the_trigger_folds_at_no_step():
    # The trigger is crossed STRICTLY, so a trigger equal to the largest prompt already never fires.
    assert first_fold_step(GROWING, GROWING[-1]) is None
    assert first_fold_step(GROWING, GROWING[-1] + 1) is None
    # One char lower is the last step, which is where the None boundary sits.
    assert first_fold_step(GROWING, GROWING[-1] - 1) == len(GROWING)


def test_the_guard_inverse_refuses_a_share_outside_the_unit_interval():
    for share in REFUSED_SHARES:
        with pytest.raises(ValueError, match="compact share must be in"):
            smallest_guard_reaching(7456, share)
    # A share of exactly 1 is the closed end and is answered: the guard IS the trigger.
    assert smallest_guard_reaching(7456, 1.0) == 7456


def test_the_usable_band_refuses_a_non_positive_peak_and_an_out_of_range_share():
    # A non-positive peak means the probe measured no prompt at all; a band around it would name a
    # guard interval for a geometry that never ran.
    for peak in (0, -1):
        with pytest.raises(ValueError, match="peak prompt chars must be positive"):
            usable_guard_band(peak, SHARE)
    for share in REFUSED_SHARES:
        with pytest.raises(ValueError, match="compact share must be in"):
            usable_guard_band(8374, share)
    # At a share of 1 the band is empty rather than refused: the trigger reaches the peak exactly
    # where the cap stops fitting, so no guard satisfies both ends.
    low, high = usable_guard_band(8374, 1.0)
    assert low == high == 8374


def test_the_peak_read_names_the_geometry_a_bare_max_would_only_call_an_empty_iterable():
    """The one reduction every cap-fitting caller takes, so the refusal is stated once for all.

    `usable_guard_band` refuses a non-positive peak, but a walk that measured NOTHING never gets
    that far: the builtin `max` fails first with a message naming neither the geometry nor the peak
    it was reducing to. The checked read closes that gap ABOVE the band rather than inside it.
    """
    assert measured_cap_peak(GROWING, geometry="depth 6") == 8374
    assert measured_cap_peak(REPEATED, geometry="depth 6") == 4000
    for sequence in (EMPTY, [0, 0, 0]):
        with pytest.raises(
            ValueError, match=rf"depth 6 measured no prompt \({len(sequence)} steps\)"
        ):
            measured_cap_peak(sequence, geometry="depth 6")
    # The geometry is the caller's own vocabulary, so an arm ladder and a surface depth read
    # differently even though both reduce the same kind of walk.
    with pytest.raises(ValueError, match="the depth 10 ladder measured no prompt"):
        measured_cap_peak(EMPTY, geometry="the depth 10 ladder")
    # The peak the band is built on is exactly what the checked read returns.
    assert usable_guard_band(measured_cap_peak(GROWING, geometry="depth 6"), SHARE) == (8374, 16748)


def test_the_whole_family_answers_for_an_empty_sequence():
    # Nothing folds, no step is reachable or foldable, and every step is out of range -- so a
    # geometry with no measured prompts fails here rather than two layers up in a placement rule.
    assert first_fold_step(EMPTY, 0) is None
    assert reachable_fold_steps(EMPTY) == []
    assert foldable_fold_steps(EMPTY) == []
    with pytest.raises(ValueError, match="outside a 0-step sequence"):
        fold_step_trigger_interval(EMPTY, 1)
