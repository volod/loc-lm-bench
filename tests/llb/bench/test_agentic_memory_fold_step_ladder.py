"""Fold-step ladder arithmetic: the trigger interval, the guard inverse, and the foldable ladder.

Every rule here is a pure function of a prompt-size SEQUENCE plus a share or a guard -- no design
file, no episode, no model -- which is why the callers that sweep them (the placement rules, the
summarize-cap ladder, the policy-change band solver) can afford to call them in a tight loop. The
crossover study that READS these rules is tested in `test_agentic_memory_fold_step_crossover.py`.
"""

from llb.bench.agentic_memory_fold_step_ladder import (
    compaction_trigger_chars,
    first_fold_step,
    fold_step_guard_interval,
    fold_step_trigger_interval,
    foldable_fold_steps,
    live_entries_at_fold_step,
    reachable_fold_steps,
    smallest_guard_reaching,
)

# A strictly growing sequence: every step exceeds the running maximum, so every step is reachable.
GROWING = [3000, 3904, 4792, 5680, 6568, 7456, 8374]
# A sequence whose second prompt does not exceed the first, so step 2's trigger interval is empty.
REPEATED = [3000, 3000, 4000]


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
