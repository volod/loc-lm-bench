"""The cap's silence is about the prompts the loop BUILDS, not only the ones it sends.

`observation_cap_chars` answers every pair it belongs to with `no geometry`, and that answer used to
rest on "a differently trimmed observation is shown in the next step's prompt" -- which is false at a
fold, because `step_prompt` discards the prompt it was building before it rebuilds. These tests hold
the widened answer to both halves: the real geometry's blocked steps stay blocked for the same
reason they always were, and the corner the old wording assumed away is closed by arithmetic that is
exercised on sequences the shipped task world does not produce.
"""

import pytest

from llb.bench.agentic.context import (
    DEFAULT_COMPACT_KEEP_RECENT,
    POLICY_COMPACT,
    ContextPolicy,
    ContextState,
    compact_state,
)
from llb.bench.agentic_memory_boundary_probe import cap_prompt_sequence, smallest_guard_reaching
from llb.bench.agentic_policy_change_interaction_band import (
    format_band_report,
    separating_guard_bands,
)
from llb.bench.agentic_policy_change_interaction_cap import (
    CONDITION_MOVES_FOLD,
    CONDITION_SILENT,
    MovedPrompts,
    cap_step_conditions,
    folded_entries,
    moved_prompts,
)
from llb.bench.agentic_policy_change_interaction_couplings import coupling_for, held_baseline
from llb.bench.agentic_policy_change_interaction_fixture import (
    geometry_kwargs,
    load_interaction_design,
)
from llb.bench.agentic_policy_change_interaction_terms import (
    FIELD_BOUND,
    FIELD_CAP,
    FIELD_SHARE,
    StepGeometry,
    foldable_fold_steps,
)

# The depth the committed fixture separates at -- the geometry the `no geometry` answers are read on.
FIXTURE_DEPTH = 10
# Every pair whose answer the cap conditions carry.
CAP_PAIRS = [(FIELD_CAP, FIELD_SHARE), (FIELD_CAP, FIELD_BOUND)]


@pytest.fixture(scope="module")
def held() -> dict[str, object]:
    return load_interaction_design()["held_fixed"]


def _step_geometry(coupling, held: dict[str, object], step: int, sequence: list[int]):
    return StepGeometry(
        change=coupling.example_change(),
        step=step,
        sequence=sequence,
        geometry=geometry_kwargs(FIXTURE_DEPTH, held_baseline(coupling, held)),
        share=float(held[FIELD_SHARE]),
    )


@pytest.mark.parametrize("pair", CAP_PAIRS, ids=lambda pair: f"{pair[0]} x {pair[1]}")
def test_a_known_blocked_step_stays_blocked_for_the_stated_reason(
    pair: tuple[str, str], held: dict[str, object]
):
    """The widened conditions must not open a band on the geometry that had none."""
    coupling = coupling_for(pair)
    change = coupling.example_change()
    geometry = held_baseline(coupling, held)
    bands = separating_guard_bands(change, depth=FIXTURE_DEPTH, held=geometry, include_empty=True)
    report = format_band_report(change, FIXTURE_DEPTH, bands)
    assert bands, report
    for band in bands:
        assert band.is_empty, report
        # And blocked by the cap's own silence: the shipped cap move re-trims EVERY observation, so
        # no fold can discard all the entries it moves and the union sees it audited alone.
        assert CONDITION_SILENT in {condition.name for condition in band.blocked_by()}, report


def test_the_shipped_cap_move_is_shown_rather_than_discarded(held: dict[str, object]):
    """Why that answer is a proof here: the moved entries outlive every fold the geometry has."""
    geometry = geometry_kwargs(FIXTURE_DEPTH, held)
    caps = (int(held["observation_cap_chars"]), 1600)
    sequences = [cap_prompt_sequence(**{**geometry, "observation_cap_chars": cap}) for cap in caps]
    moved = moved_prompts(*sequences)
    assert moved is not None
    for step in foldable_fold_steps(sequences[0]):
        discarded = folded_entries(step, DEFAULT_COMPACT_KEEP_RECENT)
        assert max(moved.entries) > discarded, (step, discarded, moved.entries)


def test_moved_entries_are_the_first_difference_of_the_prompt_sequence():
    """A prompt is the scaffold plus one line per entry, so an entry's own length is that step."""
    before = [3000, 3900, 4800, 5700]
    assert moved_prompts(before, list(before)) is None
    # Only entry 2 renders differently: prompts move from step 3 on.
    after = [3000, 3900, 4850, 5750]
    moved = moved_prompts(before, after)
    assert moved == MovedPrompts(entries=(2,), step=3, smallest=4800, largest=4850)
    # Two entries that cancel still MOVED, which a prompt-size comparison at step 4 would miss.
    canceling = [3000, 3900, 4850, 5700]
    assert moved_prompts(before, canceling).entries == (2, 3)


def test_different_step_counts_are_a_moved_episode_no_fold_can_discard(held: dict[str, object]):
    """A cap that changes how many steps the oracle plays is not hiding behind anything."""
    moved = moved_prompts([3000, 3900, 4800], [3000, 3900])
    assert moved == MovedPrompts.whole_episode(2)
    assert moved.smallest == moved.largest == 0
    # And it still ANSWERS: a zero-width move admits no guard, so the step reports a blocked
    # condition rather than a band whose emptiness nothing accounts for.
    coupling = coupling_for((FIELD_CAP, FIELD_SHARE))
    step = _step_geometry(coupling, held, step=2, sequence=[3000, 3900, 4800])
    stated = cap_step_conditions(step, [3000, 3900, 4800], [3000, 3900])
    assert [condition.name for condition in stated.conditions if condition.is_empty] == [
        CONDITION_SILENT,
        CONDITION_MOVES_FOLD,
    ]


@pytest.mark.parametrize("keep_recent", [1, 2, 3])
@pytest.mark.parametrize("live", [0, 1, 2, 5])
def test_folded_entries_is_what_compact_state_actually_folds(live: int, keep_recent: int):
    """The corner rests on this count, so it is measured against the fold rather than assumed."""
    state = ContextState(entries=[("search", {}, f"hit {index}") for index in range(live)])
    policy = ContextPolicy(name=POLICY_COMPACT, compact_keep_recent=keep_recent)
    folded = compact_state(policy, state, lambda older: "summary")
    # `live_entries_at_fold_step(step) == step - 1`, so the fold at step `live + 1` sees `live`.
    expected = folded_entries(live + 1, keep_recent)
    assert state.n_dropped == expected
    assert folded is (expected > 0)
    assert len(state.entries) == live - expected


def test_a_cap_confined_to_the_discarded_prompt_leaves_the_conditions_no_guard(
    held: dict[str, object],
):
    """The corner itself: only the folded entry moves, and the two bounds still cross.

    The shipped task world cannot build this -- its observations are uniform, so a cap that re-trims
    the first one re-trims them all -- so the sequences are stated directly. Silence needs a guard
    BELOW the moved prompt (the `observation_cap` arm has no fold to hide it behind and would send
    it), and the compound needs a trigger that REACHES the moved prompt, which no smaller guard has:
    a share is at most 1, so `smallest_guard_reaching` never returns less than what it reaches.
    """
    coupling = coupling_for((FIELD_CAP, FIELD_SHARE))
    # Entry 1 moves and nothing else does, and the fold at step 2 has one live entry -- the
    # fold-everything fallback -- so the only prompt the caps move is the one it discards.
    before = [3000, 3900, 4800, 5700]
    after = [3000, 3950, 4850, 5750]
    step = _step_geometry(coupling, held, step=2, sequence=before)
    assert folded_entries(step.step, DEFAULT_COMPACT_KEEP_RECENT) == 1
    moved = moved_prompts(before, after)
    assert moved == MovedPrompts(entries=(1,), step=2, smallest=3900, largest=3950)

    stated = cap_step_conditions(step, before, after)
    conditions = {condition.name: condition for condition in stated.conditions}
    silent, moves = conditions[CONDITION_SILENT], conditions[CONDITION_MOVES_FOLD]
    # Neither statement is impossible on its own -- the corner is real, not ruled out by fiat.
    assert not silent.is_empty and not moves.is_empty
    assert silent.high == moved.smallest
    assert moves.low >= moved.smallest >= silent.high


@pytest.mark.parametrize("share", [0.05, 0.3, 0.48, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("prompt_chars", [1, 800, 3900, 11926, 21500])
def test_a_guard_whose_trigger_reaches_a_prompt_is_at_least_that_prompt(
    prompt_chars: int, share: float
):
    """The inequality that closes the corner, stated on its own so a share change cannot hide it."""
    assert smallest_guard_reaching(prompt_chars, share) >= prompt_chars
