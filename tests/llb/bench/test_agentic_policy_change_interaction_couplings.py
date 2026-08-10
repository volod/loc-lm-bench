"""One pair of constants tests the compound guarantee -- and these are the other fourteen.

The compound policy-change reading is held apart from the per-field union by ONE coupling, so the
honest question is not "does that coupling still separate" (the fixture answers that) but "is it the
only one that can". This module holds the enumeration to that: every pair of `AUDITABLE_FIELDS` is
answered by the band arithmetic, and the answers are checked BOTH ways -- the conditions the solver
intersects, and a replay scan that asks the loop itself.

The wide scan is `slow`; `make ci` scans the two guards where the one separating pair actually
separates, which is where another pair would have to separate too if the arithmetic were wrong. Both
the fixture scan and the wide one sweep `FIELD_CANDIDATE_GRID`, so a silent answer is across the
values a commit could ship rather than only the neighbour `FIELD_MOVES` names.
"""

import itertools

import pytest

from llb.bench.agentic.context_policy import (
    POLICY_COMPACT,
    POLICY_OBSERVATION_CAP,
    ContextPolicy,
)
from llb.bench.agentic.context_summary import is_summary_prompt
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_memory_fold_step_ladder import live_entries_at_fold_step
from llb.bench.agentic_memory_transcript import build_memory_dependent_tasks
from llb.bench.agentic_policy_change_audit import AUDITABLE_FIELDS, PolicyChange, audit_cell_prompts
from llb.bench.agentic_policy_change_interaction_band import (
    format_band_report,
    separating_guard_bands,
)
from llb.bench.agentic_policy_change_interaction_terms import (
    FIELD_BOUND,
    FIELD_CAP,
    FIELD_HEAD,
    FIELD_KEEP_LAST,
    FIELD_KEEP_RECENT,
    FIELD_SHARE,
)
from llb.bench.agentic_policy_change_interaction_couplings import (
    ANSWER_BAND,
    COUPLINGS,
    FIELD_CANDIDATE_GRID,
    FIELD_MOVES,
    Coupling,
    coupling_for,
    held_baseline,
    separating_couplings,
)
from llb.bench.agentic_policy_change_interaction_fixture import (
    INTERACTING_FIELDS,
    interaction_change,
    load_interaction_design,
)
from llb.bench.agentic_policy_change_interaction_scan import (
    check_baseline_is_the_replayed_one,
    format_scan_report,
    format_value_sweep_report,
    scan_couplings_across_candidate_values,
    scan_separating_cells,
)
from llb.bench.agentic_policy_change_replay import prompt_sequence_digest, replay_episode

# The depth the committed fixture separates at, and the two guards it separates on. A second pair
# would have to be visible here if the arithmetic that rules the others out were wrong.
FIXTURE_DEPTH = 10
FIXTURE_GUARDS = [21500, 23700]
# The wide grid the `slow` scan walks: shallow / fixture / deep, and guards from an episode that
# dies immediately to one that never compacts. The committed guards are in the ladder because the
# separating bands are only a few hundred chars wide -- a round ladder alone steps over them, and a
# scan that cannot find the ONE separation it knows about proves nothing about the other fourteen.
SCAN_DEPTHS = [6, 10, 14]
SCAN_GUARDS = sorted(set(range(2000, 34001, 1000)) | set(FIXTURE_GUARDS))


@pytest.fixture(scope="module")
def design() -> dict[str, object]:
    return load_interaction_design()


@pytest.fixture(scope="module")
def held(design: dict[str, object]) -> dict[str, object]:
    return design["held_fixed"]


def _other_couplings() -> list[Coupling]:
    return [coupling for coupling in COUPLINGS.values() if not coupling.separates]


def _assert_hits_lie_inside_solved_bands(
    hits: list[dict[str, object]], held: dict[str, object]
) -> None:
    """Every separating cell a value-sweep hit reports must sit inside a band for THAT change."""
    for hit in hits:
        change = hit["change"]
        assert isinstance(change, PolicyChange)
        geometry = held_baseline(coupling_for(hit["fields"]), held)
        for row in hit["rows"]:
            bands = separating_guard_bands(change, depth=row["depth"], held=geometry)
            assert any(band.contains(row["max_prompt_chars"]) for band in bands), (
                f"{hit['label']} @ {hit['candidate']}: {row['cell_id']}\n"
                f"{format_band_report(change, row['depth'], bands)}"
            )


def test_every_pair_of_auditable_fields_is_answered(design: dict[str, object]):
    """An unenumerated pair is an unasked question, so a NEW policy constant fails here."""
    pairs = {pair for pair in itertools.combinations(AUDITABLE_FIELDS, 2)}
    assert set(COUPLINGS) == pairs
    for pair, coupling in COUPLINGS.items():
        assert coupling.fields == pair
        assert coupling_for(pair) is coupling


def test_the_candidate_grid_extends_the_fixture_moves():
    """The sweep's first candidate is the FIELD_MOVES neighbour; further entries are extras."""
    assert set(FIELD_CANDIDATE_GRID) == set(FIELD_MOVES) == set(AUDITABLE_FIELDS)
    for field, (baseline, neighbour) in FIELD_MOVES.items():
        grid = FIELD_CANDIDATE_GRID[field]
        assert grid[0] == neighbour, field
        assert baseline not in grid, field
        assert len(grid) == len(set(grid)), field


def test_the_committed_fixture_is_the_one_pair_that_separates(design: dict[str, object]):
    """The compound guarantee rests on exactly the pair the fixture commits a geometry for."""
    separating = separating_couplings()
    assert [coupling.fields for coupling in separating] == [INTERACTING_FIELDS]
    assert separating[0].answer == ANSWER_BAND
    # The enumeration's example change IS the fixture's change, so the two cannot drift apart.
    assert separating[0].example_change() == interaction_change(design)
    # And that change is the first candidate combo the value sweep asks about.
    assert separating[0].example_change() == separating[0].candidate_changes()[0]


@pytest.mark.parametrize("coupling", _other_couplings(), ids=lambda coupling: coupling.label)
def test_no_other_pair_leaves_a_guard_its_conditions_all_admit(
    coupling: Coupling, held: dict[str, object]
):
    """The solver's own answer: at every reachable fold step some condition admits no guard."""
    change = coupling.example_change()
    geometry = held_baseline(coupling, held)
    bands = separating_guard_bands(change, depth=FIXTURE_DEPTH, held=geometry, include_empty=True)
    report = format_band_report(change, FIXTURE_DEPTH, bands)
    assert bands, report
    for band in bands:
        assert band.is_empty, report
        assert band.blocked_by(), report


@pytest.mark.parametrize("coupling", _other_couplings(), ids=lambda coupling: coupling.label)
def test_no_other_pair_separates_where_the_one_that_does_separates(
    coupling: Coupling, held: dict[str, object]
):
    """And the loop agrees with the arithmetic: a replay of the same geometry disagrees nowhere."""
    change = coupling.example_change()
    rows = scan_separating_cells(
        change,
        depths=[FIXTURE_DEPTH],
        guards=FIXTURE_GUARDS,
        held=held_baseline(coupling, held),
    )
    assert rows == [], format_scan_report(change, rows)


def test_a_candidate_value_sweep_at_the_fixture_keeps_every_other_pair_silent(
    held: dict[str, object],
):
    """Negative answers hold across the candidate grid, not only the FIELD_MOVES neighbour.

    The share direction is the instructive case: 0.5 -> 0.48 opens the band at the fixture guards,
    and 0.5 -> 0.55 opens none, so a sweep that only asked 0.48 would never see that the separation
    is directional. The other fourteen pairs must stay silent at every candidate combo.
    """
    result = scan_couplings_across_candidate_values(
        list(COUPLINGS.values()),
        depths=[FIXTURE_DEPTH],
        guards=FIXTURE_GUARDS,
        held=held,
    )
    report = format_value_sweep_report(result)
    silent = set(result["silent_pairs"])
    assert silent == {coupling.fields for coupling in _other_couplings()}, report
    hits = result["hits"]
    assert hits, report
    assert {hit["fields"] for hit in hits} == {INTERACTING_FIELDS}, report
    by_share = {hit["candidate"][FIELD_SHARE]: hit for hit in hits}
    assert 0.48 in by_share and 0.55 not in by_share, report
    _assert_hits_lie_inside_solved_bands(hits, held)


def test_the_scan_refuses_a_baseline_the_per_field_arm_would_not_replay(held: dict[str, object]):
    """A union read at a policy the change does not name is not a reading of the change at all."""
    change = PolicyChange(
        baseline={FIELD_CAP: 400, FIELD_SHARE: 0.5}, candidate={FIELD_CAP: 1600, FIELD_SHARE: 0.48}
    )
    with pytest.raises(ValueError, match="restate held_fixed at the baseline"):
        check_baseline_is_the_replayed_one(change, held)


def test_the_keep_conditions_are_two_halves_of_one_fold(held: dict[str, object]):
    """`compact_keep_recent` is silent exactly when it contributes nothing -- at EVERY fold step."""
    coupling = coupling_for((FIELD_SHARE, FIELD_KEEP_RECENT))
    bands = separating_guard_bands(
        coupling.example_change(),
        depth=FIXTURE_DEPTH,
        held=held_baseline(coupling, held),
        include_empty=True,
    )
    blocked = {
        band.fold_step: {condition.name for condition in band.blocked_by()} for band in bands
    }
    silent = "the_keep_audited_alone_is_silent"
    folds_a_span = "the_moved_keep_folds_a_different_span"
    for step, names in blocked.items():
        # One or the other blocks, never neither: at most one live entry means the two keeps fold
        # the same whole transcript, and more than one means the keep alone already changed a call.
        assert (silent in names) != (folds_a_span in names), (step, names)
        assert (folds_a_span in names) is (live_entries_at_fold_step(step) <= 1), (step, names)


def test_live_entries_at_a_fold_step_is_what_the_loop_actually_folds(held: dict[str, object]):
    """The keep conditions rest on this count, so it is measured against a replay, not assumed."""
    task = AgenticTask.from_record(
        build_memory_dependent_tasks(n_tasks=1, depth=FIXTURE_DEPTH, pad_chars=held["pad_chars"])[0]
    )
    for keep in (1, 2):
        policy = ContextPolicy(
            name=POLICY_COMPACT,
            observation_cap_chars=held["observation_cap_chars"],
            observation_head_share=held["observation_head_share"],
            compact_share=held[FIELD_SHARE],
            compact_keep_recent=keep,
        )
        prompts = replay_episode(
            policy,
            task=task,
            max_prompt_chars=FIXTURE_GUARDS[0],
            max_steps=FIXTURE_DEPTH + held["max_steps_margin"],
        ).prompts
        # The summarize call is made while the prompt for its fold step is being built, so its own
        # 1-based position among the model calls IS that step: the calls before it are the steps
        # before it. What it folded is then announced in the prompt that follows.
        fold_step = next(
            step for step, prompt in enumerate(prompts, start=1) if is_summary_prompt(prompt)
        )
        dropped = live_entries_at_fold_step(fold_step) - keep
        assert f"попередніх кроків ({dropped})" in prompts[fold_step], (fold_step, keep)


def test_keep_last_n_moves_no_prompt_in_either_audited_arm(held: dict[str, object]):
    """The `independent` answer for every `keep_last_n` pair, replayed rather than asserted."""
    cell = {
        "cell_id": "keep-last-inert",
        "depth": FIXTURE_DEPTH,
        "compact_share": held[FIELD_SHARE],
        "max_prompt_chars": FIXTURE_GUARDS[0],
        "pinned_fields": [],
    }
    row = audit_cell_prompts(cell, held, PolicyChange.of(FIELD_KEEP_LAST, 3, 1))
    assert row["changed_arms"] == []
    assert set(row["arms"]) == {POLICY_OBSERVATION_CAP, POLICY_COMPACT}


def test_the_head_share_moves_bytes_and_never_a_prompt_length(held: dict[str, object]):
    """The `observation_head_share` answer rests on this, so it is measured on real prompts."""
    task = AgenticTask.from_record(
        build_memory_dependent_tasks(n_tasks=1, depth=FIXTURE_DEPTH, pad_chars=held["pad_chars"])[0]
    )
    sides = [
        replay_episode(
            ContextPolicy(
                name=POLICY_OBSERVATION_CAP,
                observation_cap_chars=held["observation_cap_chars"],
                observation_head_share=head,
            ),
            task=task,
            max_prompt_chars=FIXTURE_GUARDS[0],
            max_steps=FIXTURE_DEPTH + held["max_steps_margin"],
        ).prompts
        for head in (held[FIELD_HEAD], 0.5)
    ]
    assert [len(prompt) for prompt in sides[0]] == [len(prompt) for prompt in sides[1]]
    assert prompt_sequence_digest(sides[0]) != prompt_sequence_digest(sides[1])


@pytest.mark.slow
def test_a_wide_value_sweep_finds_a_separating_geometry_only_where_the_band_says(
    held: dict[str, object],
):
    """The recorded proof: depths, guards, AND candidate values -- only one pair disagrees.

    For every other pair there must be no hit at any candidate combo. For the separating pair every
    hit must sit inside a solved band for the change that produced it, and the share direction that
    cannot elide (0.55) must contribute no hit of its own.
    """
    result = scan_couplings_across_candidate_values(
        list(COUPLINGS.values()),
        depths=SCAN_DEPTHS,
        guards=SCAN_GUARDS,
        held=held,
    )
    report = format_value_sweep_report(result)
    assert set(result["silent_pairs"]) == {coupling.fields for coupling in _other_couplings()}, (
        report
    )
    hits = result["hits"]
    assert hits, report
    assert {hit["fields"] for hit in hits} == {INTERACTING_FIELDS}, report
    assert all(hit["candidate"][FIELD_SHARE] == 0.48 for hit in hits), report
    assert all(hit["candidate"][FIELD_BOUND] == "trigger" for hit in hits), report
    _assert_hits_lie_inside_solved_bands(hits, held)
