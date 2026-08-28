"""The stopping rule: rank agreement, block snapshots, and what actually ends the search."""

from pathlib import Path

from llb.optimize.joint_search.long_run.plan import declare_plan
from llb.optimize.joint_search.long_run.sequential import (
    STOPPED_BY_BUDGET,
    STOPPED_BY_STABILITY,
    run_trial_blocks,
)
from llb.optimize.joint_search.long_run.stability import (
    build_snapshot,
    rank_agreement,
    ranking_from,
)

DELTAS = [0.3, -0.2, 0.4, -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1]


def _plan(tmp_path: Path, *, budget: int, block: int, blocks: int, agreement: float = 1.0):
    return declare_plan(
        tmp_path / "reference.json",
        DELTAS,
        minimum_detectable_gain=0.05,
        available_n=82,
        trial_budget=budget,
        trial_block=block,
        stability_blocks=blocks,
        stability_agreement=agreement,
        selector={"lane": "test", "candidate": "a", "baseline": "b", "metric": "q"},
    )


def test_ranking_breaks_ties_by_name_so_equal_objectives_never_flap():
    assert ranking_from({"bravo": 0.5, "alpha": 0.5, "charlie": 0.9}) == (
        "charlie",
        "alpha",
        "bravo",
    )


def test_rank_agreement_is_one_for_an_identical_order_and_zero_for_a_reversal():
    assert rank_agreement(("a", "b", "c"), ("a", "b", "c")) == 1.0
    assert rank_agreement(("a", "b", "c"), ("c", "b", "a")) == 0.0
    # One swapped adjacent pair out of three ordered pairs.
    assert rank_agreement(("a", "b", "c"), ("b", "a", "c")) == 2 / 3


def test_the_first_block_can_never_be_stable():
    """A rule about how the order MOVED needs two blocks; one block has no transition."""
    first = build_snapshot(
        index=0,
        trials_per_finalist=5,
        consumed_trials=10,
        objective={"alpha": 0.4, "bravo": 0.6},
        previous=None,
        agreement_floor=1.0,
    )
    assert first.agreement is None
    assert first.leader_held is None
    assert not first.stable
    assert first.stable_streak == 0


def test_a_reshuffled_leader_resets_the_streak():
    first = build_snapshot(
        index=0,
        trials_per_finalist=5,
        consumed_trials=10,
        objective={"alpha": 0.4, "bravo": 0.6},
        previous=None,
        agreement_floor=1.0,
    )
    held = build_snapshot(
        index=1,
        trials_per_finalist=10,
        consumed_trials=20,
        objective={"alpha": 0.45, "bravo": 0.62},
        previous=first,
        agreement_floor=1.0,
    )
    assert held.stable and held.stable_streak == 1
    flipped = build_snapshot(
        index=2,
        trials_per_finalist=15,
        consumed_trials=30,
        objective={"alpha": 0.9, "bravo": 0.62},
        previous=held,
        agreement_floor=1.0,
    )
    assert not flipped.stable
    assert flipped.leader_held is False
    assert flipped.stable_streak == 0


def test_a_settled_ranking_stops_the_search_before_the_budget(tmp_path: Path):
    """Two consecutive unchanged transitions end it at block 2, well short of 30 trials."""
    plan = _plan(tmp_path, budget=30, block=5, blocks=2)
    quality = {"alpha": 0.40, "bravo": 0.60}
    calls: list[tuple[str, int]] = []

    def advance(name: str, target: int) -> tuple[float, int]:
        calls.append((name, target))
        return quality[name], target

    trail = run_trial_blocks(["alpha", "bravo"], plan=plan, advance=advance)
    assert trail.stopped_by == STOPPED_BY_STABILITY
    assert trail.trials_per_finalist == 15
    assert trail.consumed_total == 30
    assert [block.stable_streak for block in trail.blocks] == [0, 1, 2]
    assert calls[-1] == ("bravo", 15)


def test_a_ranking_that_never_settles_spends_the_declared_budget_and_says_so(tmp_path: Path):
    """The budget is the other way out, and the trail records which one fired."""
    plan = _plan(tmp_path, budget=20, block=5, blocks=2)
    flips = iter([{"alpha": 0.4}, {"alpha": 0.7}, {"alpha": 0.4}, {"alpha": 0.7}])
    current = {"alpha": 0.0, "bravo": 0.6}

    def advance(name: str, target: int) -> tuple[float, int]:
        if name == "alpha":
            current.update(next(flips))
        return current[name], target

    trail = run_trial_blocks(["alpha", "bravo"], plan=plan, advance=advance)
    assert trail.stopped_by == STOPPED_BY_BUDGET
    assert trail.trials_per_finalist == 20
    assert trail.to_dict()["budget_exhausted"] is True
    assert max(block.stable_streak for block in trail.blocks) < plan.stability_blocks


def test_a_final_partial_block_never_exceeds_the_budget(tmp_path: Path):
    """A budget that is not a whole number of blocks stops exactly ON the budget."""
    plan = _plan(tmp_path, budget=12, block=5, blocks=99)
    trail = run_trial_blocks(["alpha"], plan=plan, advance=lambda name, target: (0.5, target))
    assert [block.trials_per_finalist for block in trail.blocks] == [5, 10, 12]
    assert trail.trials_per_finalist == 12


def test_a_looser_agreement_tolerates_tail_churn_but_not_a_leader_swap(tmp_path: Path):
    """At agreement 0.6 a reshuffle below the leader still counts as stable; a swap does not."""
    plan = _plan(tmp_path, budget=30, block=5, blocks=1, agreement=0.6)
    orders = iter(
        [
            {"alpha": 0.9, "bravo": 0.5, "charlie": 0.4},
            {"alpha": 0.9, "bravo": 0.3, "charlie": 0.4},
        ]
    )
    current: dict[str, float] = {}

    def advance(name: str, target: int) -> tuple[float, int]:
        if not current or name == "alpha":
            current.update(next(orders))
        return current[name], target

    trail = run_trial_blocks(["alpha", "bravo", "charlie"], plan=plan, advance=advance)
    assert trail.stopped_by == STOPPED_BY_STABILITY
    assert trail.blocks[-1].leader_held is True
    assert trail.blocks[-1].agreement == 2 / 3
