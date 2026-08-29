"""The balanced arm schedule: what executes when, and that only the ORDER changed.

Two-arm agentic studies on this host drive one stateful serving endpoint, so running whole arm
blocks makes "the second arm" and "the arm under test" the same column. These are the contracts
the interleaved runner replaces that with: both arms of a task run adjacently, the first position
alternates, no arm holds a position more than one task more than the other, and every episode is
still SCORED under its own arm in task order so the readings downstream are unchanged.
"""

import pytest

from llb.backends.context_budget import fixed_budget
from llb.bench.agentic.context_policy import POLICY_COMPACT, ContextPolicy
from llb.bench.agentic.model import AgenticTask
from llb.bench.context_policy.interleave import (
    alternating_arm_schedule,
    run_arms_interleaved,
)

ARMS = ("head_tail", "per_entry_head")


def _tasks(n: int) -> list[AgenticTask]:
    return [
        AgenticTask(
            id=f"t{index}",
            prompt=f"Задача {index}: знайди дані і повідом готово.",
            setup={"corpus": {"d1": "дані"}},
            success=[{"kind": "answer_contains", "value": "готово"}],
        )
        for index in range(n)
    ]


def _policies() -> dict[str, ContextPolicy]:
    return {
        arm: ContextPolicy(name=POLICY_COMPACT, summary_trim_strategy=arm, compact_share=0.8)
        for arm in ARMS
    }


def _finisher(prompt: str) -> str:
    return '{"name": "finish", "arguments": {"answer": "готово"}}'


def test_the_schedule_alternates_the_first_position_task_by_task():
    schedule = alternating_arm_schedule(ARMS, 4)
    assert schedule == [ARMS, ARMS[::-1], ARMS, ARMS[::-1]]
    assert all(set(order) == set(ARMS) for order in schedule)


def test_an_odd_task_count_leaves_at_most_one_extra_first_position():
    """Perfect balance is impossible on an odd set; one task of slack is the whole error."""
    firsts = [order[0] for order in alternating_arm_schedule(ARMS, 23)]
    assert abs(firsts.count(ARMS[0]) - firsts.count(ARMS[1])) == 1


def test_the_offset_carries_the_rotation_so_remainders_cancel():
    """A second workload continues the alternation rather than restarting on the same arm."""
    first = alternating_arm_schedule(ARMS, 3)
    second = alternating_arm_schedule(ARMS, 3, offset=3)
    firsts = [order[0] for order in first + second]
    assert firsts.count(ARMS[0]) == firsts.count(ARMS[1]) == 3


def test_a_duplicate_arm_name_is_refused():
    with pytest.raises(ValueError, match="unique"):
        alternating_arm_schedule(("a", "a"), 2)


def test_the_runner_executes_arms_task_adjacent_and_scores_them_in_task_order():
    tasks = _tasks(3)
    run = run_arms_interleaved(
        tasks,
        _policies(),
        backend="fake",
        complete=_finisher,
        max_steps=4,
        budget=fixed_budget(20000),
    )
    executed = [(row["task_index"], row["arm"]) for row in run.schedule]
    # Task-adjacent: a task's two episodes are consecutive, and the pairs alternate.
    assert executed == [
        (0, "head_tail"),
        (0, "per_entry_head"),
        (1, "per_entry_head"),
        (1, "head_tail"),
        (2, "head_tail"),
        (2, "per_entry_head"),
    ]
    # Scored per arm, in TASK order -- not in the order the endpoint saw them.
    for arm in ARMS:
        report = run.reports[arm]
        assert [row["item_id"] for row in report.rows] == [task.id for task in tasks]
        assert len(report.episodes) == len(tasks)


def test_every_arm_runs_every_task_exactly_once():
    """The interleaving is a re-ordering, not a re-sampling: the episode budget is unchanged."""
    tasks = _tasks(5)
    run = run_arms_interleaved(
        tasks,
        _policies(),
        backend="fake",
        complete=_finisher,
        max_steps=4,
        budget=fixed_budget(20000),
    )
    assert len(run.schedule) == len(tasks) * len(ARMS)
    for arm in ARMS:
        ran = [row["item_id"] for row in run.schedule if row["arm"] == arm]
        assert sorted(ran) == sorted(task.id for task in tasks)
    positions = [row["position"] for row in run.schedule if row["arm"] == ARMS[0]]
    assert sorted(positions) == [1, 1, 1, 2, 2]
