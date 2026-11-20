"""`make plan-status` names the next task by the plan's own rule, or refuses to name one.

The rule is document order: the first task of the earliest group, skipping one still waiting on an
open `Requires`. An optional task of an earlier capability therefore precedes a required task of a
later one -- the line is the trust chain, and `(optional)` only sorts within its own group.
"""

import re

from llb.core.paths import PROJECT_ROOT
from llb.quality.plan_summary import INVALID_PLAN_MESSAGE, main, summary_lines
from tests.llb.quality.plan_fixtures import PLAN, group, plan_with, task_block, write_project


def _summary(tmp_path, plan: str = PLAN) -> list[str]:
    return summary_lines(write_project(tmp_path, plan=plan))


def test_the_shipped_plan_is_schedulable_and_fully_counted():
    """The summary agrees with the heading count AGENTS.md used to ask for by hand."""
    lines = summary_lines(PROJECT_ROOT)
    headings = re.findall(r"^#### ", (PROJECT_ROOT / "docs/impl/plan.md").read_text(), re.MULTILINE)

    assert lines != [INVALID_PLAN_MESSAGE]
    assert lines[0] == f"tasks: {len(headings)}"


def test_counts_lanes_statuses_and_waiting_tasks(tmp_path):
    plan = plan_with(
        agent=(group("gold-data", task_block(requires="`review-the-slice`"), task_block("other")),),
        human=(group("gold-data", task_block("review-the-slice", status="HUMAN-GATED")),),
    )

    assert _summary(tmp_path, plan) == [
        "tasks: 3",
        "agent lane: 2",
        "human lane: 1",
        "statuses: CLEAR=2, HUMAN-GATED=1",
        "waiting on open Requires: 1",
        "priority agent task: other [gold-data]",
        "priority human task: review-the-slice [gold-data]",
    ]


def test_an_optional_task_of_an_earlier_capability_comes_first(tmp_path):
    plan = plan_with(
        agent=(
            group("gold-data", task_block("nice-to-have", optional=True)),
            group("retrieval-evidence", task_block("rerank", capability="retrieval-evidence")),
        )
    )

    assert "priority agent task: nice-to-have (optional) [gold-data]" in _summary(tmp_path, plan)


def test_a_lane_whose_every_task_waits_says_so(tmp_path):
    plan = plan_with(
        agent=(group("gold-data", task_block(requires="`review-the-slice`")),),
        human=(group("gold-data", task_block("review-the-slice", status="HUMAN-GATED")),),
    )

    lines = _summary(tmp_path, plan)

    assert "priority agent task: all waiting on prerequisites" in lines


def test_an_empty_lane_reports_none(tmp_path):
    assert "priority human task: none" in _summary(tmp_path)


def test_an_invalid_plan_is_never_scheduled(tmp_path):
    root = write_project(tmp_path, plan=PLAN.replace("- Serves: `gold-data`", "- Serves: `nope`"))

    assert summary_lines(root) == [INVALID_PLAN_MESSAGE]
    assert main(["--root", str(root)]) == 1
