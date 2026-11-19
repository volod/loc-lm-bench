"""The readers see exactly the shape the planning workflow prescribes, and nothing it does not.

Every case here is a way prose could be mistaken for structure, or structure for prose: a second
table shaped like the registry, a task-shape example quoted in a fence, a field wrapped onto its
next line, and a bullet list that follows the last task under some other `##` heading.
"""

from llb.quality.plan_model import read_registry, read_tasks
from tests.llb.quality.plan_fixtures import PLAN, group, plan_with, task_block, write_project


def _tasks(tmp_path, plan: str = PLAN):
    return read_tasks(write_project(tmp_path, plan=plan) / "docs/impl/plan.md")


def test_only_the_registry_table_is_read_as_capabilities(tmp_path):
    """A later table with the same shape is not a second registry."""
    registry = read_registry(write_project(tmp_path) / "docs/design/spec.md")

    assert [capability.identifier for capability in registry] == ["gold-data", "retrieval-evidence"]
    assert registry[0].status == "shipped"


def test_a_task_is_read_with_its_lane_group_and_declared_capability(tmp_path):
    """`(optional)` is part of the heading, not part of the id."""
    tasks = _tasks(tmp_path)

    assert [(task.identifier, task.group, task.serves, task.optional) for task in tasks] == [
        ("widen-the-review-slice", "gold-data", "gold-data", False),
        ("reranker-bake-off", "retrieval-evidence", "retrieval-evidence", True),
    ]
    assert {task.section for task in tasks} == {"Agent Implementation Tasks"}


def test_a_wrapped_field_value_is_joined_across_its_continuation_lines(tmp_path):
    block = task_block().replace(
        "- Scope boundary: the fixture slice only; the review UI is out of scope.\n",
        "- Scope boundary: the fixture slice only;\n  the review UI is out of scope.\n",
    )
    (task,) = _tasks(tmp_path, plan_with(agent=(group("gold-data", block),)))

    assert task.fields["scope boundary"] == "the fixture slice only; the review UI is out of scope."


def test_the_status_marker_drops_its_reason(tmp_path):
    block = task_block(status="clear -- fixtures only,\n  no run")
    (task,) = _tasks(tmp_path, plan_with(agent=(group("gold-data", block),)))

    assert task.agent_status == "CLEAR"


def test_the_first_occurrence_of_a_field_wins(tmp_path):
    block = task_block().replace(
        "- Dependencies: none.\n", "- Dependencies: none.\n- Dependencies: x\n"
    )
    (task,) = _tasks(tmp_path, plan_with(agent=(group("gold-data", block),)))

    assert task.fields["dependencies"] == "none."


def test_a_fenced_task_example_is_not_read(tmp_path):
    plan = PLAN + "\n```markdown\n#### example-task\n\n- Serves: `gold-data`\n```\n"

    assert [task.identifier for task in _tasks(tmp_path, plan)] == [
        "widen-the-review-slice",
        "reranker-bake-off",
    ]


def test_a_malformed_task_heading_ends_the_task_before_it(tmp_path):
    """Its fields must not land on the previous task, where they would mask a missing field."""
    plan = plan_with(
        agent=(group("gold-data", task_block(), "#### Not A Task\n\n- Requires: `x`\n"),)
    )
    (task,) = _tasks(tmp_path, plan)

    assert "requires" not in task.fields


def test_bullets_under_a_later_section_do_not_attach_to_the_last_task(tmp_path):
    """A how-to-write-a-task list after the lanes describes fields; it does not fill them."""
    block = task_block().replace("- Dependencies: none.\n", "")
    plan = (
        plan_with(agent=(group("gold-data", block),))
        + "\n## Adding Future Tasks\n\n- Dependencies: prerequisite tasks by id.\n"
    )
    (task,) = _tasks(tmp_path, plan)

    assert "dependencies" not in task.fields
