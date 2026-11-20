"""The spec and the plan are two accounts of one product, and nothing else notices when they part.

The shipped-tree assertion is the point of the tool. The synthetic cases pin each way the two
documents can disagree, and each way a task can stop being executable without guessing, because
every one of them has a failure mode that reads as "fine" to a human skimming either file alone: a
task under a capability nobody registered, a capability that shipped without docs, a `planned` row
nobody is building, a group order that quietly turns "what is next" back into an argument, a task
missing the field its implementer needs, and a prerequisite that names a task nobody can find.
"""

import pytest

from llb.core.paths import PROJECT_ROOT
from llb.quality.spec_plan_integrity import integrity_findings, main
from tests.llb.quality.plan_fixtures import PLAN, SPEC, group, plan_with, task_block, write_project


def _findings(tmp_path, *, spec: str = SPEC, plan: str = PLAN) -> list[str]:
    return integrity_findings(write_project(tmp_path, spec=spec, plan=plan))


def test_the_shipped_spec_and_plan_agree():
    """The CI-able assertion: every task serves a registered capability, in line order."""
    assert integrity_findings(PROJECT_ROOT) == []


def test_a_healthy_pair_reports_nothing(tmp_path):
    assert _findings(tmp_path) == []


def test_missing_documents_are_named(tmp_path):
    assert integrity_findings(tmp_path) == [
        "missing required document: docs/design/spec.md",
        "missing required document: docs/impl/plan.md",
    ]


def test_a_spec_with_no_registry_is_named(tmp_path):
    findings = _findings(tmp_path, spec="# Design\n\nNo registry here.\n")

    assert any("no capability registry rows" in finding for finding in findings)


# -- registry invariants ----------------------------------------------------------------------


def test_a_task_serving_an_unregistered_capability_is_named(tmp_path):
    """The drift the whole check exists for: capability arriving with no spec amendment."""
    plan = PLAN.replace("- Serves: `gold-data`", "- Serves: `corpus-telemetry`")
    findings = _findings(tmp_path, plan=plan)

    assert any(
        "widen-the-review-slice" in finding and "not a registered capability" in finding
        for finding in findings
    )


def test_a_task_with_no_serves_line_is_named(tmp_path):
    plan = PLAN.replace(
        "- Serves: `gold-data` -- [Spec](../design/spec.md#capability-registry)\n", ""
    )

    assert any("no `Serves` line" in finding for finding in _findings(tmp_path, plan=plan))


def test_a_task_filed_under_another_capabilitys_group_is_named(tmp_path):
    """A `Serves` line that disagrees with the group it sits in breaks the ordering rule."""
    plan = PLAN.replace("- Serves: `retrieval-evidence`", "- Serves: `gold-data`")

    assert any(
        "under the `retrieval-evidence` group" in finding
        for finding in _findings(tmp_path, plan=plan)
    )


def test_a_planned_capability_nobody_is_building_is_named(tmp_path):
    """A `planned` row with no task is a capability that exists only on paper."""
    spec = SPEC.replace(
        "\n## Something Else",
        "| 3 | `table-aware-chunking` | planned | Paired recall on a table corpus | -- |\n"
        "\n## Something Else",
    )

    assert _findings(tmp_path, spec=spec) == [
        "docs/design/spec.md: `table-aware-chunking`: planned but no plan task serves it"
    ]


def test_a_shipped_capability_with_no_implementation_link_is_named(tmp_path):
    spec = SPEC.replace("[Data prep](../impl/current/data-prep.md)", "not written up yet")

    assert any(
        "shipped but links to no implementation docs" in finding
        for finding in _findings(tmp_path, spec=spec)
    )


@pytest.mark.parametrize("evaluation", ["", "   "])
def test_a_capability_with_no_evaluation_is_named(tmp_path, evaluation):
    """Step 3 of the lifecycle, enforced: a capability declares how it is known to work."""
    spec = SPEC.replace("| Split validation on the fixture |", f"| {evaluation} |")

    assert any("no evaluation declared" in finding for finding in _findings(tmp_path, spec=spec))


def test_a_capability_listed_twice_is_named(tmp_path):
    spec = SPEC.replace("| 2 | `retrieval-evidence` | shipped |", "| 2 | `gold-data` | shipped |")

    assert any("listed twice" in finding for finding in _findings(tmp_path, spec=spec))


def test_an_unknown_status_is_named(tmp_path):
    spec = SPEC.replace("| `gold-data` | shipped |", "| `gold-data` | in-progress |")

    assert any("is not one of" in finding for finding in _findings(tmp_path, spec=spec))


# -- ordering ---------------------------------------------------------------------------------


def test_groups_out_of_implementation_line_order_are_named(tmp_path):
    """Swapping two groups is exactly how the plan stops answering "what is next" by itself."""
    plan = plan_with(
        agent=(
            group("retrieval-evidence", task_block("rerank", capability="retrieval-evidence")),
            group("gold-data", task_block()),
        )
    )

    assert any(
        "out of implementation-line order" in finding for finding in _findings(tmp_path, plan=plan)
    )


def test_returning_to_an_earlier_group_is_named(tmp_path):
    """A lane that revisits a capability has two positions for it, so neither is THE position."""
    plan = plan_with(
        agent=(
            group("gold-data", task_block()),
            group("retrieval-evidence", task_block("rerank", capability="retrieval-evidence")),
            group("gold-data", task_block("more-gold")),
        )
    )

    assert any(
        "out of implementation-line order" in finding for finding in _findings(tmp_path, plan=plan)
    )


def test_required_work_after_optional_work_in_its_group_is_named(tmp_path):
    """`(optional)` is binding inside its group: nothing required may sort behind it."""
    plan = plan_with(
        agent=(group("gold-data", task_block("nice-to-have", optional=True), task_block()),)
    )

    assert any("follows optional work" in finding for finding in _findings(tmp_path, plan=plan))


def test_an_optional_task_of_an_earlier_capability_may_precede_required_work(tmp_path):
    """`(optional)` never reorders the line across capabilities."""
    plan = plan_with(
        agent=(
            group("gold-data", task_block("nice-to-have", optional=True)),
            group("retrieval-evidence", task_block("rerank", capability="retrieval-evidence")),
        )
    )

    assert _findings(tmp_path, plan=plan) == []


# -- task shape -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("- Agent status: CLEAR\n", "Agent status"),
        ("- Dependencies: none.\n", "Dependencies"),
        ("- Execution path: a module and its deterministic tests.\n", "Execution path"),
        ("- Documentation target: [Data prep](current/data-prep.md).\n", "Documentation target"),
    ],
)
def test_a_task_missing_a_required_field_is_named(tmp_path, field, label):
    plan = plan_with(agent=(group("gold-data", task_block().replace(field, "")),))

    assert any(
        f"missing non-empty `{label}` field" in finding
        for finding in _findings(tmp_path, plan=plan)
    )


def test_a_field_whose_value_starts_on_the_next_line_is_not_empty(tmp_path):
    """The plan wraps at 100 columns, so a value may begin on its continuation line."""
    block = task_block().replace(
        "- Documentation target: [Data prep](current/data-prep.md).\n",
        "- Documentation target:\n  [Data prep](current/data-prep.md).\n",
    )

    assert _findings(tmp_path, plan=plan_with(agent=(group("gold-data", block),))) == []


@pytest.mark.parametrize(
    ("status", "agent_lane"),
    [("HUMAN-GATED", True), ("BLOCKED BY HUMAN", True), ("CLEAR", False), ("RUN NEEDED", False)],
)
def test_a_status_the_lane_does_not_allow_is_named(tmp_path, status, agent_lane):
    """The lane says who can finish the task; the status must agree with it."""
    lane = (group("gold-data", task_block(status=status)),)
    plan = plan_with(agent=lane) if agent_lane else plan_with(human=lane)

    assert any("is invalid in" in finding for finding in _findings(tmp_path, plan=plan))


def test_a_status_may_carry_its_reason(tmp_path):
    block = task_block(status="RUN NEEDED -- deterministic at fixed seeds on the CUDA host")

    assert _findings(tmp_path, plan=plan_with(agent=(group("gold-data", block),))) == []


def test_research_is_a_field_not_a_status(tmp_path):
    """`Research: yes` sits beside the status; folding it into the status line is refused."""
    refused = plan_with(agent=(group("gold-data", task_block(status="RUN NEEDED, RESEARCH")),))
    accepted = plan_with(
        agent=(
            group(
                "gold-data",
                task_block(status="RUN NEEDED").replace(
                    "- Dependencies:", "- Research: yes\n- Dependencies:"
                ),
            ),
        )
    )

    assert any("is invalid in" in finding for finding in _findings(tmp_path / "a", plan=refused))
    assert _findings(tmp_path / "b", plan=accepted) == []


def test_a_duplicated_task_id_is_named(tmp_path):
    plan = plan_with(agent=(group("gold-data", task_block(), task_block()),))

    assert any("task id is duplicated" in finding for finding in _findings(tmp_path, plan=plan))


def test_a_malformed_task_heading_is_named(tmp_path):
    plan = PLAN.replace("#### widen-the-review-slice", "#### Widen the review slice")

    assert any("malformed task heading" in finding for finding in _findings(tmp_path, plan=plan))


def test_a_task_outside_both_lanes_is_named(tmp_path):
    plan = PLAN + "\n## Adding Future Tasks\n\n" + group("gold-data", task_block("stray-task"))

    assert any(
        "`stray-task`: task sits outside both task lanes" in finding
        for finding in _findings(tmp_path, plan=plan)
    )


def test_a_missing_lane_is_named(tmp_path):
    plan = PLAN.replace("## Human-Assisted Tasks\n", "")

    assert any("missing required lane" in finding for finding in _findings(tmp_path, plan=plan))


@pytest.mark.parametrize(
    "history", ["Done once.", "Delivered.", "Implemented it.", "As of 2026-01-01."]
)
def test_history_in_the_forward_plan_is_named(tmp_path, history):
    plan = PLAN.replace("Widen the slice a reviewer sees before the gate reads it.", history, 1)

    assert any("history or a date" in finding for finding in _findings(tmp_path, plan=plan))


# -- Requires ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requires", "expected"),
    [
        ("`missing-task`", "requires unknown open task `missing-task`"),
        ("reranker-bake-off", "comma-separated backticked task IDs"),
        ("`widen-the-review-slice`", "task dependency cycle"),
        ("`reranker-bake-off`", "required work depends on optional task `reranker-bake-off`"),
    ],
)
def test_a_bad_requires_edge_is_named(tmp_path, requires, expected):
    plan = PLAN.replace(task_block(), task_block(requires=requires))

    assert any(expected in finding for finding in _findings(tmp_path, plan=plan))


def test_requires_may_cross_lanes(tmp_path):
    """An agent task may wait on a human-gated one; the edge is valid, only the start waits."""
    plan = plan_with(
        agent=(group("gold-data", task_block(requires="`review-the-slice`")),),
        human=(group("gold-data", task_block("review-the-slice", status="HUMAN-GATED")),),
    )

    assert _findings(tmp_path, plan=plan) == []


def test_main_exits_nonzero_on_a_disagreement_and_zero_on_agreement(tmp_path):
    bad = write_project(tmp_path / "bad", plan=PLAN.replace("## Human-Assisted Tasks\n", ""))
    good = write_project(tmp_path / "good")

    assert main(["--root", str(bad)]) == 1
    assert main(["--root", str(good)]) == 0
