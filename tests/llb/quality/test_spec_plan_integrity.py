"""The spec and the plan are two accounts of one product, and nothing else notices when they part.

The shipped-tree assertion is the point of the tool. The synthetic cases pin each way the two
documents can disagree, because every one of them has a failure mode that reads as "fine" to a
human skimming either file alone: a task under a capability nobody registered, a capability that
shipped without docs, a `planned` row nobody is building, and a group order that quietly turns
"what is next" back into an argument.
"""

import pytest

from llb.core.paths import PROJECT_ROOT
from llb.quality.spec_plan_integrity import (
    integrity_findings,
    read_registry,
    read_tasks,
)

_REGISTRY = """# Design

## Capability Registry

| # | Capability | Status | How it is evaluated | Implementation |
| --- | --- | --- | --- | --- |
| 1 | `gold-data` | shipped | Split validation on the fixture | [Data prep](../impl/current/data-prep.md) |
| 2 | `retrieval-evidence` | shipped | Recall at k against source spans | [RAG core](../impl/current/rag-core.md) |

## Something Else

| # | Capability | Status | How it is evaluated | Implementation |
| --- | --- | --- | --- | --- |
| 1 | `not-a-capability` | shipped | table outside the registry | [x](y.md) |
"""

_PLAN = """# Plan

## Agent Implementation Tasks

### Gold data -- `gold-data`

#### widen-the-review-slice

- Serves: `gold-data` -- [Gold data](../design/spec.md#data-and-ground-truth)
- Dependencies: none.

### Retrieval evidence -- `retrieval-evidence`

#### reranker-bake-off (optional)

- Serves: `retrieval-evidence` -- [Retrieval](../design/spec.md#retrieval-before-generation)
- Dependencies: none.
"""


def _docs(tmp_path, spec: str = _REGISTRY, plan: str = _PLAN):
    (tmp_path / "docs/design").mkdir(parents=True)
    (tmp_path / "docs/impl").mkdir(parents=True)
    (tmp_path / "docs/design/spec.md").write_text(spec, encoding="utf-8")
    (tmp_path / "docs/impl/plan.md").write_text(plan, encoding="utf-8")
    return tmp_path


def test_the_shipped_spec_and_plan_agree():
    """The CI-able assertion: every task serves a registered capability, in line order."""
    assert integrity_findings(PROJECT_ROOT) == []


def test_only_the_registry_table_is_read_as_capabilities(tmp_path):
    """A later table with the same shape is not a second registry."""
    registry = read_registry(_docs(tmp_path) / "docs/design/spec.md")

    assert [capability.id for capability in registry] == ["gold-data", "retrieval-evidence"]


def test_a_task_is_read_with_its_group_and_its_declared_capability(tmp_path):
    """`(optional)` is part of the heading, not part of the id."""
    tasks = read_tasks(_docs(tmp_path) / "docs/impl/plan.md")

    assert [(task.id, task.group, task.serves) for task in tasks] == [
        ("widen-the-review-slice", "gold-data", "gold-data"),
        ("reranker-bake-off", "retrieval-evidence", "retrieval-evidence"),
    ]


def test_a_healthy_pair_reports_nothing(tmp_path):
    assert integrity_findings(_docs(tmp_path)) == []


def test_a_task_serving_an_unregistered_capability_is_named(tmp_path):
    """The drift the whole check exists for: capability arriving with no spec amendment."""
    plan = _PLAN.replace("`gold-data` -- [Gold data]", "`corpus-telemetry` -- [Gold data]")
    findings = integrity_findings(_docs(tmp_path, plan=plan))

    assert any("not a registered capability" in finding for finding in findings)
    assert any("widen-the-review-slice" in finding for finding in findings)


def test_a_task_with_no_serves_line_is_named(tmp_path):
    plan = _PLAN.replace(
        "- Serves: `gold-data` -- [Gold data](../design/spec.md#data-and-ground-truth)\n", ""
    )
    findings = integrity_findings(_docs(tmp_path, plan=plan))

    assert any("no `Serves` line" in finding for finding in findings)


def test_a_task_filed_under_another_capabilitys_group_is_named(tmp_path):
    """A `Serves` line that disagrees with the group it sits in breaks the ordering rule."""
    plan = _PLAN.replace("- Serves: `retrieval-evidence` --", "- Serves: `gold-data` --")
    findings = integrity_findings(_docs(tmp_path, plan=plan))

    assert any("under the `retrieval-evidence` group" in finding for finding in findings)


def test_a_planned_capability_nobody_is_building_is_named(tmp_path):
    """A `planned` row with no task is a capability that exists only on paper."""
    spec = _REGISTRY.replace(
        "\n## Something Else",
        "| 3 | `table-aware-chunking` | planned | Paired recall on a table corpus | -- |\n"
        "\n## Something Else",
    )
    findings = integrity_findings(_docs(tmp_path, spec=spec))

    assert findings == [
        "docs/design/spec.md: `table-aware-chunking`: planned but no plan task serves it"
    ]


def test_a_shipped_capability_with_no_implementation_link_is_named(tmp_path):
    spec = _REGISTRY.replace("[Data prep](../impl/current/data-prep.md)", "not written up yet")
    findings = integrity_findings(_docs(tmp_path, spec=spec))

    assert any("shipped but links to no implementation docs" in finding for finding in findings)


@pytest.mark.parametrize("evaluation", ["", "   "])
def test_a_capability_with_no_evaluation_is_named(tmp_path, evaluation):
    """Step 3 of the lifecycle, enforced: a capability declares how it is known to work."""
    spec = _REGISTRY.replace("| Split validation on the fixture |", f"| {evaluation} |")
    findings = integrity_findings(_docs(tmp_path, spec=spec))

    assert any("no evaluation declared" in finding for finding in findings)


def test_a_capability_listed_twice_is_named(tmp_path):
    spec = _REGISTRY.replace(
        "| 2 | `retrieval-evidence` | shipped | Recall at k against source spans |",
        "| 2 | `gold-data` | shipped | Recall at k against source spans |",
    )
    findings = integrity_findings(_docs(tmp_path, spec=spec))

    assert any("listed twice" in finding for finding in findings)


def test_an_unknown_status_is_named(tmp_path):
    spec = _REGISTRY.replace("| `gold-data` | shipped |", "| `gold-data` | in-progress |")
    findings = integrity_findings(_docs(tmp_path, spec=spec))

    assert any("is not one of" in finding for finding in findings)


def test_groups_out_of_implementation_line_order_are_named(tmp_path):
    """Swapping two groups is exactly how the plan stops answering "what is next" by itself."""
    lines = _PLAN.splitlines(keepends=True)
    head = lines[: lines.index("### Gold data -- `gold-data`\n")]
    first = lines[
        lines.index("### Gold data -- `gold-data`\n") : lines.index(
            "### Retrieval evidence -- `retrieval-evidence`\n"
        )
    ]
    second = lines[lines.index("### Retrieval evidence -- `retrieval-evidence`\n") :]
    findings = integrity_findings(_docs(tmp_path, plan="".join(head + second + first)))

    assert any("out of implementation-line order" in finding for finding in findings)


def test_a_spec_with_no_registry_is_named(tmp_path):
    findings = integrity_findings(_docs(tmp_path, spec="# Design\n\nNo registry here.\n"))

    assert any("no capability registry rows" in finding for finding in findings)
