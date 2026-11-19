"""A spec and plan pair that passes `make lint-spec-plan`, for tests to break one way at a time."""

from pathlib import Path

SPEC = """# Design

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


def task_block(
    identifier: str = "widen-the-review-slice",
    *,
    capability: str = "gold-data",
    status: str = "CLEAR",
    optional: bool = False,
    requires: str | None = None,
) -> str:
    """One task carrying every required field, optionally `(optional)` or with a `Requires`."""
    suffix = " (optional)" if optional else ""
    requires_line = f"- Requires: {requires}\n" if requires is not None else ""
    return f"""#### {identifier}{suffix}

Widen the slice a reviewer sees before the gate reads it.

- Serves: `{capability}` -- [Spec](../design/spec.md#capability-registry)
- Agent status: {status}
{requires_line}- Dependencies: none.
- User-visible outcome: a reviewer sees the wider slice.
- Scope boundary: the fixture slice only; the review UI is out of scope.
- Data and artifact paths: `tests/fixtures/` only.
- Execution path: a module and its deterministic tests.
- Acceptance gates: `make ci` green, or the negative result recorded.
- Documentation target: [Data prep](current/data-prep.md).
"""


def group(capability: str, *blocks: str) -> str:
    """A `###` capability group holding the given task blocks."""
    return f"### {capability.replace('-', ' ').title()} -- `{capability}`\n\n" + "\n".join(blocks)


def plan_with(agent: tuple[str, ...] = (), human: tuple[str, ...] = ()) -> str:
    """A plan with both lanes, each holding the given rendered groups."""
    return (
        "# Plan\n\n## Agent Implementation Tasks\n\n"
        + "\n".join(agent)
        + "\n## Human-Assisted Tasks\n\n"
        + "\n".join(human)
    )


PLAN = plan_with(
    agent=(
        group("gold-data", task_block()),
        group(
            "retrieval-evidence",
            task_block("reranker-bake-off", capability="retrieval-evidence", optional=True),
        ),
    ),
)


def write_project(root: Path, *, spec: str = SPEC, plan: str = PLAN) -> Path:
    """Write the pair where the gate looks for it and return the project root."""
    (root / "docs/design").mkdir(parents=True)
    (root / "docs/impl").mkdir(parents=True)
    (root / "docs/design/spec.md").write_text(spec, encoding="utf-8")
    (root / "docs/impl/plan.md").write_text(plan, encoding="utf-8")
    return root
