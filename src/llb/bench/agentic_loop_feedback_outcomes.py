"""Task-family response-versus-completion summaries for repeat feedback."""

from typing import cast

from llb.core.contracts.benchmarks import AgenticCaseRow


def summarize_response_completion(rows: list[AgenticCaseRow]) -> dict[str, object]:
    """Separate controller responses from successful completions after those responses."""
    by_family: dict[str, dict[str, object]] = {}
    for family in sorted({cast(str, row.get("task_family", "")) for row in rows}):
        family_rows = [row for row in rows if row.get("task_family", "") == family]
        by_family[family] = _summary(family_rows)
    return {**_summary(rows), "by_family": by_family}


def compact_family_outcomes(
    by_family: dict[str, dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    """Keep the per-family response/completion fields needed by aggregate evidence."""
    return {
        family: {
            "response_rate": float(cast(float, row["response_rate"])),
            "redirected_completion_rate": float(
                cast(float, row.get("redirected_completion_rate", 0.0))
            ),
            "response_completion_rate": float(
                cast(float, row.get("response_completion_rate", 0.0))
            ),
            "redirected_tasks": int(cast(int, row.get("redirected_tasks", 0))),
            "completed_redirected_tasks": int(cast(int, row.get("completed_redirected_tasks", 0))),
        }
        for family, row in sorted(by_family.items())
    }


def _summary(rows: list[AgenticCaseRow]) -> dict[str, object]:
    activated = [row for row in rows if int(cast(int, row.get("n_repeated_noops", 0))) > 0]
    redirected = [row for row in activated if bool(row.get("repeat_feedback_redirected"))]
    completed = [row for row in redirected if float(row["success"]) > 0.0]
    return {
        "tasks": len(rows),
        "activated_tasks": len(activated),
        "redirected_tasks": len(redirected),
        "completed_redirected_tasks": len(completed),
        "response_rate": len(redirected) / len(activated) if activated else 0.0,
        "redirected_completion_rate": len(completed) / len(activated) if activated else 0.0,
        "response_completion_rate": len(completed) / len(redirected) if redirected else 0.0,
    }
