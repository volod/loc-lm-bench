"""Validate the `Requires` field: open-task prerequisites that must leave the plan first.

`Requires` is the machine-readable half of a task's prerequisites -- backticked ids of OPEN tasks,
in either lane, that must be accepted before this one can start. Everything else a task needs (a
current-docs fact, an external input, a human authorization) stays in the prose `Dependencies`
field, which a contributor verifies by reading. Keeping the two apart is what lets
`make plan-status` skip a waiting task without parsing prose.
"""

import re
from graphlib import CycleError, TopologicalSorter

from llb.quality.plan_model import Task

REQUIRES_FIELD = "requires"
_TASK_ID = re.compile(r"`([a-z0-9][a-z0-9-]*)`")
_REQUIRES_LIST = re.compile(rf"{_TASK_ID.pattern}(?:,\s*{_TASK_ID.pattern})*")


def prerequisites(task: Task) -> list[str]:
    """The open task ids this task declares it cannot start before."""
    return _TASK_ID.findall(task.fields.get(REQUIRES_FIELD, ""))


def _task_findings(task: Task, tasks_by_id: dict[str, Task]) -> list[str]:
    where = f"`{task.identifier}`"
    findings: list[str] = []
    declared = task.fields.get(REQUIRES_FIELD)
    if declared is not None and not _REQUIRES_LIST.fullmatch(declared):
        findings.append(f"{where}: Requires must list comma-separated backticked task IDs")
    for identifier in prerequisites(task):
        prerequisite = tasks_by_id.get(identifier)
        if prerequisite is None:
            findings.append(f"{where}: requires unknown open task `{identifier}`")
        elif prerequisite.optional and not task.optional:
            findings.append(f"{where}: required work depends on optional task `{identifier}`")
    return findings


def dependency_findings(tasks: list[Task]) -> list[str]:
    """Check prerequisite syntax, targets, and cycles across both lanes."""
    tasks_by_id = {task.identifier: task for task in tasks}
    findings = [finding for task in tasks for finding in _task_findings(task, tasks_by_id)]
    sorter = TopologicalSorter({task.identifier: prerequisites(task) for task in tasks})
    try:
        sorter.prepare()
    except CycleError as error:
        findings.append("task dependency cycle: " + " -> ".join(error.args[1]))
    return findings
