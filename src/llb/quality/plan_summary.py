"""Count the forward plan and name the task each lane should take next.

The scheduling rule is the plan's own: take the first task of the earliest group that still has
one. `(optional)` already sorts behind required work inside its group (the integrity gate enforces
it), and an all-optional capability moves down the registry rather than being skipped here, so
document order IS the priority. The one thing document order cannot see is a task still waiting
on an open `Requires` prerequisite, so the summary skips those.

The summary refuses to schedule over an invalid plan: a priority read off documents that disagree
would be a guess presented as a position.
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

from llb.core.paths import PROJECT_ROOT
from llb.quality.plan_dependencies import prerequisites
from llb.quality.plan_model import AGENT_SECTION, HUMAN_SECTION, PLAN_SECTIONS, Task, read_tasks
from llb.quality.spec_plan_integrity import PLAN_DOC, integrity_findings

_LOG = logging.getLogger(__name__)

INVALID_PLAN_MESSAGE = "plan is invalid; run make lint-spec-plan"
_LANE_LABELS = {AGENT_SECTION: "agent", HUMAN_SECTION: "human"}


def _lane_priority(lane_tasks: list[Task]) -> str:
    """The first task in document order with no open prerequisite, as `id [group]`."""
    task = next((task for task in lane_tasks if not prerequisites(task)), None)
    if task is None:
        return "all waiting on prerequisites" if lane_tasks else "none"
    optional = " (optional)" if task.optional else ""
    return f"{task.identifier}{optional} [{task.group}]"


def summary_lines(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Lane counts, status counts, and each lane's priority, after validating the plan."""
    if integrity_findings(project_root):
        return [INVALID_PLAN_MESSAGE]
    tasks = read_tasks(project_root / PLAN_DOC)
    lanes = {
        section: [task for task in tasks if task.section == section] for section in PLAN_SECTIONS
    }
    statuses = Counter(task.agent_status for task in tasks if task.agent_status)
    lines = [f"tasks: {len(tasks)}"]
    lines.extend(
        f"{_LANE_LABELS[section]} lane: {len(lanes[section])}" for section in PLAN_SECTIONS
    )
    if statuses:
        lines.append("statuses: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    lines.append(f"waiting on open Requires: {sum(bool(prerequisites(task)) for task in tasks)}")
    lines.extend(
        f"priority {_LANE_LABELS[section]} task: {_lane_priority(lanes[section])}"
        for section in PLAN_SECTIONS
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="summarize the forward implementation plan")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    lines = summary_lines(args.root)
    for line in lines:
        _LOG.info("%s", line)
    return 1 if lines == [INVALID_PLAN_MESSAGE] else 0


if __name__ == "__main__":
    raise SystemExit(main())
