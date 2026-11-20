"""Hold `docs/design/spec.md` and `docs/impl/plan.md` to the same account of what the product is.

The spec is a living register: a capability discovered while building is meant to be ADDED to it,
not refused. That only works while the two files agree, and agreement decays silently -- a task
lands under a capability nobody wrote down, a capability ships and its row still says `planned`, a
group drifts out of the implementation line and "what is next" quietly becomes a judgment call
again. None of that breaks a test, so without this check the drift is found by a reader months
later, if at all.

The registry invariants join what the product does (the spec), how we know it works (the
registry's evaluation column), and where the remaining work sits (the plan). The task invariants
keep every task executable without guessing: all nine fields present, a status that matches its
lane, required work ahead of optional work, `Requires` prerequisites that name open tasks and form
no cycle, and plan text that describes only work that remains. Each failure names the document to
fix: the checker never has an opinion about which capability is worth having, only about whether
both files say the same thing about it.
"""

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

from llb.core.paths import PROJECT_ROOT
from llb.quality.plan_dependencies import dependency_findings
from llb.quality.plan_model import (
    AGENT_SECTION,
    HUMAN_SECTION,
    PLAN_SECTIONS,
    REGISTRY_HEADING,
    Capability,
    Task,
    read_registry,
    read_tasks,
)

_LOG = logging.getLogger(__name__)

SPEC_DOC = Path("docs/design/spec.md")
PLAN_DOC = Path("docs/impl/plan.md")

SHIPPED = "shipped"
PLANNED = "planned"
STATUSES = (SHIPPED, PLANNED)
AGENT_STATUSES = ("CLEAR", "RUN NEEDED")
HUMAN_STATUSES = ("BLOCKED BY HUMAN", "HUMAN-GATED")
REQUIRED_FIELDS = (
    "serves",
    "agent status",
    "dependencies",
    "user-visible outcome",
    "scope boundary",
    "data and artifact paths",
    "execution path",
    "acceptance gates",
    "documentation target",
)

_LINK = re.compile(r"\[[^\]]+\]\([^)]+\)")
_HISTORY_WORD = re.compile(r"\b(done|delivered|implemented)\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_TASK_HEADING = re.compile(r"^#### ")
_VALID_TASK_HEADING = re.compile(r"^#### [a-z0-9][a-z0-9-]*(?: \(optional\))?\s*$")


def _registry_findings(registry: list[Capability], tasks: list[Task]) -> list[str]:
    """Each capability is real: a known status, a stated evaluation, and something behind it."""
    findings: list[str] = []
    seen: set[str] = set()
    served = {task.serves for task in tasks}
    for capability in registry:
        where = f"{SPEC_DOC}: `{capability.identifier}`"
        if capability.identifier in seen:
            findings.append(f"{where}: listed twice in the capability registry")
        seen.add(capability.identifier)
        if capability.status not in STATUSES:
            findings.append(f"{where}: status '{capability.status}' is not one of {STATUSES}")
        if not capability.evaluation:
            findings.append(f"{where}: no evaluation declared -- say how it is known to work")
        if capability.status == SHIPPED and not _LINK.search(capability.implementation):
            findings.append(f"{where}: shipped but links to no implementation docs")
        if capability.status == PLANNED and capability.identifier not in served:
            findings.append(f"{where}: planned but no plan task serves it")
    return findings


def _identity_findings(task: Task, known: set[str], identifiers: set[str]) -> list[str]:
    """The task is placed: a unique id, a lane, a group, and a registered capability it serves."""
    where = f"{PLAN_DOC}: `{task.identifier}`"
    findings: list[str] = []
    if task.identifier in identifiers:
        findings.append(f"{where}: task id is duplicated")
    identifiers.add(task.identifier)
    if task.section not in PLAN_SECTIONS:
        findings.append(f"{where}: task sits outside both task lanes ('{task.section}')")
    if not task.group:
        findings.append(f"{where}: task has no capability group")
    if task.serves is None:
        findings.append(f"{where}: no `Serves` line -- name the capability it advances")
    elif task.serves not in known:
        findings.append(f"{where}: serves `{task.serves}`, which is not a registered capability")
    elif task.group and task.serves != task.group:
        findings.append(f"{where}: serves `{task.serves}` under the `{task.group}` group")
    return findings


def _metadata_findings(task: Task) -> list[str]:
    where = f"{PLAN_DOC}: `{task.identifier}`"
    return [
        f"{where}: missing non-empty `{name.capitalize()}` field"
        for name in REQUIRED_FIELDS
        if name != "serves" and not task.fields.get(name)
    ]


def _status_findings(task: Task) -> list[str]:
    """A status names who can finish the task, so it must agree with the lane it sits in."""
    allowed = {AGENT_SECTION: AGENT_STATUSES, HUMAN_SECTION: HUMAN_STATUSES}.get(task.section, ())
    if task.agent_status is None or task.agent_status in allowed:
        return []
    where = f"{PLAN_DOC}: `{task.identifier}`"
    return [f"{where}: status '{task.agent_status}' is invalid in '{task.section}'"]


def _task_findings(registry: list[Capability], tasks: list[Task]) -> list[str]:
    known = {capability.identifier for capability in registry}
    identifiers: set[str] = set()
    findings: list[str] = []
    for task in tasks:
        findings.extend(_identity_findings(task, known, identifiers))
        findings.extend(_metadata_findings(task))
        findings.extend(_status_findings(task))
    return findings


def _line_order_findings(line: list[str], tasks: list[Task], section: str) -> list[str]:
    """Groups run down the registry's order; returning to an earlier group breaks it too."""
    ranks = {group: index for index, group in enumerate(line)}
    groups = [task.group for task in tasks if task.section == section and task.group in ranks]
    ranked = [ranks[group] for group in groups]
    if ranked == sorted(ranked):
        return []
    return [
        f"{PLAN_DOC}: capability groups under '{section}' are out of implementation-line "
        f"order -- got {list(dict.fromkeys(groups))}"
    ]


def _optional_order_findings(tasks: list[Task], section: str) -> list[str]:
    """`(optional)` sorts a task behind every required task of its own group."""
    by_group: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        if task.section == section:
            by_group[task.group].append(task)
    findings: list[str] = []
    for group, group_tasks in by_group.items():
        optional_seen = False
        for task in group_tasks:
            if optional_seen and not task.optional:
                findings.append(
                    f"{PLAN_DOC}: required task `{task.identifier}` follows optional work "
                    f"in `{group}`"
                )
            optional_seen = optional_seen or task.optional
    return findings


def _order_findings(registry: list[Capability], tasks: list[Task]) -> list[str]:
    """The plan's order is what makes "what is next" a position rather than an argument."""
    line = [capability.identifier for capability in registry]
    findings: list[str] = []
    for section in PLAN_SECTIONS:
        findings.extend(_line_order_findings(line, tasks, section))
        findings.extend(_optional_order_findings(tasks, section))
    return findings


def _plan_text_findings(plan: Path) -> list[str]:
    """The plan states only what remains: no history words, no dates, both lanes present."""
    text = plan.read_text(encoding="utf-8")
    findings: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        if _TASK_HEADING.match(line) and not _VALID_TASK_HEADING.match(line):
            findings.append(f"{PLAN_DOC}:{number}: malformed task heading -- `#### <id>`")
        if _HISTORY_WORD.search(line) or _ISO_DATE.search(line):
            findings.append(f"{PLAN_DOC}:{number}: forward plan contains history or a date")
    for section in PLAN_SECTIONS:
        if f"\n## {section}\n" not in f"\n{text}":
            findings.append(f"{PLAN_DOC}: missing required lane '## {section}'")
    return findings


def integrity_findings(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Every disagreement between the specification and the implementation plan."""
    missing = [path for path in (SPEC_DOC, PLAN_DOC) if not (project_root / path).is_file()]
    if missing:
        return [f"missing required document: {path}" for path in missing]
    registry = read_registry(project_root / SPEC_DOC)
    if not registry:
        return [f"{SPEC_DOC}: no capability registry rows found under '{REGISTRY_HEADING}'"]
    tasks = read_tasks(project_root / PLAN_DOC)
    return (
        _registry_findings(registry, tasks)
        + _task_findings(registry, tasks)
        + [f"{PLAN_DOC}: {finding}" for finding in dependency_findings(tasks)]
        + _order_findings(registry, tasks)
        + _plan_text_findings(project_root / PLAN_DOC)
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="check spec/plan capability integrity")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    findings = integrity_findings(args.root)
    for finding in findings:
        _LOG.error("ERROR: %s", finding)
    _LOG.info("[spec-plan] %d integrity finding(s)", len(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
