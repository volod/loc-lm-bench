"""Hold `docs/design/spec.md` and `docs/impl/plan.md` to the same account of what the product is.

The spec is a living register: a capability discovered while building is meant to be ADDED to it,
not refused. That only works while the two files agree, and agreement decays silently -- a task
lands under a capability nobody wrote down, a capability ships and its row still says `planned`, a
group drifts out of the implementation line and "what is next" quietly becomes a judgment call
again. None of that breaks a test, so without this check the drift is found by a reader months
later, if at all.

The four invariants below are the join between what the product does (the spec), how we know it
works (the registry's evaluation column), and where the remaining work sits (the plan). Each failure
names the document to fix: the checker never has an opinion about which capability is worth having,
only about whether both files say the same thing about it.
"""

import argparse
import logging
import re
from pathlib import Path

from llb.core.paths import PROJECT_ROOT

_LOG = logging.getLogger(__name__)

SPEC_DOC = Path("docs/design/spec.md")
PLAN_DOC = Path("docs/impl/plan.md")

REGISTRY_HEADING = "## Capability Registry"
SHIPPED = "shipped"
PLANNED = "planned"
STATUSES = (SHIPPED, PLANNED)

_ROW = re.compile(r"^\|(?P<cells>.+)\|\s*$")
_CAPABILITY = re.compile(r"^`(?P<id>[a-z0-9-]+)`$")
_GROUP = re.compile(r"^### .+ -- `(?P<id>[a-z0-9-]+)`\s*$")
_TASK = re.compile(r"^#### (?P<id>\S.*?)(?: \(optional\))?\s*$")
_SERVES = re.compile(r"^- Serves: `(?P<id>[a-z0-9-]+)`")
_LINK = re.compile(r"\[[^\]]+\]\([^)]+\)")
_SECTION = re.compile(r"^## (?P<title>.+?)\s*$")


class Capability:
    """One registry row: what the product does, how it is evaluated, and where it lives."""

    def __init__(self, identifier: str, status: str, evaluation: str, implementation: str):
        self.id = identifier
        self.status = status
        self.evaluation = evaluation
        self.implementation = implementation


class Task:
    """One plan entry, with the group it sits under and the capability it declares."""

    def __init__(self, identifier: str, section: str, group: str, serves: str | None):
        self.id = identifier
        self.section = section
        self.group = group
        self.serves = serves


def _cells(line: str) -> list[str] | None:
    matched = _ROW.match(line.strip())
    return [cell.strip() for cell in matched.group("cells").split("|")] if matched else None


def read_registry(spec: Path) -> list[Capability]:
    """The capability rows, in the order the spec lists them -- which is the implementation line."""
    found: list[Capability] = []
    inside = False
    for line in spec.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.strip() == REGISTRY_HEADING
            continue
        cells = _cells(line) if inside else None
        if not cells or len(cells) < 5:
            continue
        identified = _CAPABILITY.match(cells[1])
        if identified:
            found.append(Capability(identified.group("id"), cells[2], cells[3], cells[4]))
    return found


def read_tasks(plan: Path) -> list[Task]:
    """Every plan task with its section, its capability group, and its declared `Serves` id."""
    found: list[Task] = []
    section = group = ""
    for line in plan.read_text(encoding="utf-8").splitlines():
        heading = _SECTION.match(line)
        if heading and not line.startswith("###"):
            section, group = heading.group("title"), ""
            continue
        grouped = _GROUP.match(line)
        if grouped:
            group = grouped.group("id")
            continue
        task = _TASK.match(line)
        if task:
            found.append(Task(task.group("id"), section, group, None))
            continue
        serves = _SERVES.match(line)
        if serves and found:
            found[-1].serves = found[-1].serves or serves.group("id")
    return found


def _registry_findings(registry: list[Capability], tasks: list[Task]) -> list[str]:
    """Each capability is real: a known status, a stated evaluation, and something behind it."""
    findings: list[str] = []
    seen: set[str] = set()
    served = {task.serves for task in tasks}
    for capability in registry:
        where = f"{SPEC_DOC}: `{capability.id}`"
        if capability.id in seen:
            findings.append(f"{where}: listed twice in the capability registry")
        seen.add(capability.id)
        if capability.status not in STATUSES:
            findings.append(f"{where}: status '{capability.status}' is not one of {STATUSES}")
        if not capability.evaluation:
            findings.append(f"{where}: no evaluation declared -- say how it is known to work")
        if capability.status == SHIPPED and not _LINK.search(capability.implementation):
            findings.append(f"{where}: shipped but links to no implementation docs")
        if capability.status == PLANNED and capability.id not in served:
            findings.append(f"{where}: planned but no plan task serves it")
    return findings


def _task_findings(registry: list[Capability], tasks: list[Task]) -> list[str]:
    """Each task declares a registered capability, and sits in that capability's group."""
    known = {capability.id for capability in registry}
    findings: list[str] = []
    for task in tasks:
        where = f"{PLAN_DOC}: `{task.id}`"
        if task.serves is None:
            findings.append(f"{where}: no `Serves` line -- name the capability it advances")
        elif task.serves not in known:
            findings.append(
                f"{where}: serves `{task.serves}`, which is not a registered capability"
            )
        elif task.group and task.serves != task.group:
            findings.append(f"{where}: serves `{task.serves}` under the `{task.group}` group")
    return findings


def _order_findings(registry: list[Capability], tasks: list[Task]) -> list[str]:
    """The plan's groups run down the registry's order, so "what is next" stays a position."""
    line = [capability.id for capability in registry]
    findings: list[str] = []
    for section in dict.fromkeys(task.section for task in tasks):
        groups = dict.fromkeys(
            task.group for task in tasks if task.section == section and task.group
        )
        ranked = [line.index(group) for group in groups if group in line]
        if ranked != sorted(ranked):
            findings.append(
                f"{PLAN_DOC}: capability groups under '{section}' are out of implementation-line "
                f"order -- got {list(groups)}"
            )
    return findings


def integrity_findings(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Every disagreement between the specification and the implementation plan."""
    registry = read_registry(project_root / SPEC_DOC)
    if not registry:
        return [f"{SPEC_DOC}: no capability registry rows found under '{REGISTRY_HEADING}'"]
    tasks = read_tasks(project_root / PLAN_DOC)
    return (
        _registry_findings(registry, tasks)
        + _task_findings(registry, tasks)
        + _order_findings(registry, tasks)
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
