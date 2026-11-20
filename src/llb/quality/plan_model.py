"""Read the capability registry in `docs/design/spec.md` and the tasks in `docs/impl/plan.md`.

Both readers are deliberately literal: they recognize the headings and field bullets the
[planning workflow](../../../docs/guides/development/planning-workflow.md) prescribes and nothing
else, so a task written in another shape surfaces as a finding instead of being guessed at.

A field's value may wrap onto indented continuation lines -- the plan wraps at 100 columns, and a
value such as a documentation target often starts on the line after its label. Fenced blocks are
skipped, so a task-shape example quoted inside a fence is never read as a task.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY_HEADING = "## Capability Registry"
AGENT_SECTION = "Agent Implementation Tasks"
HUMAN_SECTION = "Human-Assisted Tasks"
PLAN_SECTIONS = (AGENT_SECTION, HUMAN_SECTION)
# An `Agent status` may carry its reason after this separator: `CLEAR -- <why it is clear>`.
STATUS_REASON_SEPARATOR = " -- "
_REGISTRY_COLUMNS = 5

_FENCE = "```"
_ROW = re.compile(r"^\|(?P<cells>.+)\|\s*$")
_CAPABILITY = re.compile(r"^`(?P<id>[a-z0-9-]+)`$")
_SECTION = re.compile(r"^## (?P<title>.+?)\s*$")
_GROUP = re.compile(r"^### .+ -- `(?P<id>[a-z0-9-]+)`\s*$")
_TASK = re.compile(r"^#### (?P<id>[a-z0-9][a-z0-9-]*)(?P<optional> \(optional\))?\s*$")
_FIELD = re.compile(r"^- (?P<key>[A-Za-z][A-Za-z -]+):\s*(?P<value>.*)$")
_CONTINUATION = re.compile(r"^\s+\S")
_SERVES = re.compile(r"^`(?P<id>[a-z0-9-]+)`(?:\s|$)")


@dataclass(frozen=True, slots=True)
class Capability:
    """One registry row: what the product does, how it is evaluated, and where it lives."""

    identifier: str
    status: str
    evaluation: str
    implementation: str


@dataclass(slots=True)
class Task:
    """One plan entry, with its lane, the group it sits under, and its field bullets."""

    identifier: str
    section: str
    group: str
    optional: bool
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def serves(self) -> str | None:
        """The capability id the `Serves` field names, or None when it names none."""
        matched = _SERVES.match(self.fields.get("serves", ""))
        return matched.group("id") if matched else None

    @property
    def agent_status(self) -> str | None:
        """The normalized status marker, without any `-- reason` that follows it."""
        value = self.fields.get("agent status")
        if not value:
            return None
        return value.split(STATUS_REASON_SEPARATOR, 1)[0].strip().upper()


@dataclass(slots=True)
class _Cursor:
    """Where the reader stands in the plan: lane, group, task, and the field still open."""

    section: str = ""
    group: str = ""
    task: Task | None = None
    open_field: str | None = None
    fenced: bool = False


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
        if not cells or len(cells) < _REGISTRY_COLUMNS:
            continue
        identified = _CAPABILITY.match(cells[1])
        if identified:
            found.append(Capability(identified.group("id"), cells[2], cells[3], cells[4]))
    return found


def _new_task(line: str, cursor: _Cursor) -> Task | None:
    matched = _TASK.match(line)
    if matched is None:
        return None
    return Task(
        identifier=matched.group("id"),
        section=cursor.section,
        group=cursor.group,
        optional=bool(matched.group("optional")),
    )


def _enter_heading(line: str, cursor: _Cursor, tasks: list[Task]) -> bool:
    """Move the cursor over a lane, group, or task heading; False when the line is none of them."""
    if line.startswith("## "):
        section = _SECTION.match(line)
        cursor.section = section.group("title") if section else ""
        cursor.group, cursor.task = "", None
    elif line.startswith("### "):
        grouped = _GROUP.match(line)
        cursor.group = grouped.group("id") if grouped else ""
        cursor.task = None
    elif line.startswith("#### "):
        # A malformed task heading still ends the previous task, so its fields go nowhere.
        cursor.task = _new_task(line, cursor)
        if cursor.task is not None:
            tasks.append(cursor.task)
    else:
        return False
    cursor.open_field = None
    return True


def _read_field(line: str, cursor: _Cursor) -> None:
    """Record a field bullet, extend the open field with a continuation, or close it."""
    task = cursor.task
    if task is None:
        return
    matched = _FIELD.match(line)
    if matched:
        key = matched.group("key").strip().lower()
        # The first occurrence of a field wins; a repeated label and its continuation are ignored.
        cursor.open_field = None if key in task.fields else key
        if cursor.open_field:
            task.fields[key] = matched.group("value").strip()
        return
    if cursor.open_field and _CONTINUATION.match(line):
        joined = f"{task.fields[cursor.open_field]} {line.strip()}"
        task.fields[cursor.open_field] = joined.strip()
        return
    cursor.open_field = None


def read_tasks(plan: Path) -> list[Task]:
    """Every plan task in document order, with its lane, capability group, and field values."""
    tasks: list[Task] = []
    cursor = _Cursor()
    for line in plan.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith(_FENCE):
            cursor.fenced = not cursor.fenced
            continue
        if cursor.fenced or _enter_heading(line, cursor, tasks):
            continue
        _read_field(line, cursor)
    return tasks
