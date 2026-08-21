"""Branch arcs an operation's declared probe set reaches, scoped to its own function body.

One probe call observes one path.  The operation audit records input reads along that path, while
this module records the complementary fact: which conditional arcs in the operation's own code no
declared point took.  It uses ``sys.monitoring`` rather than tracing lines, because a source line can
hold both outcomes of a conditional and only the ``BRANCH`` event names the destination actually
taken.

The result is deliberately a report, not a refusal.  An operation can have a legal domain that
makes one outcome impossible to probe, such as a quotient whose non-positive denominator guard must
raise.  Until operations can declare such an arc unreachable-by-probe, missed coverage is evidence
for an author rather than a gate.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import dis
import sys
from types import CodeType

from llb.bench.published_value.operations.registry import DerivationOperation

_MONITORING_TOOL_NAME = "llb-operation-probe-branches"
_MONITORING_TOOL_IDS = (4, 3, 5, 2, 1, 0)


@dataclass(frozen=True, slots=True)
class OperationUnreachedBranches:
    """Conditional arcs one operation's probe set missed, named at their source lines.

    ``source_lines`` has one entry per missed arc, so the same line can appear twice when neither
    outcome of a nested conditional was reached.  ``count`` is explicit because this record is the
    compact, user-visible registry report rather than an internal collection to be interpreted.
    """

    operation: str
    count: int
    source_lines: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _BranchArc:
    source_offset: int
    destination_offset: int
    source_line: int


def _is_conditional_branch(instruction: dis.Instruction) -> bool:
    """Whether CPython reports the instruction through ``sys.monitoring.events.BRANCH``."""
    name = instruction.opname
    return name in {"FOR_ITER", "SEND"} or name.startswith("POP_JUMP") or name.startswith("JUMP_IF")


def _branch_arcs(code: CodeType) -> tuple[_BranchArc, ...]:
    """Both static destinations of every conditional instruction in one code object."""
    instructions = tuple(dis.get_instructions(code))
    arcs: list[_BranchArc] = []
    for position, instruction in enumerate(instructions[:-1]):
        if not _is_conditional_branch(instruction):
            continue
        destination = instruction.argval
        if not isinstance(destination, int):
            continue
        source_line = (
            instruction.positions.lineno if instruction.positions is not None else None
        ) or code.co_firstlineno
        fallthrough = instructions[position + 1].offset
        arcs.extend(
            (
                _BranchArc(instruction.offset, fallthrough, source_line),
                _BranchArc(instruction.offset, destination, source_line),
            )
        )
    return tuple(arcs)


def _claim_monitoring_tool() -> int:
    """Reserve one of CPython's six tool IDs without displacing another monitoring client."""
    for tool_id in _MONITORING_TOOL_IDS:
        try:
            sys.monitoring.use_tool_id(tool_id, _MONITORING_TOOL_NAME)
        except ValueError:
            continue
        return tool_id
    raise RuntimeError("no sys.monitoring tool ID is available for operation branch probing")


@dataclass(slots=True)
class OperationBranchMonitor:
    """Accumulate branch arcs across all calls in one operation's declared probe set."""

    operation: DerivationOperation
    _code: CodeType = field(init=False)
    _arcs: tuple[_BranchArc, ...] = field(init=False)
    _reached: set[tuple[int, int]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        code = getattr(self.operation.compute, "__code__", None)
        if not isinstance(code, CodeType):
            raise TypeError(
                f"the `{self.operation.name}` operation's compute callable has no code object to "
                "monitor for branches"
            )
        self._code = code
        self._arcs = _branch_arcs(code)

    @contextmanager
    def recording_call(self) -> Iterator[None]:
        """Record branch destinations while exactly one probe call executes the operation body."""
        tool_id = _claim_monitoring_tool()

        def reached(code: CodeType, source: int, destination: int) -> None:
            if code is self._code:
                self._reached.add((source, destination))

        sys.monitoring.register_callback(tool_id, sys.monitoring.events.BRANCH, reached)
        try:
            sys.monitoring.set_local_events(tool_id, self._code, sys.monitoring.events.BRANCH)
            yield
        finally:
            sys.monitoring.set_local_events(tool_id, self._code, sys.monitoring.events.NO_EVENTS)
            sys.monitoring.register_callback(tool_id, sys.monitoring.events.BRANCH, None)
            sys.monitoring.free_tool_id(tool_id)

    def report(self) -> OperationUnreachedBranches:
        """Name every static conditional arc no recorded call took, in source order."""
        missed = tuple(
            arc
            for arc in self._arcs
            if (arc.source_offset, arc.destination_offset) not in self._reached
        )
        return OperationUnreachedBranches(
            operation=self.operation.name,
            count=len(missed),
            source_lines=tuple(arc.source_line for arc in missed),
        )
