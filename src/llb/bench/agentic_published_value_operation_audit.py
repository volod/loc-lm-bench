"""The other direction of an operation's declaration: what its BODY reads, and who names it at all.

`source_forms`, `stated_fields`, and `reads_own_measurement` are checked against the DESIGN before a
number is read (`_check_operands` and `_stated_operands` in `agentic_published_value_derivation`),
which is one direction of a two-directional claim. Nothing checked them against the FUNCTION
registered beside them, so three defects were invisible until a study adopted the arithmetic:

  - a body that reads a stated field it did not declare raises a `KeyError` inside whichever reader
    got there first, instead of refusing the design that failed to state it;
  - a declaration listing an input the body never reads makes every adopting design carry a number
    for nothing, and nothing ever checks that number;
  - an operation no registered design names is arithmetic nobody exercises -- where a wrong quotient
    would sit until the first study adopted it and published a number out of it.

The first two are read off the recording probe next door
(`agentic_published_value_operation_probe`), which calls each registered operation through inputs
answering only its declaration -- at every point of the probe SET the operation declares, unioning
the reads, because a body reads its declaration along a path and one point observes one path. The
third is a walk of the registered DESIGNS, which is why a registry entry carries a reader for its
published values: no per-design validation can answer a question about the designs collectively.
All three are refusals rather than notes, on the reasoning
the registry uses everywhere else -- arithmetic exercised by no design is arithmetic whose first
exercise is a published number.

What deliberately stays out is any claim that an operation is PURE or deterministic beyond its
declared inputs. The probe records reaches through design inputs directly and checks shipped policy
fields by perturbing the `ContextPolicy` supplied to every point. An unrelated module global remains
out of reach of anything short of an expression language, which is exactly what the design file does
not have. Branch COVERAGE is reported rather than refused: ``sys.monitoring`` records the arcs each
probe call takes through the operation's own code, making a missed path visible without rejecting a
legal domain guard that no successful probe may take.
"""

from dataclasses import dataclass
from pathlib import Path

from llb.bench.agentic_published_value_operation_branches import (
    OperationBranchMonitor,
    OperationUnreachedBranches,
)
from llb.bench.agentic_published_value_operation_probe import (
    UndeclaredRead,
    declared_reads,
    probe_inputs,
)
from llb.bench.agentic_published_value_operation_policy import policy_declaration_refusals
from llb.bench.agentic_published_value_operations import (
    DERIVATION_OPERATIONS,
    OPERATION,
    DerivationInputs,
    DerivationOperation,
    probe_point_named,
)
from llb.bench.agentic_published_value_registry import (
    PUBLISHED_VALUE_DESIGNS,
    registered_design_path,
)


def _named_at(operation: DerivationOperation, position: int) -> str:
    """The operation as a refusal names it, located at a point only when the set has more than one.

    Unqualified for a one-point set, because "at probe point 1" is noise an author has to read past
    when there is nowhere else the reach could have happened.
    """
    named = f"the `{operation.name}` operation"
    if len(operation.probes) == 1:
        return named
    return f"{named} at {probe_point_named(position)}"


def _reach_refusals(
    operation: DerivationOperation,
    position: int,
    probe: DerivationInputs,
    reads: set[str],
    branches: OperationBranchMonitor,
) -> tuple[str, ...]:
    """Call the operation at ONE point of its set, recording into `reads` and naming any reach out."""
    named = _named_at(operation, position)
    inputs = probe_inputs(operation, probe, reads)
    try:
        with branches.recording_call():
            operation.apply(inputs.sources, inputs.stated, measured=inputs.measured, where=named)
    except UndeclaredRead as undeclared:
        return (
            f"{named} reads {undeclared.read}, which its declaration does not carry -- a design "
            "adopting it fails inside whichever reader got there first, instead of being refused "
            "for not stating what the arithmetic is computed over",
        )
    except (KeyError, IndexError, TypeError) as reached:
        return (
            f"{named} reaches outside the inputs it declares ({type(reached).__name__}: "
            f"{reached}), so what it is computed over is not what it says it is",
        )
    except ValueError as broken:
        return (
            f"{named} does not compute at the probe point it declares ({broken}), so nothing can "
            "check its declaration against its arithmetic",
        )
    return ()


def _over_declaration_refusals(operation: DerivationOperation, reads: set[str]) -> tuple[str, ...]:
    """Read the UNION of the set's reads off, and name every declared input none of them reached."""
    across = "" if len(operation.probes) == 1 else f" at any of its {len(operation.probes)} points"
    return tuple(
        f"the `{operation.name}` operation declares {read} and never reads it{across}, so every "
        "design adopting it would carry a number for nothing and nothing would ever check that "
        "number"
        for read in declared_reads(operation)
        if read not in reads
    )


def _audit_operation(
    operation: DerivationOperation,
) -> tuple[tuple[str, ...], OperationUnreachedBranches]:
    """Call one registered operation at every point of its probe set, and say where it disagrees.

    One recording spans the whole set: over-declaration is read off the UNION, so an input the body
    reaches for only on a branch counts as read once the point taking that branch is declared. The
    first point that reaches OUTSIDE the declaration ends the walk instead -- the reach is the
    answer, and the remaining points would restate it or bury it.
    """
    reads: set[str] = set()
    branches = OperationBranchMonitor(operation)
    for position, probe in enumerate(operation.probes):
        reached = _reach_refusals(operation, position, probe, reads, branches)
        if reached:
            return reached, branches.report()
    refusals = (
        *_over_declaration_refusals(operation, reads),
        *policy_declaration_refusals(operation),
    )
    return refusals, branches.report()


def operation_refusals(operation: DerivationOperation) -> tuple[str, ...]:
    """Return declaration/body disagreements; branch misses remain report-only evidence."""
    refusals, _branches = _audit_operation(operation)
    return refusals


def published_operations(design_root: Path) -> dict[str, list[str]]:
    """Every arithmetic a registered design names, mapped to the designs that name it.

    Mapped rather than collected into a set for the reason `published_citations` is: a walk over the
    registry is only useful when its answer can name WHOSE design a refusal is about.
    """
    named: dict[str, list[str]] = {}
    for kind, design in sorted(PUBLISHED_VALUE_DESIGNS.items()):
        path = registered_design_path(kind, design, design_root)
        for value in design.published_values(path):
            operation = value.get(OPERATION)
            if not isinstance(operation, str):
                continue
            naming = named.setdefault(operation, [])
            if kind not in naming:
                naming.append(kind)
    return named


def unpublished_operations(design_root: Path) -> tuple[str, ...]:
    """Registered arithmetic no registered design names, in registry order."""
    named = published_operations(design_root)
    return tuple(name for name in sorted(DERIVATION_OPERATIONS) if name not in named)


@dataclass(frozen=True, slots=True)
class OperationRegistryReport:
    """One registry self-check: what it called, missed branches, and declaration/body refusals.

    `checked` is carried for the reason the design walk carries `walked` -- a self-check that
    exercised nothing passes exactly like one that exercised every entry, so the caller states what
    it expected rather than trusting a clean run.  `unreached_branches` has one record per checked
    operation, including a zero count, so adding a path changes visible evidence without becoming a
    refusal before unreachable-by-probe branches can be declared.
    """

    checked: tuple[str, ...]
    unreached_branches: tuple[OperationUnreachedBranches, ...]
    refusals: tuple[str, ...]


def report_operation_registry(*, design_root: Path) -> OperationRegistryReport:
    """Probe every registered operation and walk the designs, collecting refusals instead of raising.

    Collecting primitive, refusing wrapper -- the shape the design registry already uses, so what CI
    fails on and what a caller reads are one walk rather than two that can drift.
    """
    checked = tuple(sorted(DERIVATION_OPERATIONS))
    refusals: list[str] = []
    unreached_branches: list[OperationUnreachedBranches] = []
    for name in checked:
        operation_refusal, branch_report = _audit_operation(DERIVATION_OPERATIONS[name])
        refusals.extend(operation_refusal)
        unreached_branches.append(branch_report)
    refusals.extend(
        f"the `{name}` operation is registered and no registered design names it, so its arithmetic "
        "is exercised by nothing -- a wrong quotient would sit here until the first study adopted "
        "it and published a number out of it"
        for name in unpublished_operations(design_root)
    )
    return OperationRegistryReport(
        checked=checked,
        unreached_branches=tuple(unreached_branches),
        refusals=tuple(refusals),
    )


def validate_operation_registry(*, design_root: Path) -> list[str]:
    """The refusing form: probe every registered operation, refusing if any disagrees with itself.

    Returns the operations checked, and names EVERY refusal rather than the first, because a reader
    fixing one registration at a time is the same slow loop in CI that it is at a refresh.
    """
    report = report_operation_registry(design_root=design_root)
    if report.refusals:
        raise ValueError("; ".join(report.refusals))
    return list(report.checked)
