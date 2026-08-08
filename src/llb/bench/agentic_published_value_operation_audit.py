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
answering only its declaration. The third is a walk of the registered DESIGNS, which is why a
registry entry carries a reader for its published values: no per-design validation can answer a
question about the designs collectively. All three are refusals rather than notes, on the reasoning
the registry uses everywhere else -- arithmetic exercised by no design is arithmetic whose first
exercise is a published number.

What deliberately stays out is any claim that an operation is PURE or deterministic beyond its
declared inputs. The probe records reaches through the inputs it was handed; an operation that reads
a module global is out of reach of anything short of an expression language, which is exactly what
the design file does not have.
"""

from dataclasses import dataclass
from pathlib import Path

from llb.bench.agentic_published_value_operation_probe import (
    UndeclaredRead,
    declared_reads,
    probe_inputs,
)
from llb.bench.agentic_published_value_operations import (
    DERIVATION_OPERATIONS,
    OPERATION,
    DerivationOperation,
)
from llb.bench.agentic_published_value_registry import (
    PUBLISHED_VALUE_DESIGNS,
    registered_design_path,
)


def operation_refusals(operation: DerivationOperation) -> tuple[str, ...]:
    """Call one registered operation through its declaration, and say where the two disagree."""
    reads: set[str] = set()
    inputs = probe_inputs(operation, reads)
    named = f"the `{operation.name}` operation"
    try:
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
    return tuple(
        f"{named} declares {read} and never reads it, so every design adopting it would carry a "
        "number for nothing and nothing would ever check that number"
        for read in declared_reads(operation)
        if read not in reads
    )


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
    """One self-check of the operation registry: what it called, and where declaration met body.

    `checked` is carried for the reason the design walk carries `walked` -- a self-check that
    exercised nothing passes exactly like one that exercised every entry, so the caller states what
    it expected rather than trusting a clean run.
    """

    checked: tuple[str, ...]
    refusals: tuple[str, ...]


def report_operation_registry(*, design_root: Path) -> OperationRegistryReport:
    """Probe every registered operation and walk the designs, collecting refusals instead of raising.

    Collecting primitive, refusing wrapper -- the shape the design registry already uses, so what CI
    fails on and what a caller reads are one walk rather than two that can drift.
    """
    checked = tuple(sorted(DERIVATION_OPERATIONS))
    refusals: list[str] = []
    for name in checked:
        refusals.extend(operation_refusals(DERIVATION_OPERATIONS[name]))
    refusals.extend(
        f"the `{name}` operation is registered and no registered design names it, so its arithmetic "
        "is exercised by nothing -- a wrong quotient would sit here until the first study adopted "
        "it and published a number out of it"
        for name in unpublished_operations(design_root)
    )
    return OperationRegistryReport(checked=checked, refusals=tuple(refusals))


def validate_operation_registry(*, design_root: Path) -> list[str]:
    """The refusing form: probe every registered operation, refusing if any disagrees with itself.

    Returns the operations checked, and names EVERY refusal rather than the first, because a reader
    fixing one registration at a time is the same slow loop in CI that it is at a refresh.
    """
    report = report_operation_registry(design_root=design_root)
    if report.refusals:
        raise ValueError("; ".join(report.refusals))
    return list(report.checked)
