"""What a published value is COMPUTED OUT OF and HOW, declared by the design rather than by a reader.

A design that publishes several numbers usually publishes some that are not measurements: a ratio is
a quotient of a guard, a normalized cost is a cost over a peak, a span is a difference of two edges.
That distinction decides what a collecting validation may SAY. When the guard a ratio divides no
longer resolves, the ratio is not a second moved number -- it is the consequence of the first, and
naming both sends the operator to restate a figure nothing can evaluate until the cause is restated.

The rule is real; what was wrong was where it lived. It was written into one study's resolver as the
single edge "the trigger ratio is a quotient of the boundary surface's interpolated guard at the same
depth", so the study-agnostic accumulator could not see it, a second publishing design inherited
nothing, and a value derived from two others -- or from another derived value -- had no way to say
so. Here the EDGE is a property of the published value, declared beside its provenance:

    "derived_from": [{"study_kind": ..., "depth": ..., "form": ...}, ...]
    "operation": "<a name the operation registry carries>"
    "reading": "<a name the reading registry carries>"

Each entry names another value the SAME design publishes, so the declaration is checkable against the
design rather than taken on faith -- which is what the DESIGN-wide walk next door
(`agentic_published_value_derivation_graph`) does with what is read here.

The sources are half a derivation. The other half is the ARITHMETIC over them, which used to live in
the readers -- so `operation` names a registered pure function
(`agentic_published_value_operations`), and the two halves are validated together: the operation
states how many sources of which form it takes, and a declaration that does not match it is refused
here rather than misread later. A value declares both or neither: sources with no operation is a
number nothing can re-derive, and an operation with no sources is arithmetic over nothing.

Reproducing the number still does not say how its published statement is JUDGED, so a derived value
also names a registered `reading`. `Derivation` binds that comparison beside the operation, making a
missing or unknown rule a declaration refusal rather than a choice each downstream reader repeats.
"""

from dataclasses import dataclass
from typing import cast

from llb.bench.published_value.operations.registry import (
    OPERATION,
    DerivationOperation,
    DerivedValue,
    registered_operation,
)
from llb.bench.published_value.readings import (
    READING,
    PublishedValueReading,
    required_reading,
)

# The field a published value declares its sources in. A list rather than a single object, because a
# derived value can rest on two published numbers (a difference, a normalization against a second
# study's peak) as readily as on one; a value that is measured rather than derived declares nothing.
DERIVED_FROM = "derived_from"


@dataclass(frozen=True, slots=True)
class ValueKey:
    """One published value's identity inside a design: which study, which depth, which form.

    The form is part of the identity rather than decoration: one study can publish two numbers at a
    depth -- a guard and the boundary of the step it sits in -- and a declaration that named only the
    depth would be ambiguous exactly where a derivation edge has to be exact.
    """

    study_kind: str
    depth: int
    form: str

    def label(self) -> str:
        return f"{self.study_kind} depth {self.depth} {self.form}"


def published_key(value: dict[str, object]) -> ValueKey:
    """The identity of one published value, refusing a row whose identity cannot be read."""
    depth = value.get("depth")
    if isinstance(depth, bool) or not isinstance(depth, (int, float)):
        raise ValueError(
            f"{value.get('study_kind')} depth {depth!r}: a published value must state a numeric "
            "`depth`, because that is part of what a derivation edge names"
        )
    return ValueKey(str(value.get("study_kind")), int(depth), str(value.get("form")))


def declared_sources(value: dict[str, object]) -> tuple[ValueKey, ...]:
    """The values this one DECLARES it is computed out of, or nothing when it is a measurement."""
    declared = value.get(DERIVED_FROM)
    if declared is None:
        return ()
    key = published_key(value)
    if not isinstance(declared, list) or not declared:
        raise ValueError(
            f"{key.label()}: `{DERIVED_FROM}` states what this value is computed out of, so it must "
            f"be a non-empty list of sources; drop it entirely if the value is measured, got "
            f"{declared!r}"
        )
    return tuple(_source_key(entry, key) for entry in cast(list[object], declared))


@dataclass(frozen=True, slots=True)
class Derivation:
    """One derived value's sources, arithmetic, and reading of its published statement.

    Built once and handed to whoever re-derives the value, so the design is read in one place and
    every reader computes the number the same way by construction rather than by review.
    """

    key: ValueKey
    operation: DerivationOperation
    reading: PublishedValueReading
    sources: tuple[ValueKey, ...]
    stated: dict[str, float]

    def source_of_form(self, form: str) -> ValueKey:
        """The one source of a given form, for a reader that needs the identity and not the value.

        Still asked for by SHAPE -- a row naming which study its figure came from knows it names a
        guard, not which study publishes that guard -- and the declaration answers with the edge.
        """
        sources = [source for source in self.sources if source.form == form]
        if len(sources) != 1:
            raise ValueError(
                f"{self.key.label()}: this reader needs exactly one {form} among the sources "
                f"`{DERIVED_FROM}` declares, and the value declares {len(sources)} of them, so "
                "nothing in the run can restate it"
            )
        return sources[0]

    def compute(self, sources: tuple[float, ...], *, measured: float | None = None) -> DerivedValue:
        """Re-derive the value from its sources' values, through the operation the design named."""
        return self.operation.apply(sources, self.stated, measured=measured, where=self.key.label())


def declared_derivation(value: dict[str, object]) -> Derivation | None:
    """What a published value declares it is computed out of and how, or None when it is measured.

    Sources, operation, and reading are refused together when partial. Silence on all three is the
    ordinary measured value; any declaration makes it derived and requires the other two.
    """
    key = published_key(value)
    sources = declared_sources(value)
    named = value.get(OPERATION)
    named_reading = value.get(READING)
    if not sources and named is None:
        if named_reading is None:
            return None
        raise ValueError(
            f"{key.label()}: it names the `{READING}` {named_reading!r} but declares neither "
            f"`{DERIVED_FROM}` nor `{OPERATION}`; a measured value declares none of the three"
        )
    if named is None:
        raise ValueError(
            f"{key.label()}: it declares what it is derived from but names no `{OPERATION}`, so "
            f"the arithmetic over {', '.join(source.label() for source in sources)} lives wherever "
            "someone wrote it rather than in the design -- nothing in the run can restate it"
        )
    operation = registered_operation(named, where=key.label())
    if not sources:
        raise ValueError(
            f"{key.label()}: it names the `{OPERATION}` {operation.name!r}, which is computed over "
            f"{len(operation.source_forms)} declared source(s), while `{DERIVED_FROM}` names none, "
            "so nothing in the run can restate it"
        )
    _check_operands(key, operation, sources)
    return Derivation(
        key=key,
        operation=operation,
        reading=required_reading(value, where=key.label()),
        sources=sources,
        stated=_stated_operands(value, key, operation),
    )


def required_derivation(value: dict[str, object]) -> Derivation:
    """The same, for a reader whose whole job is re-deriving this value: a measurement is a defect."""
    derivation = declared_derivation(value)
    if derivation is None:
        raise ValueError(
            f"{published_key(value).label()}: this value is derived, so it must declare its "
            f"sources, its `{OPERATION}`, and its `{READING}` -- it declares none, so nothing in the run "
            "can restate it"
        )
    return derivation


def _check_operands(
    key: ValueKey, operation: DerivationOperation, sources: tuple[ValueKey, ...]
) -> None:
    """Refuse a declaration whose sources are not the ones its named arithmetic is computed over.

    Positional rather than by set, because an operation over two sources of ONE form (a span between
    two edges) is ordinary, and the order is then the only thing that says which is which.
    """
    if len(sources) != len(operation.source_forms):
        raise ValueError(
            f"{key.label()}: the `{operation.name}` operation is computed over "
            f"{len(operation.source_forms)} source(s) ({', '.join(operation.source_forms)}), while "
            f"`{DERIVED_FROM}` names {len(sources)}"
        )
    for position, (source, form) in enumerate(zip(sources, operation.source_forms)):
        if source.form != form:
            raise ValueError(
                f"{key.label()}: the `{operation.name}` operation takes a {form} as source "
                f"{position + 1}, while `{DERIVED_FROM}` names {source.label()} there"
            )


def _stated_operands(
    value: dict[str, object], key: ValueKey, operation: DerivationOperation
) -> dict[str, float]:
    """The value's own fields the named arithmetic reads, refusing one the design does not state."""
    operands: dict[str, float] = {}
    for name in operation.stated_fields:
        stated = value.get(name)
        if isinstance(stated, bool) or not isinstance(stated, (int, float)):
            raise ValueError(
                f"{key.label()}: the `{operation.name}` operation is computed over this value's own "
                f"`{name}`, so the design must state it numerically, got {stated!r}"
            )
        operands[name] = float(stated)
    return operands


def _source_key(entry: object, key: ValueKey) -> ValueKey:
    """One declared source, refusing an entry that does not name a value completely."""
    named = cast(dict[str, object], entry) if isinstance(entry, dict) else {}
    study_kind = named.get("study_kind")
    depth = named.get("depth")
    form = named.get("form")
    if (
        not isinstance(study_kind, str)
        or not isinstance(form, str)
        or isinstance(depth, bool)
        or not isinstance(depth, int)
    ):
        raise ValueError(
            f"{key.label()}: every `{DERIVED_FROM}` entry must name the `study_kind`, `depth`, and "
            f"`form` of the published value it is computed out of, got {entry!r}"
        )
    return ValueKey(study_kind, depth, form)
