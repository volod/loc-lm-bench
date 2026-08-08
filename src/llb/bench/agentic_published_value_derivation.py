"""What a published value is COMPUTED OUT OF, declared by the design rather than known to a resolver.

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

Each entry names another value the SAME design publishes, so the declaration is checkable against the
design rather than taken on faith: a source the design does not publish, a value that declares itself,
and a cycle are all refused before a single number is read. What the graph then answers is transitive
-- a value two derivation steps above a moved measurement names that measurement, not the derived
figure in between, because the figure in between is a consequence too.
"""

from dataclasses import dataclass, field
from typing import cast

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


@dataclass(frozen=True, slots=True)
class DerivationGraph:
    """Every declared derivation edge of one design, and what each value transitively rests on."""

    sources: dict[ValueKey, tuple[ValueKey, ...]] = field(default_factory=dict)

    def sources_of(self, key: ValueKey) -> tuple[ValueKey, ...]:
        """The values this one declares it is computed out of, directly."""
        return self.sources.get(key, ())

    def unresolved_roots(
        self, key: ValueKey, unresolved: frozenset[ValueKey] | set[ValueKey]
    ) -> tuple[ValueKey, ...]:
        """Every value in `unresolved` this one rests on, naming the ROOT of each chain only.

        A chain is walked THROUGH a source that resolved and stops AT one that did not: a value that
        failed its own resolution is the measurement to restate, and whatever it in turn derives from
        is beside the point, because restating it is what makes the rest checkable again. That is
        what keeps a two-step derivation from naming the derived figure in between -- itself a
        consequence -- instead of the measurement at the root.
        """
        roots: list[ValueKey] = []
        seen: set[ValueKey] = {key}
        queue: list[ValueKey] = list(self.sources_of(key))
        while queue:
            source = queue.pop(0)
            if source in seen:
                continue
            seen.add(source)
            if source in unresolved:
                roots.append(source)
            else:
                queue.extend(self.sources_of(source))
        return tuple(sorted(roots, key=ValueKey.label))


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


def derived_source_of_form(value: dict[str, object], form: str) -> ValueKey:
    """The one source of a given form a derived value declares, refusing anything but exactly one.

    The arithmetic that re-derives a value knows how many inputs of which form it takes -- that is
    the part of a derivation no declaration can carry. What it must NOT know is which study, depth,
    and form supply them, so it asks for its shape and the design answers with the edge.
    """
    key = published_key(value)
    sources = [source for source in declared_sources(value) if source.form == form]
    if len(sources) != 1:
        raise ValueError(
            f"{key.label()}: this value is computed out of exactly one {form}, which it must name "
            f"in `{DERIVED_FROM}` -- it declares {len(sources)}, so nothing in the run can restate it"
        )
    return sources[0]


def derivation_graph(values: list[dict[str, object]]) -> DerivationGraph:
    """Read every declaration in one design, refusing any the design itself cannot support.

    Fail-fast rather than collected, for the reason shape refusals are: a declaration naming a value
    the design does not publish is not evidence that moved, it is a design that never said what its
    number rests on, and that message must not arrive underneath the consequences it causes.
    """
    published = _published_keys(values)
    sources: dict[ValueKey, tuple[ValueKey, ...]] = {}
    for value in values:
        key = published_key(value)
        declared = declared_sources(value)
        for source in declared:
            if source == key:
                raise ValueError(
                    f"{key.label()}: a published value declares itself as its own source, so there "
                    "is nothing underneath it that could resolve it"
                )
            if source not in published:
                raise ValueError(
                    f"{key.label()}: it declares that its value is derived from {source.label()}, "
                    "which this design does not publish, so nothing here can tell whether what it "
                    "rests on still resolves"
                )
        if declared:
            sources[key] = declared
    graph = DerivationGraph(sources=sources)
    _refuse_cycles(graph)
    return graph


def _published_keys(values: list[dict[str, object]]) -> set[ValueKey]:
    """The identity of every published value, refusing two rows that claim the same one."""
    published: set[ValueKey] = set()
    for value in values:
        key = published_key(value)
        if key in published:
            raise ValueError(
                f"{key.label()}: two published values claim this identity, so a declaration naming "
                "it as a source would resolve against whichever of them the walk reached first"
            )
        published.add(key)
    return published


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


def _refuse_cycles(graph: DerivationGraph) -> None:
    """Refuse a declaration set where a value transitively derives from itself.

    A cycle is not a walk that loops -- the walks here are all guarded -- it is a design claiming a
    number is computed out of a number computed out of it, where neither end is a measurement any
    restatement could start from.
    """
    settled: set[ValueKey] = set()
    for start in graph.sources:
        if start in settled:
            continue
        # The stack IS the chain being explored, so a source already on it closes a loop and the
        # stack itself is the readable name for that loop.
        stack: list[ValueKey] = [start]
        on_stack: set[ValueKey] = {start}
        while stack:
            descended = False
            for source in graph.sources_of(stack[-1]):
                if source in on_stack:
                    named = " -> ".join(node.label() for node in [*stack, source])
                    raise ValueError(
                        f"{start.label()}: the `{DERIVED_FROM}` declarations form a cycle ({named}), "
                        "so no value in it rests on a measurement anything could restate"
                    )
                if source not in settled:
                    stack.append(source)
                    on_stack.add(source)
                    descended = True
                    break
            if not descended:
                node = stack.pop()
                on_stack.discard(node)
                settled.add(node)
