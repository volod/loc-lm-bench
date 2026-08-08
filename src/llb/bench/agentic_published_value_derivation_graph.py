"""Every declared derivation edge of one design, and what each published value transitively rests on.

The declarations themselves are read one value at a time
(`agentic_published_value_derivation`); this is the other half of the question, which only the WHOLE
design can answer: are the edges checkable against what this design publishes, and what does a value
rest on once the chains are followed.

Both halves are refusals rather than lookups, for the same reason. A declaration naming a value the
design does not publish, one naming itself, or a cycle is not evidence that moved -- it is a design
that never said what its number rests on, and that message must not arrive underneath the
consequences it causes. What the graph then answers is transitive: a value two derivation steps above
a moved measurement names that measurement, not the derived figure in between, because the figure in
between is a consequence too.
"""

from dataclasses import dataclass, field

from llb.bench.agentic_published_value_derivation import (
    DERIVED_FROM,
    ValueKey,
    declared_derivation,
    published_key,
)


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


def derivation_graph(values: list[dict[str, object]]) -> DerivationGraph:
    """Read every declaration in one design, refusing any the design itself cannot support.

    Fail-fast rather than collected, for the reason shape refusals are: a declaration naming a value
    the design does not publish is not evidence that moved, it is a design that never said what its
    number rests on, and that message must not arrive underneath the consequences it causes. Reading
    each declaration WHOLE here is what subjects every design to the operation refusals too -- an
    unregistered arithmetic is refused by the same walk that refuses a dangling edge.
    """
    published = _published_keys(values)
    sources: dict[ValueKey, tuple[ValueKey, ...]] = {}
    for value in values:
        key = published_key(value)
        derivation = declared_derivation(value)
        declared = derivation.sources if derivation is not None else ()
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


class _DerivationCycle(Exception):
    """A declared source that closes the chain currently being explored."""

    def __init__(self, source: ValueKey) -> None:
        super().__init__(source.label())
        self.source = source


def _descend_target(
    graph: DerivationGraph, node: ValueKey, on_stack: set[ValueKey], settled: set[ValueKey]
) -> ValueKey | None:
    """The next source to descend into, or None when this node's sources are all settled.

    Raises when a source is already on the chain: that is the cycle, caught by the walk that owns
    the stack, because the stack is the readable name for it.
    """
    for source in graph.sources_of(node):
        if source in on_stack:
            raise _DerivationCycle(source)
        if source not in settled:
            return source
    return None


def _settle_below(graph: DerivationGraph, start: ValueKey, settled: set[ValueKey]) -> None:
    """Walk every chain below `start`, settling each node, and refuse one that closes on itself."""
    # The stack IS the chain being explored, so a source already on it closes a loop.
    stack: list[ValueKey] = [start]
    on_stack: set[ValueKey] = {start}
    while stack:
        try:
            target = _descend_target(graph, stack[-1], on_stack, settled)
        except _DerivationCycle as cycle:
            named = " -> ".join(node.label() for node in [*stack, cycle.source])
            raise ValueError(
                f"{start.label()}: the `{DERIVED_FROM}` declarations form a cycle ({named}), "
                "so no value in it rests on a measurement anything could restate"
            ) from None
        if target is None:
            node = stack.pop()
            on_stack.discard(node)
            settled.add(node)
        else:
            stack.append(target)
            on_stack.add(target)


def _refuse_cycles(graph: DerivationGraph) -> None:
    """Refuse a declaration set where a value transitively derives from itself.

    A cycle is not a walk that loops -- the walks here are all guarded -- it is a design claiming a
    number is computed out of a number computed out of it, where neither end is a measurement any
    restatement could start from.
    """
    settled: set[ValueKey] = set()
    for start in graph.sources:
        if start not in settled:
            _settle_below(graph, start, settled)
