"""Resolve a design's published values, collect what did not resolve, and refuse once with all of it.

A design publishes several numbers, and a re-run moves as many of them as it moves. A validation that
raised at the first mismatch therefore handed the operator ONE name per run of the check -- restate
that number, re-run, meet the next one -- which is the same slow loop at the value level that the
registry walk in `agentic_published_value_registry` ends at the design level. This is that walk's
per-value twin, and it is study-agnostic for the same reason the resolver is: the next design to
publish resolvable values inherits the behavior rather than re-deriving it.

Two lists, not one, because a collecting walk has to say two different things. A value that did not
resolve is a number to RESTATE. A value the walk declines to judge is a check it cannot make until
the cause above it is restated -- typically a DERIVED figure whose source moved, where reporting the
consequence beside the cause names one moved measurement twice and sends the operator to restate a
figure nothing here can evaluate.

Which value is a consequence of which is not something this accumulator could know, and it is not
something a resolver should hardcode either -- a rule written for one study's ratio is silent about
the next design's derived number. The design DECLARES its derivation edges
(`llb.bench.agentic_published_value_derivation`), the accumulator carries that graph, and
`rests_on_unresolved` answers the one question a collecting walk has to ask of it: does this value
rest, at any distance, on something already named above. So the cause-versus-consequence rule is
study-agnostic here for the same reason the rest of the accumulator is.

What stays outside: SHAPE refusals. A design that never said where a number came from -- or that
declared it derives from a value the design does not publish -- has no values to collect, and its
defect is about the design rather than about the evidence, so callers refuse those before the first
value is read instead of burying them under the consequences they cause.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from llb.bench.agentic_published_value_derivation import DerivationGraph, ValueKey

T = TypeVar("T")


@dataclass(slots=True)
class CollectedRefusals:
    """Every published value one walk could not resolve, and every one it therefore cannot judge."""

    derivations: DerivationGraph = field(default_factory=DerivationGraph)
    unresolved: list[str] = field(default_factory=list)
    unjudged: list[str] = field(default_factory=list)
    unresolved_keys: set[ValueKey] = field(default_factory=set)

    def collect(self, resolve: Callable[[], T], *, key: ValueKey | None = None) -> T | None:
        """Run one value's resolution, keeping its refusal instead of letting it end the walk.

        `key` is what the value is known BY, so a later value that declares it derives from this one
        can be told apart from one that merely moved as well. A resolution with no key is collected
        all the same -- it just cannot be the named cause of anything.
        """
        try:
            return resolve()
        except ValueError as exc:
            self._record(str(exc), key)
            return None

    def unresolvable(self, reason: str, *, key: ValueKey | None = None) -> None:
        """Name a published value the evidence does not state, where nothing was read to find out."""
        self._record(reason, key)

    def not_judged(self, reason: str) -> None:
        """Name a check this walk declines to make, because a value it rests on is named above."""
        self.unjudged.append(reason)

    def rests_on_unresolved(self, key: ValueKey) -> bool:
        """Mark a value NOT JUDGED when anything it declares it is derived from did not resolve.

        Transitive, and reporting only the root of each chain: a value two derivation steps above a
        moved measurement rests on that measurement, and the derived figure in between is a
        consequence too, so naming it would send the operator to restate a number that will resolve
        itself the moment the real one does.
        """
        roots = self.derivations.unresolved_roots(key, self.unresolved_keys)
        if not roots:
            return False
        named = ", ".join(root.label() for root in roots)
        self.not_judged(
            f"{key.label()}: this value is DERIVED from {named}, which did not resolve above, so "
            "whether it moved as well cannot be told from here -- restate what it rests on and "
            "re-run this check"
        )
        return True

    def _record(self, reason: str, key: ValueKey | None) -> None:
        self.unresolved.append(reason)
        if key is not None:
            self.unresolved_keys.add(key)

    def refuse(self, *, total: int) -> None:
        """Refuse once, naming every unresolved value and everything that leaves unjudged.

        A list rather than a sentence, because the operator's next act is one design edit per line;
        counted against the total, because "how many of my published numbers moved" is what decides
        whether they are restating a value or withdrawing a study.
        """
        if not self.unresolved and not self.unjudged:
            return
        named = [f"  - {unresolved}" for unresolved in self.unresolved]
        named += [f"  - [not judged] {unjudged}" for unjudged in self.unjudged]
        raise ValueError(
            "\n".join(
                [
                    f"{len(self.unresolved)}/{total} published values do not resolve out of the "
                    "evidence the design cites, and every one of them is named here so that "
                    "restating them is one design edit rather than one per re-run of this check:",
                    *named,
                ]
            )
        )
