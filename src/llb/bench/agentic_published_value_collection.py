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

What stays outside: SHAPE refusals. A design that never said where a number came from has no values
to collect, and its defect is about the design rather than about the evidence, so callers refuse
those before the first value is read instead of burying them under the consequences they cause.
"""

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(slots=True)
class CollectedRefusals:
    """Every published value one walk could not resolve, and every one it therefore cannot judge."""

    unresolved: list[str] = field(default_factory=list)
    unjudged: list[str] = field(default_factory=list)

    def collect(self, resolve: Callable[[], float]) -> float | None:
        """Run one value's resolution, keeping its refusal instead of letting it end the walk."""
        try:
            return resolve()
        except ValueError as exc:
            self.unresolved.append(str(exc))
            return None

    def unresolvable(self, reason: str) -> None:
        """Name a published value the evidence does not state, where nothing was read to find out."""
        self.unresolved.append(reason)

    def not_judged(self, reason: str) -> None:
        """Name a check this walk declines to make, because a value it rests on is named above."""
        self.unjudged.append(reason)

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
