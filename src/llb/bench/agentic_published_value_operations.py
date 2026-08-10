"""How a published value is COMPUTED out of what it declares, named by the design and applied by all.

A design already states WHAT a derived number rests on (`derived_from`, in
`agentic_published_value_derivation`), which is enough to tell a cause from its consequence. It is
not enough to RE-DERIVE anything: the arithmetic stayed in the readers, where a trigger ratio was
`compaction_trigger_chars(guard, share)` over the value's own cap peak in two separate modules that
agreed only because both were written to agree. A second design whose derived number is a difference,
a normalization, or a two-input quotient would have landed in a resolver edit for the same reason.

So the design names its arithmetic too:

    "operation": "<a name this registry carries>"

and every reader calls the SAME registered function over the SAME declared inputs. The registry is a
table of pure functions, in the shape of `PUBLISHED_VALUE_DESIGNS`: an operation states how many
sources of which FORM it is computed over, which of the value's own stated fields it reads, and
whether it also reads the figure the value's own aggregate measured. It also names any shipped
context-policy fields its arithmetic reads. Those are supplied from `ContextPolicy`, not copied out
of a design row, so a policy-pin drift can name the published values whose arithmetic moves. All are
checked against the function, and an operation the registry does not carry is refused -- a design
that named arithmetic nothing implements publishes a number no reader can reproduce.

The declaration is checked in BOTH directions. Against the design it is `_check_operands` next door;
against the FUNCTION beside it, it is the registry self-check in
`agentic_published_value_operation_audit`, which calls every entry here through inputs answering only
what it declared. That is why an entry also carries `probes`: an operation the self-check cannot call
is an operation whose declaration nothing checks against its body. It is a SET of points rather than
one point because a body reads its declaration along a PATH -- a source read only when a share is
partial is unread at a point with a whole one -- so which branches the declaration is certified on is
the operation's own declaration too, and adding a point is how an author states one.

What deliberately stays out is an expression language. A design picks an operation by NAME; adding a
kind of arithmetic is a registered function with a test, not a formula in a JSON file that every
reader would have to evaluate the same way.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from llb.bench.agentic.context_policy import ContextPolicy
from llb.bench.agentic_memory_crossover_restatement_reading import FORM_INTERPOLATED
from llb.bench.agentic_memory_fold_step_ladder import compaction_trigger_chars

# The field a published value names its arithmetic in, beside the sources that arithmetic is over.
OPERATION = "operation"

# The shipped policy field the trigger arithmetic reads.
POLICY_COMPACT_SHARE = "compact_share"

# Intermediates an operation names so a reader can report them without recomputing them.
TERM_TRIGGER_CHARS = "trigger_chars"

# The points the registry self-check calls the trigger ratio at. Synthetic and round: the probe set
# exists to record WHICH declared inputs the arithmetic reaches for, not to re-check any published
# number. Two points, differing in every declared input, so a read the quotient performs only at a
# whole compact share or only at a guard the trigger truncates differently is still recorded.
PROBE_GUARD_CHARS = 1024.0
PROBE_COMPACT_SHARE = 0.5
PROBE_CAP_PEAK_CHARS = 512.0
PROBE_WIDE_GUARD_CHARS = 4096.0
PROBE_PARTIAL_COMPACT_SHARE = 0.25
PROBE_WIDE_CAP_PEAK_CHARS = 2048.0


@dataclass(frozen=True, slots=True)
class DerivationInputs:
    """Everything one re-derivation is computed over, gathered by the caller that has it.

    Four kinds, because a derived value can rest on the values of its declared sources, fields its
    own design row states, the figure its own aggregate measured, and shipped context-policy fields.
    The last are a separate input so a policy change can invalidate arithmetic even when no measured
    row changes.
    """

    sources: tuple[float, ...]
    stated: Mapping[str, float]
    measured: float | None = None
    policy: ContextPolicy = field(default_factory=ContextPolicy)


@dataclass(frozen=True, slots=True)
class DerivedValue:
    """One re-derived number, and the intermediates the operation is willing to name.

    The terms exist so a restated row can report the trigger it divided by the peak without a second
    module re-deriving that trigger -- which is exactly the duplication the operation removes.
    """

    value: float
    terms: Mapping[str, float] = field(default_factory=dict)


def probe_point_named(position: int) -> str:
    """How one point of a probe set is named in a refusal, positionally -- two may look alike."""
    return f"probe point {position + 1}"


@dataclass(frozen=True, slots=True, kw_only=True)
class DerivationOperation:
    """One registered re-derivation: what it is computed over, and the pure function that does it."""

    name: str
    source_forms: tuple[str, ...]
    stated_fields: tuple[str, ...] = ()
    reads_own_measurement: bool = False
    policy_fields: tuple[str, ...] = ()
    compute: Callable[[DerivationInputs], DerivedValue]
    probes: tuple[DerivationInputs, ...]

    def __post_init__(self) -> None:
        """Refuse a probe set that cannot check the declaration it is meant to exercise.

        The set is a required field rather than an optional convenience: it is how the registry
        self-check (`agentic_published_value_operation_audit`) calls the arithmetic through inputs
        that answer only what it declared, so an operation registered without one would be arithmetic
        nothing could check the declaration of -- the exact gap the self-check exists to close.
        """
        if not self.probes:
            raise ValueError(
                f"the `{self.name}` operation declares no probe point, so nothing can call it "
                "through its own declaration and nothing checks what it is computed over"
            )
        if len(set(self.policy_fields)) != len(self.policy_fields):
            raise ValueError(
                f"the `{self.name}` operation repeats a shipped policy field in "
                f"{self.policy_fields!r}; a dependency is declared once"
            )
        unknown_policy_fields = tuple(
            name
            for name in self.policy_fields
            if name == "name" or name not in ContextPolicy.__dataclass_fields__
        )
        if unknown_policy_fields:
            raise ValueError(
                f"the `{self.name}` operation declares unknown shipped policy field(s) "
                f"{unknown_policy_fields!r}"
            )
        for position, probe in enumerate(self.probes):
            self._check_probe_answers_declaration(position, probe)
        self._check_probes_differ()

    def _check_probe_answers_declaration(self, position: int, probe: DerivationInputs) -> None:
        """Refuse one point that answers more or less than the declaration it is checked against."""
        for field_name, declared, offered in (
            ("source_forms", len(self.source_forms), len(probe.sources)),
            ("stated_fields", sorted(self.stated_fields), sorted(probe.stated)),
            ("reads_own_measurement", self.reads_own_measurement, probe.measured is not None),
        ):
            if declared != offered:
                raise ValueError(
                    f"the `{self.name}` operation declares {field_name} {declared!r} while the "
                    f"{probe_point_named(position)} it is checked at offers {offered!r}; every probe "
                    "point must answer exactly what the operation declares, because answering more "
                    "is what would hide a read the declaration does not carry"
                )

    def _declared_point(self, probe: DerivationInputs) -> tuple[object, ...]:
        """Everything about one point the operation can actually reach, in declared order."""
        return (
            probe.sources,
            tuple(probe.stated[name] for name in self.stated_fields),
            probe.measured if self.reads_own_measurement else None,
            tuple(getattr(probe.policy, name) for name in self.policy_fields),
        )

    def _check_probes_differ(self) -> None:
        """Refuse a set whose points cannot exercise different paths through the body.

        Two points equal in every input the operation declares hand the body the same numbers, so the
        second takes whatever path the first took and the set certifies exactly what one point does.
        Refused rather than tolerated so that adding a point is a claim about a BRANCH -- which is the
        whole reason the probe is a set -- and not a line an author adds to look thorough.
        """
        seen: dict[tuple[object, ...], int] = {}
        for position, probe in enumerate(self.probes):
            first = seen.setdefault(self._declared_point(probe), position)
            if first != position:
                raise ValueError(
                    f"the `{self.name}` operation declares {probe_point_named(position)} equal to "
                    f"{probe_point_named(first)} in every input it declares, so it takes the same "
                    "path through the body and certifies nothing the earlier point did not; a probe "
                    "SET states which branches the declaration is checked on, and a repeated point "
                    "states no branch"
                )

    def apply(
        self,
        sources: tuple[float, ...],
        stated: Mapping[str, float],
        *,
        measured: float | None = None,
        policy: ContextPolicy | None = None,
        where: str,
    ) -> DerivedValue:
        """Run the operation, refusing inputs that are not the ones it declared it takes.

        Checked here rather than trusted in each `compute`, so a registered function is written as
        the arithmetic alone and every operation refuses a mis-shaped call the same way.
        """
        if len(sources) != len(self.source_forms):
            raise ValueError(
                f"{where}: the `{self.name}` operation is computed over {len(self.source_forms)} "
                f"declared source(s) ({', '.join(self.source_forms)}), got {len(sources)}"
            )
        missing = [name for name in self.stated_fields if name not in stated]
        if missing:
            raise ValueError(
                f"{where}: the `{self.name}` operation reads this value's own "
                f"{', '.join(missing)}, which was not supplied to it"
            )
        if self.reads_own_measurement and measured is None:
            raise ValueError(
                f"{where}: the `{self.name}` operation is computed against the figure this value's "
                "own aggregate measured, and none was supplied to it"
            )
        return self.compute(
            DerivationInputs(
                sources=sources,
                stated=stated,
                measured=measured,
                policy=policy if policy is not None else ContextPolicy(),
            )
        )


def _trigger_over_own_cap_peak(inputs: DerivationInputs) -> DerivedValue:
    """A compaction trigger, derived from a declared guard, over the value's own cap peak.

    The trigger is the runtime's own arithmetic rather than a formula restated here, so a published
    edge, a restated row, and the runtime that will route on the number are never three ways of
    computing one quotient.
    """
    (guard,) = inputs.sources
    trigger = compaction_trigger_chars(int(guard), inputs.policy.compact_share)
    peak = float(inputs.measured if inputs.measured is not None else 0.0)
    if peak <= 0.0:
        raise ValueError(
            f"a trigger ratio is a quotient over a cap peak, so the peak must be positive, got "
            f"{inputs.measured!r}"
        )
    return DerivedValue(value=trigger / peak, terms={TERM_TRIGGER_CHARS: float(trigger)})


OPERATION_TRIGGER_OVER_OWN_CAP_PEAK = "trigger_over_own_cap_peak"

# The registry. One entry per arithmetic a registered design may name; adding one is a pure function
# and its test, never a branch in a reader.
DERIVATION_OPERATIONS: dict[str, DerivationOperation] = {
    OPERATION_TRIGGER_OVER_OWN_CAP_PEAK: DerivationOperation(
        name=OPERATION_TRIGGER_OVER_OWN_CAP_PEAK,
        source_forms=(FORM_INTERPOLATED,),
        reads_own_measurement=True,
        policy_fields=(POLICY_COMPACT_SHARE,),
        compute=_trigger_over_own_cap_peak,
        probes=(
            DerivationInputs(
                sources=(PROBE_GUARD_CHARS,),
                stated={},
                measured=PROBE_CAP_PEAK_CHARS,
                policy=ContextPolicy(compact_share=PROBE_COMPACT_SHARE),
            ),
            DerivationInputs(
                sources=(PROBE_WIDE_GUARD_CHARS,),
                stated={},
                measured=PROBE_WIDE_CAP_PEAK_CHARS,
                policy=ContextPolicy(compact_share=PROBE_PARTIAL_COMPACT_SHARE),
            ),
        ),
    ),
}


def registered_operation(name: object, *, where: str) -> DerivationOperation:
    """The operation a design names, refusing one this registry does not carry.

    Refused rather than defaulted: a design naming arithmetic nothing implements publishes a number
    no reader can reproduce, and guessing which registered function it meant is how the two readers
    would silently disagree again.
    """
    operation = DERIVATION_OPERATIONS.get(name) if isinstance(name, str) else None
    if operation is None:
        raise ValueError(
            f"{where}: it declares the `{OPERATION}` {name!r}, which no registered re-derivation "
            f"carries, so nothing can re-derive its value -- the registry holds "
            f"{sorted(DERIVATION_OPERATIONS)}"
        )
    return operation
