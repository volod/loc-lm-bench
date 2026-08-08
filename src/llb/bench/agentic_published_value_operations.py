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
whether it also reads the figure the value's own aggregate measured. Those are checked against the
declaration before a number is read, and an operation the registry does not carry is refused -- a
design that named arithmetic nothing implements publishes a number no reader can reproduce.

What deliberately stays out is an expression language. A design picks an operation by NAME; adding a
kind of arithmetic is a registered function with a test, not a formula in a JSON file that every
reader would have to evaluate the same way.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from llb.bench.agentic_memory_crossover_restatement_reading import FORM_INTERPOLATED
from llb.bench.agentic_memory_fold_step_ladder import compaction_trigger_chars

# The field a published value names its arithmetic in, beside the sources that arithmetic is over.
OPERATION = "operation"

# What a value's own design row states for an operation to read, beside its declared sources.
STATED_COMPACT_SHARE = "compact_share"

# Intermediates an operation names so a reader can report them without recomputing them.
TERM_TRIGGER_CHARS = "trigger_chars"


@dataclass(frozen=True, slots=True)
class DerivationInputs:
    """Everything one re-derivation is computed over, gathered by the caller that has it.

    Three kinds, because a derived value rests on three different things: the values of the sources
    it DECLARES (in declared order, so an operation over two of one form is still unambiguous), the
    fields its own design row states, and -- for a value whose own aggregate measured something, like
    a ratio published against the cap peak its own study measured -- that measurement.
    """

    sources: tuple[float, ...]
    stated: Mapping[str, float]
    measured: float | None = None


@dataclass(frozen=True, slots=True)
class DerivedValue:
    """One re-derived number, and the intermediates the operation is willing to name.

    The terms exist so a restated row can report the trigger it divided by the peak without a second
    module re-deriving that trigger -- which is exactly the duplication the operation removes.
    """

    value: float
    terms: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class DerivationOperation:
    """One registered re-derivation: what it is computed over, and the pure function that does it."""

    name: str
    source_forms: tuple[str, ...]
    stated_fields: tuple[str, ...] = ()
    reads_own_measurement: bool = False
    compute: Callable[[DerivationInputs], DerivedValue]

    def apply(
        self,
        sources: tuple[float, ...],
        stated: Mapping[str, float],
        *,
        measured: float | None = None,
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
        return self.compute(DerivationInputs(sources=sources, stated=stated, measured=measured))


def _trigger_over_own_cap_peak(inputs: DerivationInputs) -> DerivedValue:
    """A compaction trigger, derived from a declared guard, over the value's own cap peak.

    The trigger is the runtime's own arithmetic rather than a formula restated here, so a published
    edge, a restated row, and the runtime that will route on the number are never three ways of
    computing one quotient.
    """
    (guard,) = inputs.sources
    trigger = compaction_trigger_chars(int(guard), inputs.stated[STATED_COMPACT_SHARE])
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
        stated_fields=(STATED_COMPACT_SHARE,),
        reads_own_measurement=True,
        compute=_trigger_over_own_cap_peak,
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
