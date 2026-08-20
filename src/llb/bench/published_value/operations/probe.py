"""Inputs that answer only what an operation declared, and record the operation looking at them.

An operation's declaration (`source_forms`, `stated_fields`, `reads_own_measurement` in
`agentic_published_value_operations`) is checked against the DESIGN before a number is read. Checking
it against the FUNCTION beside it needs a different instrument, and this is it: the operation is
CALLED, through inputs built to answer exactly its declaration and nothing else, and every input
records the reach. Reading is then observed rather than argued from the source, which is what keeps
the check honest as arithmetic is added.

One call observes one PATH, so these inputs are built per point of the operation's declared probe SET
and the audit unions the reads across them. A branch no point takes can still hide an undeclared
READ, but the branch audit beside this recorder now reports that missed arc and its source line. The
set remains the operation's own statement of which paths its declaration holds on, instead of
whichever single point its author found convenient.

Two rules make the recording mean what it says. A reach past the declaration RAISES with the input
named -- an undeclared stated field, an undeclared measurement -- rather than producing the `KeyError`
or `TypeError` two frames deeper that a caller would otherwise have to interpret. And membership is
deliberately not a read: `apply` asks `name not in stated` for every declared field before computing
anything, so counting that would mark each declaration read and the over-declaration refusal would
never fire.

What the probe cannot see is stated at `agentic_published_value_operation_audit`, which reads the
recording: reaches that do not pass through these inputs at all.
"""

from collections.abc import Callable, Iterator, Mapping

from llb.bench.published_value.operations.registry import DerivationInputs, DerivationOperation


class UndeclaredRead(Exception):
    """An operation reached for an input its own declaration does not carry.

    Its own exception type rather than the `KeyError` / `IndexError` / `TypeError` the reach would
    otherwise raise, because those are also what a genuinely broken operation raises, and the two
    read very differently to whoever has to fix them.
    """

    def __init__(self, read: str) -> None:
        super().__init__(read)
        self.read = read


def source_read(position: int, form: str) -> str:
    """How one declared source is named in a refusal, positionally -- two may share a form."""
    return f"source {position + 1} (the declared {form})"


def stated_read(name: str) -> str:
    """How one of the value's own stated fields is named in a refusal."""
    return f"the value's own stated `{name}`"


# The figure a value's own aggregate measured, named the way `reads_own_measurement` declares it.
MEASURED_READ = "the figure the value's own aggregate measured"


class _ProbeNumber(float):
    """A number that records the operation looking at it, and refuses when it was never declared.

    A `float` subclass rather than a wrapper, so an operation is handed the thing it declared it
    takes and the recording happens through the arithmetic it performs on it -- `int(guard)`,
    `float(peak)`, a comparison. An undeclared input is handed as one of these too, so reaching for
    it raises with the input named instead of the `TypeError` a bare `None` would produce two frames
    deeper.
    """

    __slots__ = ("_declared", "_read", "_reads")

    _read: str
    _reads: set[str]
    _declared: bool

    def __new__(
        cls, value: float, *, read: str, reads: set[str], declared: bool = True
    ) -> "_ProbeNumber":
        number = super().__new__(cls, value)
        number._read = read
        number._reads = reads
        number._declared = declared
        return number

    def _record(self) -> None:
        if not self._declared:
            raise UndeclaredRead(self._read)
        self._reads.add(self._read)


# Every way an operation can look at a number it was handed. Comparisons count: an operation that
# only compares a source still needs it, and one that does not compare it did not need it declared.
_NUMBER_READS = (
    "__int__",
    "__float__",
    "__trunc__",
    "__round__",
    "__abs__",
    "__neg__",
    "__pos__",
    "__bool__",
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__truediv__",
    "__rtruediv__",
    "__floordiv__",
    "__rfloordiv__",
    "__mod__",
    "__rmod__",
    "__divmod__",
    "__rdivmod__",
    "__pow__",
    "__rpow__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__eq__",
    "__ne__",
)


def _recording(name: str) -> Callable[..., object]:
    """One `float` operation, recording the read before delegating to the arithmetic itself."""
    inherited = getattr(float, name)

    def recorded(self: _ProbeNumber, *args: object) -> object:
        self._record()
        return inherited(self, *args)

    recorded.__name__ = name
    return recorded


for _name in _NUMBER_READS:
    # Set after the class body rather than written out: thirty identical delegations would bury the
    # one line that matters, and `__hash__` survives because `__eq__` arrives by assignment.
    setattr(_ProbeNumber, _name, _recording(_name))


class _ProbeStated(Mapping[str, float]):
    """The stated fields an operation declared, answering nothing else and recording every read."""

    def __init__(self, values: Mapping[str, float], reads: set[str]) -> None:
        self._values = dict(values)
        self._reads = reads

    def __getitem__(self, name: str) -> float:
        if name not in self._values:
            # Named, not explained: the reader of the refusal is the audit, which supplies the why
            # once for every kind of undeclared reach rather than each probe wording its own.
            raise UndeclaredRead(stated_read(name))
        self._reads.add(stated_read(name))
        return self._values[name]

    def __contains__(self, name: object) -> bool:
        # Membership is NOT a read; see the module docstring for why that distinction carries the
        # whole over-declaration refusal.
        return name in self._values

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def declared_reads(operation: DerivationOperation) -> tuple[str, ...]:
    """Every input the operation DECLARES it is computed over, named as a refusal would name it."""
    stated = [stated_read(name) for name in operation.stated_fields]
    sources = [source_read(at, form) for at, form in enumerate(operation.source_forms)]
    return (*sources, *stated, *([MEASURED_READ] if operation.reads_own_measurement else []))


def probe_inputs(
    operation: DerivationOperation, probe: DerivationInputs, reads: set[str]
) -> DerivationInputs:
    """One point of the operation's probe set, answering only its declaration and recording reads.

    The point is passed in rather than taken off the operation, and `reads` is the caller's set, so
    the audit walks the whole set into ONE recording -- an input read at only one point is read.
    """
    return DerivationInputs(
        sources=tuple(
            _ProbeNumber(value, read=source_read(at, form), reads=reads)
            for at, (value, form) in enumerate(zip(probe.sources, operation.source_forms))
        ),
        stated=_ProbeStated({name: probe.stated[name] for name in operation.stated_fields}, reads),
        measured=_ProbeNumber(
            probe.measured if probe.measured is not None else 0.0,
            read=MEASURED_READ,
            reads=reads,
            declared=operation.reads_own_measurement,
        ),
        policy=probe.policy,
    )
