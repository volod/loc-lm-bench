"""What a proposed generation swap invalidates, listed before the swap rather than after it.

Adopting a generation is cheap to type and expensive to finish: every number in this repo that was
measured on the generation being replaced is now a statement about a model the roster no longer
runs, and until it is re-measured the docs read as current. Reconstructing that list by hand from
the board is the step that gets skipped, so this module builds it from the register and the evidence
surfaces -- given a family and the generation an operator proposes to adopt, it names every record
measured on the OUTGOING generation and says plainly when there are none.

It reports only. Nothing here re-runs a study, edits the roster, touches the board, or judges
whether the target generation is worth adopting -- the last one is a measurement the sweep takes,
and the first is the cost this report exists to make visible in advance.
"""

from dataclasses import dataclass
from pathlib import Path

from llb.backends.currency.generations import generation_key
from llb.backends.invalidation.identity import ModelIndex
from llb.backends.invalidation.surfaces import MeasuredRecord, SurfaceReading, read_evidence
from llb.backends.roster import Register

ADOPTION = "adoption"
ROLLBACK = "rollback"
UNORDERED = "unordered"

_NO_FAMILY = "no such family in the register: `{family}` -- known families: {known}"
_NO_CURRENT = "family `{family}` records no current generation, so nothing is being replaced"
_ALREADY_CARRIED = (
    "family `{family}` already carries generation `{target}` as its current one -- a swap to the "
    "generation already in place replaces nothing, so there is nothing to invalidate"
)


@dataclass(frozen=True)
class InvalidationReport:
    """One proposed swap, costed: the surfaces walked, what they hold, and what the swap voids.

    `readings` is carried beside the findings for the reason the currency report carries its
    registry readings: a walk that read nothing produces the same empty finding list as a walk that
    read everything and found nothing, and only the first of those is a reason to distrust the "no
    re-measurement needed" answer.
    """

    family_id: str
    family_label: str
    outgoing: str
    target: str
    direction: str
    readings: tuple[SurfaceReading, ...]
    invalidated: tuple[MeasuredRecord, ...]
    unresolved: tuple[MeasuredRecord, ...]

    @property
    def scanned(self) -> int:
        """Every record the surfaces held, whether or not this swap touches it."""
        return sum(len(reading.records) for reading in self.readings)

    @property
    def entries(self) -> tuple[str, ...]:
        """The roster entries a swap makes stale -- the re-measurement cost in models, not rows.

        A record count sizes the EDIT; this sizes the RUN. Twelve rows measured on two logical
        models are two `make measure-throughput` invocations and twelve restatements, and an
        operator sizing a swap needs both numbers rather than the larger one twice.
        """
        return tuple(
            sorted({record.resolved.model_name for record in self.invalidated if record.resolved})
        )

    @property
    def unread(self) -> tuple[SurfaceReading, ...]:
        """The surfaces that could not be read -- the reason a clean answer may be incomplete."""
        return tuple(reading for reading in self.readings if reading.error)

    def by_surface(self, surface: str) -> tuple[MeasuredRecord, ...]:
        """The invalidated records of one surface, in the order that surface reported them."""
        return tuple(record for record in self.invalidated if record.surface == surface)


def swap_direction(outgoing: str, target: str) -> str:
    """Whether the proposed target is newer than the generation it replaces.

    Reported, never enforced. A rollback to a `previous` generation invalidates exactly the same
    measurements a forward adoption does, so the direction changes what an operator is doing and
    not what it costs -- refusing one would be this module deciding an adoption question it says it
    does not decide.
    """
    outgoing_key, target_key = generation_key(outgoing), generation_key(target)
    if outgoing_key is None or target_key is None or outgoing_key == target_key:
        return UNORDERED
    return ADOPTION if target_key > outgoing_key else ROLLBACK


def report_invalidation(
    register: Register, family_id: str, target: str, *, root: Path
) -> InvalidationReport:
    """Cost one proposed swap against every registered evidence surface.

    `root` is the repo root the committed evidence and delivered docs are read from, so the report
    is reproducible on any checkout rather than depending on what this host's `DATA_DIR` still
    holds -- the evidence a swap invalidates is the evidence the repo publishes. It is resolved
    here so a caller may pass a relative one and still get doc locations relative to the repo.
    """
    family = register.family(family_id)
    if family is None:
        known = ", ".join(entry.id for entry in register.families) or "(none)"
        raise ValueError(_NO_FAMILY.format(family=family_id, known=known))
    carried = family.current
    if carried is None:
        raise ValueError(_NO_CURRENT.format(family=family_id))
    if target.strip() == carried.id:
        raise ValueError(_ALREADY_CARRIED.format(family=family_id, target=target.strip()))

    readings = read_evidence(Path(root).resolve(), ModelIndex(register))
    records = [record for reading in readings for record in reading.records]
    return InvalidationReport(
        family_id=family.id,
        family_label=family.label,
        outgoing=carried.id,
        target=target.strip(),
        direction=swap_direction(carried.id, target.strip()),
        readings=readings,
        invalidated=tuple(
            record
            for record in records
            if record.resolved is not None
            and record.resolved.family_id == family.id
            and record.resolved.generation_id == carried.id
        ),
        unresolved=tuple(record for record in records if record.resolved is None),
    )
