"""The gate that stops a shipped context-policy constant from silently retiring published numbers.

Editing one constant in `llb.bench.agentic.context` is a one-character change that passes every
existing check green, and the published agentic cells measured under the old value keep being
published as if nothing happened. The policy-change audit can say exactly which cells a change
invalidates, but only when somebody remembers to ask -- and the person editing the constant is
precisely the person who does not know the question exists.

So this module asks for them. A committed fixture PINS each shipped constant to the value the
published evidence stands on; the gate compares the pins with the live `ContextPolicy` defaults and,
for every field that drifted, runs the audit and names both the cells the drift invalidates and the
registered published values whose arithmetic declares that field. The audit is a model-free replay,
so a clean build pays nothing (no drift, no replay or registry walk) and a drifted one pays under a
second per field.

Failing on ANY drift is the point, including a drift the audit clears: the pin is the record of what
the evidence was measured under, and a change that invalidates nothing costs one fixture line to
restate. What the gate refuses is the silent case -- a constant moving while the docs keep quoting
numbers measured under its old value.

Constants that drift TOGETHER are audited together, as the one change the commit actually made: the
baseline arm replays the full pinned policy and the candidate arm the full shipped policy. Auditing
each drifted field on its own would replay "pinned cap + shipped keep" against "shipped cap + shipped
keep" -- two configurations the published cells were never measured under, which can name a first
divergent step neither build ever reaches. The gate passes the full pinned map into the replay so
fields the change does NOT move also come from the pins: a `restated` pin on a held field would
otherwise leave the design's stale `held_fixed` value on the baseline arm.

The fixture also declares, per field, how it relates to the committed study designs (`agree`,
`restated`, `unstated`), and the gate verifies that claim against each design's `held_fixed`. A pin
that quietly disagrees with the studies it claims to match would defeat the whole mechanism.

`agentic_policy_pin_gate_report.py` renders what a failure says; this module decides what is true.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from llb.bench.agentic.context_policy import ContextPolicy
from llb.bench.agentic_policy_change_audit import (
    AUDITABLE_FIELDS,
    POLICY_FIELD_TYPES,
    PolicyChange,
    audit_policy_change,
)
from llb.bench.agentic_policy_change_audit_report import policy_change_summary
from llb.bench.agentic_published_value_operation_scope import (
    PolicyAffectedPublishedValue,
    policy_affected_published_values,
)
from llb.core.paths import PROJECT_ROOT

# The committed pins, relative to the project root.
PINS_PATH = "samples/benchmarks/agentic_context_policy_pins.json"
PINS_SCHEMA_VERSION = 1

# How a pin relates to the `held_fixed` block of the committed studies it is audited against.
DESIGNS_AGREE = "agree"  # every design that states the field states the pinned value
DESIGNS_RESTATED = "restated"  # a design states another value, and the pin supersedes it
DESIGNS_UNSTATED = "unstated"  # no design states the field
DESIGN_STATES: tuple[str, ...] = (DESIGNS_AGREE, DESIGNS_RESTATED, DESIGNS_UNSTATED)


@dataclass(frozen=True, slots=True)
class PolicyPin:
    """One shipped constant, pinned to the value the published evidence was measured under."""

    field: str
    value: Any
    designs: str
    note: str


@dataclass(frozen=True, slots=True)
class PolicyPins:
    """The committed fixture: every pinned constant, and where the numbers they carry are published.

    The doc anchors are part of the fixture rather than of the gate because they are what a drift
    actually costs: the cells the audit names are quoted THERE, and a failure that does not say
    where leaves the reader to find them.
    """

    pins: dict[str, PolicyPin]
    published_in: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PinMove:
    """One shipped constant that no longer matches its pin."""

    field: str
    pinned: Any
    shipped: Any


@dataclass(frozen=True, slots=True)
class PinDrift:
    """Everything that moved, audited as ONE change, plus its cell and arithmetic scopes.

    Several constants moving in one commit is one change, not several: the published cells were
    measured under ALL the pinned values and the new build ships ALL the shipped ones, so those two
    whole policies are the only pair a re-run scope can honestly be computed between.
    """

    moves: tuple[PinMove, ...]
    summary: dict[str, object]
    affected_published_values: tuple[PolicyAffectedPublishedValue, ...]

    @property
    def n_invalidated(self) -> int:
        return cast(int, self.summary["n_invalidated"])

    @property
    def is_compound(self) -> bool:
        return len(self.moves) > 1


@dataclass(frozen=True, slots=True)
class PinClaim:
    """A pin whose declared relation to the committed designs is not the one they support."""

    field: str
    declared: str
    supported: str
    stated: tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class PinCheck:
    """The gate's whole verdict: what drifted (as one change), and which pin claims are stale."""

    pins: PolicyPins
    shipped: dict[str, Any]
    drift: PinDrift | None
    stale_claims: tuple[PinClaim, ...]

    @property
    def ok(self) -> bool:
        return self.drift is None and not self.stale_claims

    @property
    def moves(self) -> tuple[PinMove, ...]:
        return self.drift.moves if self.drift else ()


def load_policy_pins(path: Path | str) -> PolicyPins:
    """Load the committed pins, refusing any fixture the gate could not act on."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read context-policy pins {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != PINS_SCHEMA_VERSION:
        raise ValueError(f"context-policy pins schema_version must be {PINS_SCHEMA_VERSION}")
    pins = raw.get("pins")
    if not isinstance(pins, dict) or set(pins) != set(AUDITABLE_FIELDS):
        raise ValueError(
            "context-policy pins must state exactly one pin per auditable field "
            f"{AUDITABLE_FIELDS}; a new shipped constant is pinned here or it is unguarded"
        )
    published = raw.get("published_in")
    if not isinstance(published, list) or not all(
        isinstance(anchor, str) and "#" in anchor for anchor in published
    ):
        raise ValueError(
            "context-policy pins must name the doc sections that publish the pinned numbers, "
            "as `<path>#<anchor>` entries"
        )
    return PolicyPins(
        pins={
            field: _pin(field, cast(dict[str, object], entry))
            for field, entry in sorted(pins.items())
        },
        published_in=tuple(cast(list[str], published)),
    )


def shipped_policy_values() -> dict[str, Any]:
    """What the code ships today -- read off the dataclass so a default cannot dodge the gate."""
    shipped = ContextPolicy()
    return {field: getattr(shipped, field) for field in AUDITABLE_FIELDS}


def check_policy_pins(
    pins: PolicyPins,
    designs: dict[str, dict[str, object]],
    *,
    shipped: dict[str, Any] | None = None,
    design_root: Path = PROJECT_ROOT,
) -> PinCheck:
    """Compare the pins with the shipped constants and audit whatever moved, as one change."""
    values = shipped_policy_values() if shipped is None else shipped
    moves = tuple(
        PinMove(field=field, pinned=pin.value, shipped=values[field])
        for field, pin in pins.pins.items()
        if values[field] != pin.value
    )
    return PinCheck(
        pins=pins,
        shipped=values,
        drift=(
            _drift(
                moves,
                designs,
                design_root,
                pinned={field: pin.value for field, pin in pins.pins.items()},
            )
            if moves
            else None
        ),
        stale_claims=tuple(filter(None, (_claim(pin, designs) for pin in pins.pins.values()))),
    )


def _drift(
    moves: tuple[PinMove, ...],
    designs: dict[str, dict[str, object]],
    design_root: Path,
    pinned: Mapping[str, Any],
) -> PinDrift:
    change = PolicyChange(
        baseline={move.field: move.pinned for move in moves},
        candidate={move.field: move.shipped for move in moves},
    )
    audits = audit_policy_change(designs, change, pinned=pinned)
    return PinDrift(
        moves=moves,
        summary=policy_change_summary(audits, change),
        affected_published_values=policy_affected_published_values(design_root, change.fields),
    )


def _claim(pin: PolicyPin, designs: dict[str, dict[str, object]]) -> PinClaim | None:
    stated = design_held_values(designs, pin.field)
    supported = _supported_state(pin, stated)
    if supported == pin.designs:
        return None
    return PinClaim(
        field=pin.field, declared=pin.designs, supported=supported, stated=tuple(stated.items())
    )


def design_held_values(designs: dict[str, dict[str, object]], field: str) -> dict[str, Any]:
    """What each committed study's `held_fixed` states for one policy field, where it states it."""
    held = {kind: cast(dict[str, object], design["held_fixed"]) for kind, design in designs.items()}
    return {kind: values[field] for kind, values in held.items() if field in values}


def _supported_state(pin: PolicyPin, stated: dict[str, Any]) -> str:
    if not stated:
        return DESIGNS_UNSTATED
    return (
        DESIGNS_AGREE if all(value == pin.value for value in stated.values()) else DESIGNS_RESTATED
    )


def _pin(field: str, entry: dict[str, object]) -> PolicyPin:
    if field not in POLICY_FIELD_TYPES:
        raise ValueError(f"{field!r} is not an auditable policy field")
    designs = entry.get("designs")
    note = entry.get("note")
    if not isinstance(designs, str) or designs not in DESIGN_STATES:
        raise ValueError(
            f"pin {field!r} must declare designs from {DESIGN_STATES}, got {designs!r}"
        )
    if not isinstance(note, str) or not note.strip():
        raise ValueError(f"pin {field!r} must carry a note saying what the value rests on")
    return PolicyPin(
        field=field, value=_pin_value(field, entry.get("value")), designs=designs, note=note
    )


def _pin_value(field: str, value: object) -> Any:
    """The pinned value in the type its policy field takes -- never coerced from a wrong one.

    Coercion is right for a CLI string and wrong here: a pin typed `"800"` or `0.5` where an int
    belongs would compare unequal to the shipped constant forever, which is a permanently red gate
    rather than a caught drift.
    """
    expected = POLICY_FIELD_TYPES[field]
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise ValueError(f"pin {field!r} must state a {expected.__name__} value, got {value!r}")
    return value
