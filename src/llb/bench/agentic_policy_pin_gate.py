"""The gate that stops a shipped context-policy constant from silently retiring published numbers.

Editing one constant in `llb.bench.agentic.context` is a one-character change that passes every
existing check green, and the published agentic cells measured under the old value keep being
published as if nothing happened. The policy-change audit can say exactly which cells a change
invalidates, but only when somebody remembers to ask -- and the person editing the constant is
precisely the person who does not know the question exists.

So this module asks for them. A committed fixture PINS each shipped constant to the value the
published evidence stands on; the gate compares the pins with the live `ContextPolicy` defaults and,
for every field that drifted, runs the audit and names the cells the drift invalidates. The audit is
a model-free replay, so a clean build pays nothing (no drift, no replay) and a drifted one pays under
a second per field.

Failing on ANY drift is the point, including a drift the audit clears: the pin is the record of what
the evidence was measured under, and a change that invalidates nothing costs one fixture line to
restate. What the gate refuses is the silent case -- a constant moving while the docs keep quoting
numbers measured under its old value.

The fixture also declares, per field, how it relates to the committed study designs (`agree`,
`restated`, `unstated`), and the gate verifies that claim against each design's `held_fixed`. A pin
that quietly disagrees with the studies it claims to match would defeat the whole mechanism.

`agentic_policy_pin_gate_report.py` renders what a failure says; this module decides what is true.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from llb.bench.agentic.context import ContextPolicy
from llb.bench.agentic_policy_change_audit import (
    AUDITABLE_FIELDS,
    POLICY_FIELD_TYPES,
    audit_policy_change,
)
from llb.bench.agentic_policy_change_audit_report import policy_change_summary

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
class PinDrift:
    """A shipped constant that no longer matches its pin, plus what the audit says it costs."""

    field: str
    pinned: Any
    shipped: Any
    summary: dict[str, object]

    @property
    def n_invalidated(self) -> int:
        return cast(int, self.summary["n_invalidated"])


@dataclass(frozen=True, slots=True)
class PinClaim:
    """A pin whose declared relation to the committed designs is not the one they support."""

    field: str
    declared: str
    supported: str
    stated: tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class PinCheck:
    """The gate's whole verdict: which constants drifted, and which pin claims are stale."""

    pins: PolicyPins
    shipped: dict[str, Any]
    drifted: tuple[PinDrift, ...]
    stale_claims: tuple[PinClaim, ...]

    @property
    def ok(self) -> bool:
        return not self.drifted and not self.stale_claims


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
) -> PinCheck:
    """Compare the pins with the shipped constants and audit whatever moved."""
    values = shipped_policy_values() if shipped is None else shipped
    drifted = tuple(
        _drift(pin, values[field], designs)
        for field, pin in pins.pins.items()
        if values[field] != pin.value
    )
    return PinCheck(
        pins=pins,
        shipped=values,
        drifted=drifted,
        stale_claims=tuple(filter(None, (_claim(pin, designs) for pin in pins.pins.values()))),
    )


def _drift(pin: PolicyPin, shipped: Any, designs: dict[str, dict[str, object]]) -> PinDrift:
    audits = audit_policy_change(designs, field=pin.field, baseline=pin.value, candidate=shipped)
    return PinDrift(
        field=pin.field,
        pinned=pin.value,
        shipped=shipped,
        summary=policy_change_summary(
            audits, field=pin.field, baseline=pin.value, candidate=shipped
        ),
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
