"""The vocabulary a coupling states its band conditions in -- one interval type and one geometry.

A geometry separates the compound policy-change reading from the per-field union when three things
hold at one fold step, and they are the SAME three statements for every pair of moved fields:

  - the first moved field, audited alone, changes no prompt (else the union already reports it);
  - the second moved field, audited alone, changes no prompt (same reason);
  - the two moved together DO change a prompt (else there is nothing to separate).

What differs per pair is the arithmetic each statement turns into, so each statement is expressed
here as a HALF-OPEN guard interval `[low, high)` with `None` for an unbounded edge, and the solver
(`agentic_policy_change_interaction_band`) only has to intersect them. `BandCondition.impossible` is
the interesting case rather than an error case: it is how a pair states that no geometry can
separate it, in the same vocabulary a solvable pair states its bounds in.

The conditions themselves live in `agentic_policy_change_interaction_conditions`, and which pair
uses which set in `agentic_policy_change_interaction_couplings`.
"""

from dataclasses import dataclass
from typing import Any, Callable

from llb.bench.agentic.context_policy import (
    POLICY_COMPACT,
    SUMMARY_INPUT_CAP_TRIGGER,
    SUMMARY_INPUT_CAP_WINDOW,
    ContextPolicy,
)
from llb.bench.agentic_policy_change_audit import PolicyChange

# The auditable `ContextPolicy` fields, by the name the audit and the pins use for them.
FIELD_CAP = "observation_cap_chars"
FIELD_HEAD = "observation_head_share"
FIELD_KEEP_LAST = "keep_last_n"
FIELD_SHARE = "compact_share"
FIELD_KEEP_RECENT = "compact_keep_recent"
FIELD_BOUND = "summary_input_cap"

# The only bound direction that can separate: a baseline that elides nothing, and a candidate whose
# cap rides the moved share down onto the folded transcript.
SEPARATING_BOUNDS = (SUMMARY_INPUT_CAP_WINDOW, SUMMARY_INPUT_CAP_TRIGGER)
# The fields the two audited policies (`observation_cap`, `compact`) are parameterized by. A field
# outside this set cannot move a prompt in either arm whatever it is set to.
AUDITED_POLICY_FIELDS = frozenset(
    {FIELD_CAP, FIELD_HEAD, FIELD_SHARE, FIELD_KEEP_RECENT, FIELD_BOUND}
)


@dataclass(frozen=True, slots=True)
class BandCondition:
    """One statement a separating guard must satisfy, as the interval of guards that satisfy it."""

    name: str
    why: str
    low: int | None = None
    high: int | None = None

    @classmethod
    def always(cls, name: str, why: str) -> "BandCondition":
        """A condition every guard satisfies -- it constrains nothing and blocks nothing."""
        return cls(name=name, why=why)

    @classmethod
    def impossible(cls, name: str, why: str) -> "BandCondition":
        """A condition no guard satisfies, carrying the reason in `why`."""
        return cls(name=name, why=why, low=0, high=0)

    @classmethod
    def satisfied_when(cls, holds: bool, name: str, why: str) -> "BandCondition":
        """The same statement either way -- `why` states the demand, the interval answers it."""
        return cls.always(name, why) if holds else cls.impossible(name, why)

    @property
    def is_empty(self) -> bool:
        return self.low is not None and self.high is not None and self.low >= self.high

    def describe(self) -> str:
        edge = "impossible" if self.is_empty else f"[{self.low}, {self.high})"
        return f"{self.name} {edge}: {self.why}"


@dataclass(frozen=True, slots=True)
class StepGeometry:
    """One fold step of one task world, and the change the solver is asking about there."""

    change: PolicyChange
    step: int
    # The per-step `observation_cap` prompt sizes under perfect play, at the cell's own geometry.
    sequence: list[int]
    # The task world as `cap_prompt_sequence` / `compact_fold_input_probe` take it.
    geometry: dict[str, Any]
    # The compact share the cell holds -- the change's BASELINE where the change moves it.
    share: float

    def moved(self, field: str) -> tuple[Any, Any]:
        """The `(baseline, candidate)` values this change gives one of its moved fields."""
        return self.change.baseline[field], self.change.candidate[field]


@dataclass(frozen=True, slots=True)
class StepConditions:
    """What a coupling says about one fold step: the conditions, and the line that reads them."""

    detail: str
    conditions: tuple[BandCondition, ...]


ConditionsFn = Callable[[StepGeometry], StepConditions]


def shipped_policy_value(field: str) -> Any:
    """What the shipped `ContextPolicy` gives a field the replay is not told about."""
    return getattr(ContextPolicy(name=POLICY_COMPACT), field)
