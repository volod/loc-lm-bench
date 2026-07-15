"""Focused text analysis labels implementation."""

from dataclasses import dataclass, field
from typing import Any
from llb.core.contracts.benchmarks import PlantedLabelRecord

KEY_FACT = "key_fact"  # a planted atomic fact the answer must recover

ENTITY = "entity"  # a named entity present in the doc

TOPIC = "topic"  # a planted topic/theme of the doc

TREND = "trend"  # a planted directional trend (attrs: subject, direction)

RISK = "risk"  # a planted risk/problem

DECISION = "decision"  # a planted decision/action item

CONTRADICTION = "contradiction"  # a planted internal contradiction (attrs: span ids)

NARRATIVE = "narrative"  # the doc's overarching narrative (free-form quality -> judged)

INSIGHT = "insight"  # a non-stated inference (free-form quality -> judged)

LONG_DOC = "long_doc"  # long-doc comprehension answer (map-reduce; correctness/judge)

OBJECTIVE_KINDS = frozenset({KEY_FACT, ENTITY, TOPIC, TREND, RISK, DECISION, CONTRADICTION})

JUDGED_KINDS = frozenset({NARRATIVE, INSIGHT, LONG_DOC})

ALL_KINDS = OBJECTIVE_KINDS | JUDGED_KINDS

TAU_FULL = 0.85

TAU_PARTIAL = 0.70

PARTIAL_CREDIT = 0.5

DIRECTION_CONFLICT_CREDIT = 0.0

DIRECTION_UP = "up"

DIRECTION_DOWN = "down"

DIRECTION_FLAT = "flat"

_DIRECTION_STEMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        DIRECTION_UP,
        (
            "зрос",
            "зріс",
            "виріс",
            "збільш",
            "підвищ",
            "поліпш",
            "increase",
            "rose",
            "grow",
            "rise",
            "rising",
            "higher",
            "uptrend",
        ),
    ),
    (
        DIRECTION_DOWN,
        (
            "зниж",
            "зменш",
            "спад",
            "погірш",
            "пад",
            "впа",
            "скороч",
            "decrease",
            "decline",
            "fall",
            "fell",
            "drop",
            "lower",
            "down",
            "shrink",
        ),
    ),
    (DIRECTION_FLAT, ("стаб", "незмін", "сталий", "flat", "stable", "unchanged", "plateau")),
)

_PUNCT_STRIP = " \t\r\n.,;:!?\"'`«»“”()[]{}-–—"


def normalize_surface(text: str) -> str:
    """Casefold, collapse whitespace, and strip surrounding punctuation -- the canonical form
    used for exact label-ID surface matching (deliberately NOT lemmatization, per text-analysis sign-off)."""
    return " ".join(text.casefold().split()).strip(_PUNCT_STRIP)


def direction_of(text: str) -> str | None:
    """Infer a trend DIRECTION (up | down | flat) from free text via the UA/EN stem lexicon, or
    None when no direction word is present. Used only for direction-aware `trend` credit."""
    low = text.casefold()
    for direction, stems in _DIRECTION_STEMS:
        if any(stem in low for stem in stems):
            return direction
    return None


@dataclass(frozen=True)
class PlantedLabel:
    """A planted ground-truth label for one text-analysis sub-task (see `PlantedLabelRecord`)."""

    label_id: str
    kind: str
    value: str
    aliases: tuple[str, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)
    scoring: str = ""

    @property
    def surfaces(self) -> tuple[str, ...]:
        """All accepted surface forms (value + aliases)."""
        return (self.value, *self.aliases)

    @property
    def is_objective(self) -> bool:
        if self.scoring:
            return self.scoring == "objective"
        return self.kind in OBJECTIVE_KINDS

    @classmethod
    def from_record(cls, record: PlantedLabelRecord) -> "PlantedLabel":
        kind = record["kind"]
        if kind not in ALL_KINDS:
            raise ValueError(f"unknown text-analysis label kind: {kind!r}")
        return cls(
            label_id=record["label_id"],
            kind=kind,
            value=record["value"],
            aliases=tuple(record.get("aliases", []) or ()),
            attrs=dict(record.get("attrs", {}) or {}),
            scoring=record.get("scoring", ""),
        )


def load_planted_labels(records: list[PlantedLabelRecord]) -> list[PlantedLabel]:
    """Build `PlantedLabel`s from the planter's emitted records, rejecting unknown kinds."""
    return [PlantedLabel.from_record(r) for r in records]
