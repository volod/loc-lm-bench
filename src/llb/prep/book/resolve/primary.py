"""Which raw reading a page class treats as primary.

The run configuration names the reading. A block that does not carry it falls back to the first
reading in engine order, and the resolved block says so. Falling back does not vote among the
others.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from llb.core.contracts.data_prep.book_blocks import Block, RawReading
from llb.core.contracts.data_prep.book_resolve import ResolvedBlock


@dataclass(frozen=True)
class PrimaryChoice:
    """The reading the configuration selected for this page."""

    stage: str
    engine: str
    variant: str
    model: str | None
    page_class: str


def choice_from_params(params: Mapping[str, Any]) -> PrimaryChoice:
    """The flat choice `page_params` already resolved for this page."""
    model = params.get("model")
    return PrimaryChoice(
        stage=str(params["stage"]),
        engine=str(params["engine"]),
        variant=str(params["variant"]),
        model=None if model in (None, "") else str(model),
        page_class=str(params.get("page_class") or ""),
    )


def select_reading(block: Block, choice: PrimaryChoice) -> tuple[RawReading, bool]:
    """The configured reading, or `(fallback, True)` when the block does not carry it."""
    matches = [reading for reading in block.readings if _matches(reading, choice)]
    if matches:
        return matches[0], False
    ordered = sorted(
        block.readings, key=lambda reading: (reading.engine, reading.variant, reading.model or "")
    )
    return ordered[0], True


def resolved_reading(block: Block, resolved: ResolvedBlock | None) -> RawReading | None:
    """The ledger reading a resolved block was taken from: engine, variant, then model.

    Every consumer that reads structure or confidence off the primary reading asks here, so a
    block resolved from a fallback reading is read from that reading everywhere.
    """
    if not block.readings:
        return None
    if resolved is None:
        return block.readings[0]
    for reading in block.readings:
        if reading.engine != resolved.engine or reading.variant != resolved.variant:
            continue
        if resolved.model is None or reading.model == resolved.model:
            return reading
    return block.readings[0]


def _matches(reading: RawReading, choice: PrimaryChoice) -> bool:
    if reading.engine != choice.engine or reading.variant != choice.variant:
        return False
    return choice.model is None or reading.model == choice.model
