"""Scorer-policy lane vocabulary shared by config, CLI, and the resolve seam."""

from typing import Literal

ScorerLane = Literal["human", "local", "frontier"]
SCORER_LANES: tuple[ScorerLane, ...] = ("human", "local", "frontier")
DEFAULT_SCORER_LANE: ScorerLane = "local"
