"""Types and constants for the verified category composite."""

from dataclasses import dataclass

from llb.scoring.aggregate import (
    TIER_AGENTIC,
    TIER_SECURITY,
    TIER_STRUCTURED,
    TIER_SUMMARIZATION,
    TIER_TEXT_ANALYSIS,
    TIER_TOOLING,
)
from llb.scoring.leaderboard import ModelResult

CATEGORY_COMPOSITE_RAW_WEIGHTS: dict[str, float] = {
    TIER_TEXT_ANALYSIS: 20.0,
    TIER_SUMMARIZATION: 10.0,
    TIER_STRUCTURED: 10.0,
    TIER_SECURITY: 10.0,
    TIER_AGENTIC: 10.0,
    TIER_TOOLING: 5.0,
}
CATEGORY_COMPOSITE_REQUIRED_TIERS: tuple[str, ...] = tuple(CATEGORY_COMPOSITE_RAW_WEIGHTS)


@dataclass(frozen=True)
class CompositeComponent:
    """One category result plus the data-gate metadata needed before headline use."""

    result: ModelResult
    data_verified: bool = False
    verification_ref: str | None = None
    verification_error: str | None = None


@dataclass(frozen=True)
class CompositeIssue:
    """Why a model is blocked from the composite headline."""

    model: str
    reason: str
    tier: str | None = None
