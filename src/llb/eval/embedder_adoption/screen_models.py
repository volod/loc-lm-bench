"""Contracts and defaults for the per-model adoption screen."""

from typing_extensions import NotRequired, TypedDict

DEFAULT_SCREEN_SIZES = (10, 15, 20, 25, 30, 35)
DEFAULT_DRAWS = 120
DEFAULT_STUDY_RESAMPLES = 300
DEFAULT_TARGET_AGREEMENT = 0.9

DECISION_SCREEN_SUPPORTED = "screen_supported"
DECISION_FULL_SET_REQUIRED = "full_set_required"


class SizeAgreement(TypedDict):
    size: int
    agreement: float
    readings: dict[str, int]


class ModelScreen(TypedDict):
    model: str
    n: int
    full_reading: str
    recorded_reading: str
    reproduced: bool
    sizes: list[SizeAgreement]
    min_size: int | None


class ScreenVerdict(TypedDict):
    decision: str
    focus_cell: str
    target_agreement: float
    min_size: int | None
    full_n: int
    bundles_full_grid: int
    bundles_focus_cell: int
    reason: str


class ScreenReport(TypedDict):
    focus_cell: str
    baseline: str
    candidate: str
    sizes: list[int]
    draws: int
    resamples: int
    confidence: float
    seed: int
    models: list[ModelScreen]
    verdict: ScreenVerdict
    metadata: NotRequired[dict[str, object]]
