"""Judge calibration, scoring, status, and worksheet contracts."""

from typing_extensions import NotRequired, TypedDict


class CalibrationResult(TypedDict):
    rho: float
    ci_low: float
    ci_high: float
    n: int
    threshold: float
    trusted: bool


class JudgeInputRecord(TypedDict):
    question: str
    answer: str
    contexts: list[str]


class JudgeScore(TypedDict):
    faithfulness: float
    answer_relevancy: float


class JudgeDiagnostics(TypedDict):
    """Counts and reasons that distinguish candidate and local judge failures."""

    n: int
    n_ok: int
    n_zero: int
    reasons: dict[str, int]


class JudgeStatus(TypedDict):
    calibration_rho: float | None
    threshold: float
    trusted: bool
    provider: NotRequired[str]
    model: NotRequired[str]
    base_url: NotRequired[str | None]
    prompt_language: NotRequired[str]
    metrics: NotRequired[list[str]]
    diagnostics: NotRequired[JudgeDiagnostics | None]


class WorksheetItem(TypedDict):
    id: str
    split: str
    provenance: NotRequired[str]
    question: str
    reference_answer: str
