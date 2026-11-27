"""The comparison report. There is no field that folds the metrics into one number."""

from typing import Literal

from pydantic import Field, model_validator

from llb.core.contracts.data_prep.corpus import DataPrepRow
from llb.prep.book.compare.constants import KIND, SCHEMA_VERSION
from llb.prep.book.metrics.report import ClassScore

MarginStatus = Literal["gain", "loss", "inconclusive"]


class MarginReport(DataPrepRow):
    """A mean margin. Positive means the candidate beat the reference side of the pair."""

    point: float
    low: float
    high: float
    status: MarginStatus


class ClassLaneReport(DataPrepRow):
    """One lane on one class. The baseline's margin is null; it is the side others subtract."""

    lane: str
    engine: str
    metrics: ClassScore
    margin: MarginReport | None
    dangerous_rate: float | None
    baseline_dangerous_rate: float | None
    dangerous_margin: MarginReport | None
    blocked_dangerous: bool
    predraft_character_agreement: float | None


class ClassDecision(DataPrepRow):
    """The winner of one page class, and the lanes that were scored on it."""

    page_class: str
    pages: int = Field(ge=1)
    winner_lane: str
    configured_engine: str
    variant: str
    dewarp: str | None
    rationale: str
    lanes: list[ClassLaneReport]


class SubstitutionExample(DataPrepRow):
    """One confusion class. ``found`` is false when the pilot never produced it."""

    confusion_class: str
    found: bool
    lane: str | None = None
    pdf_page: int | None = None
    page_class: str | None = None
    reference: str | None = None
    hypothesis: str | None = None

    @model_validator(mode="after")
    def _found_examples_are_complete(self) -> "SubstitutionExample":
        if not self.found:
            return self
        missing = (
            self.lane is None
            or self.pdf_page is None
            or self.page_class is None
            or self.reference is None
            or self.hypothesis is None
        )
        if missing:
            raise ValueError("a found example names its lane, page, and both windows")
        return self


class NotCompared(DataPrepRow):
    """A candidate that was not scored. It wins no class."""

    lane: str
    engine: str
    reason: Literal["over-vram", "unavailable"]
    detail: str


class ComparisonReport(DataPrepRow):
    """The pilot decision. ``selection_set`` is true on every report this builder writes."""

    kind: Literal["book-ocr-pilot-comparison"] = KIND
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    language: str = Field(min_length=1)
    baseline_lane: str
    selection_set: Literal[True] = True
    negative: bool
    reopen_when: str | None
    classes: list[ClassDecision]
    won_no_class: list[str]
    variants_not_adopted: list[str]
    not_compared: list[NotCompared]
    examples: list[SubstitutionExample]
    projected_seconds: float | None
    projection_kind: Literal["selection-set"] = "selection-set"
    unprojected_classes: list[str]
    peak_vram_mb: int | None
    peak_vram_engine: str | None
    reference_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
