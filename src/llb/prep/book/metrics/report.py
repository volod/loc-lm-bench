"""The score report. Counts and rates stay in their own fields; nothing here is a headline score."""

from typing import Literal

from pydantic import Field, model_validator

from llb.core.contracts.data_prep.book_blocks import UNCERTAINTY_CLASSES
from llb.core.contracts.data_prep.corpus import DataPrepRow
from llb.prep.book.metrics.constants import (
    CONFUSION_CLASSES,
    STATUS_ABSENT,
    STATUS_NOT_APPLICABLE,
    STATUS_SCORED,
    STRUCTURE_FIELDS,
)

MetricStatus = Literal["scored", "absent", "not_applicable"]


class MetricValue(DataPrepRow):
    """One metric. ``absent`` means the lane cannot express it; it is not a zero."""

    status: MetricStatus
    value: float | None = None

    @model_validator(mode="after")
    def _value_follows_status(self) -> "MetricValue":
        if (self.status == STATUS_SCORED) != (self.value is not None):
            raise ValueError("only a scored metric carries a value")
        return self


def scored(value: float) -> MetricValue:
    """A computed value."""
    return MetricValue(status=STATUS_SCORED, value=value)


def absent() -> MetricValue:
    """The lane has no way to express this structure."""
    return MetricValue(status=STATUS_ABSENT)


def not_applicable() -> MetricValue:
    """The page has nothing for this metric to measure."""
    return MetricValue(status=STATUS_NOT_APPLICABLE)


def maybe(value: float | None) -> MetricValue:
    """Scored when a number exists, and not applicable when the comparison had nothing to rank."""
    return not_applicable() if value is None else scored(value)


class PageScore(DataPrepRow):
    """Every metric for one page of one lane."""

    lane: str
    pdf_page: int = Field(ge=1)
    page_class: str
    character_error_rate: MetricValue
    word_error_rate: MetricValue
    missing_words: int = Field(ge=0)
    duplicated_words: int = Field(ge=0)
    reference_chars: int = Field(ge=0)
    reference_words: int = Field(ge=0)
    reading_order_agreement: MetricValue
    dangerous_substitutions: dict[str, int]
    formula_fidelity: MetricValue
    formula_parse_valid: bool | None
    table_similarity: MetricValue
    figure_overlap: MetricValue
    caption_association: MetricValue
    printed_page_accuracy: MetricValue
    submitted: bool
    abstentions: dict[str, int]

    @model_validator(mode="after")
    def _every_class(self) -> "PageScore":
        if set(self.dangerous_substitutions) != set(CONFUSION_CLASSES):
            raise ValueError("every confusion class is reported")
        if set(self.abstentions) != set(UNCERTAINTY_CLASSES):
            raise ValueError("every abstention class is reported")
        return self


class ClassScore(DataPrepRow):
    """The pages of one class. Substitution counts are sums; the other metrics are means."""

    page_class: str
    pages: int = Field(ge=1)
    character_error_rate: MetricValue
    word_error_rate: MetricValue
    missing_words: int = Field(ge=0)
    duplicated_words: int = Field(ge=0)
    reference_chars: int = Field(ge=0)
    reading_order_agreement: MetricValue
    dangerous_substitutions: dict[str, int]
    dangerous_substitution_rates: dict[str, float | None]
    formula_fidelity: MetricValue
    table_similarity: MetricValue
    figure_overlap: MetricValue
    caption_association: MetricValue
    printed_page_accuracy: MetricValue
    missing_submissions: int = Field(ge=0)
    unscorable_fields: dict[str, int]
    abstentions: dict[str, int]

    @model_validator(mode="after")
    def _abstentions_cover_every_class(self) -> "ClassScore":
        if set(self.abstentions) != set(UNCERTAINTY_CLASSES):
            raise ValueError("every abstention class is reported")
        return self


class LaneScore(DataPrepRow):
    """One lane, its pages, and the per-class rollup. Lanes are not averaged together."""

    lane: str
    pages: list[PageScore]
    by_page_class: dict[str, ClassScore]
    missing_submissions: int = Field(ge=0)
    unscorable_fields: dict[str, int]
    abstentions: dict[str, int]

    @model_validator(mode="after")
    def _abstentions_cover_every_class(self) -> "LaneScore":
        if set(self.abstentions) != set(UNCERTAINTY_CLASSES):
            raise ValueError("every abstention class is reported")
        return self


class ScoreReport(DataPrepRow):
    """The score command's report. There is no field that folds the metrics into one number."""

    lanes: list[LaneScore]


def rollup(pages: list[PageScore]) -> dict[str, ClassScore]:
    """Mean the scored metrics and sum the substitution counts, one entry per page class."""
    grouped: dict[str, list[PageScore]] = {}
    for page in pages:
        grouped.setdefault(page.page_class, []).append(page)
    return {name: _class_score(name, grouped[name]) for name in sorted(grouped)}


def _class_score(name: str, pages: list[PageScore]) -> ClassScore:
    counts = {
        kind: sum(page.dangerous_substitutions[kind] for page in pages)
        for kind in CONFUSION_CLASSES
    }
    chars = sum(page.reference_chars for page in pages)
    rates = {kind: (counts[kind] / chars if chars else None) for kind in CONFUSION_CLASSES}
    missing, unscorable, abstentions = field_coverage(pages)
    return ClassScore(
        page_class=name,
        pages=len(pages),
        character_error_rate=_mean([page.character_error_rate for page in pages]),
        word_error_rate=_mean([page.word_error_rate for page in pages]),
        missing_words=sum(page.missing_words for page in pages),
        duplicated_words=sum(page.duplicated_words for page in pages),
        reference_chars=chars,
        reading_order_agreement=_mean([page.reading_order_agreement for page in pages]),
        dangerous_substitutions=counts,
        dangerous_substitution_rates=rates,
        formula_fidelity=_mean([page.formula_fidelity for page in pages]),
        table_similarity=_mean([page.table_similarity for page in pages]),
        figure_overlap=_mean([page.figure_overlap for page in pages]),
        caption_association=_mean([page.caption_association for page in pages]),
        printed_page_accuracy=_mean([page.printed_page_accuracy for page in pages]),
        missing_submissions=missing,
        unscorable_fields=unscorable,
        abstentions=abstentions,
    )


def field_coverage(pages: list[PageScore]) -> tuple[int, dict[str, int], dict[str, int]]:
    """Missing pages, absent structural fields, and abstention totals.

    An absent field is omitted when its count is zero. Every abstention class is present.
    """
    missing: int = sum(0 if page.submitted else 1 for page in pages)
    absent_fields: dict[str, int] = {}
    abstentions: dict[str, int] = dict.fromkeys(UNCERTAINTY_CLASSES, 0)
    for page in pages:
        for name in STRUCTURE_FIELDS:
            metric = getattr(page, name)
            if isinstance(metric, MetricValue) and metric.status == STATUS_ABSENT:
                absent_fields[name] = absent_fields.get(name, 0) + 1
        for name, count in page.abstentions.items():
            abstentions[name] += count
    return missing, absent_fields, abstentions


def _mean(values: list[MetricValue]) -> MetricValue:
    scored_values = [
        item.value for item in values if item.status == STATUS_SCORED and item.value is not None
    ]
    if scored_values:
        return scored(sum(scored_values) / len(scored_values))
    if any(item.status == STATUS_ABSENT for item in values):
        return absent()
    return not_applicable()
