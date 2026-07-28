"""Declared RAG ranking policy and verbosity-sensitivity primitives."""

from collections.abc import Sequence
from math import sqrt

FORMAT_WEIGHT = 0.25
FACT_WEIGHT = 1.0 - FORMAT_WEIGHT
POLICY_NAME = "recall_75_precision_25"


def ranking_score(
    token_precision: float,
    token_recall: float,
    *,
    format_weight: float = FORMAT_WEIGHT,
) -> float:
    """Fact-first linear quality: recall plus a declared answer-format share."""
    if not 0.0 <= format_weight <= 1.0:
        raise ValueError("format_weight must be between 0 and 1")
    return (1.0 - format_weight) * token_recall + format_weight * token_precision


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson correlation, or None when the paired series has no variance."""
    if len(left) != len(right):
        raise ValueError("correlation series must have equal lengths")
    if len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = sqrt(sum((y - right_mean) ** 2 for y in right))
    denominator = left_scale * right_scale
    return numerator / denominator if denominator else None


def policy_description() -> str:
    return (
        f"{FACT_WEIGHT:.0%} token recall (fact coverage) + "
        f"{FORMAT_WEIGHT:.0%} token precision (answer format)"
    )
