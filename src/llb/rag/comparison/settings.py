"""Alignment checks and resolved statistical settings for retrieval comparisons."""

from collections.abc import Sequence
from dataclasses import dataclass

from llb.rag.comparison.models import CompareItem, Retriever


@dataclass(frozen=True, slots=True)
class ComparisonSettings:
    baseline: str | None
    eligible: list[str]
    resamples: int
    confidence: float
    seed: int

    @classmethod
    def resolve(
        cls,
        stores: dict[str, Retriever],
        items: list[CompareItem],
        *,
        slice_labels: list[str | None] | None,
        item_ids: Sequence[str] | None,
        baseline: str | None,
        eligible_lanes: Sequence[str] | None,
        resamples: int | None,
        confidence: float | None,
        seed: int | None,
    ) -> "ComparisonSettings":
        from llb.rag.fusion_evidence.stats import (
            DEFAULT_CONFIDENCE,
            DEFAULT_RESAMPLES,
            DEFAULT_SEED,
        )

        _check_alignment(items, slice_labels, item_ids)
        resolved_baseline, eligible = _resolved_lanes(stores, baseline, eligible_lanes)
        return cls(
            baseline=resolved_baseline,
            eligible=eligible,
            resamples=DEFAULT_RESAMPLES if resamples is None else resamples,
            confidence=DEFAULT_CONFIDENCE if confidence is None else confidence,
            seed=DEFAULT_SEED if seed is None else seed,
        )


def _check_alignment(
    items: list[CompareItem],
    slice_labels: list[str | None] | None,
    item_ids: Sequence[str] | None,
) -> None:
    if slice_labels is not None and len(slice_labels) != len(items):
        raise ValueError("retrieval slice labels must align one-to-one with items")
    if item_ids is not None and len(item_ids) != len(items):
        raise ValueError("retrieval item ids must align one-to-one with items")


def _resolved_lanes(
    stores: dict[str, Retriever],
    baseline: str | None,
    eligible_lanes: Sequence[str] | None,
) -> tuple[str | None, list[str]]:
    if baseline is not None and baseline not in stores:
        raise ValueError(f"retrieval baseline lane `{baseline}` was not scored")
    resolved = baseline if baseline is not None else next(iter(stores), None)
    eligible = list(eligible_lanes) if eligible_lanes is not None else list(stores)
    unknown = [lane for lane in eligible if lane not in stores]
    if unknown:
        raise ValueError(f"unknown retrieval verdict lane(s): {', '.join(unknown)}")
    if resolved is not None and resolved not in eligible:
        eligible.insert(0, resolved)
    return resolved, eligible
