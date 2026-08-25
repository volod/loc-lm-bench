"""The ablation read one question type at a time (pure).

A pooled ablation answers "does retrieval pay for itself on this corpus" with ONE number over a
mixed item set, and that average is the thing an operator cannot act on: retrieval almost certainly
pays unevenly, because a factoid whose answer sits in one span is not the same retrieval problem as
a multi-hop question whose evidence is scattered over documents. This module states the same derived
deltas and the same reading over each question-type slice separately, so the report can name the
slices retrieval fails to pay for instead of averaging them away.

Each slice is decided on its OWN items, which is what makes these readings diagnostic rather than
decisions: a slice holds a fraction of the run's items, so its interval is wider and its verdict
turns over far less evidence than the pooled one. The pooled verdict remains the corpus decision,
and the adopt-or-reject call on `retrieved_document` is deliberately NOT taken per slice -- a
shippable configuration chosen off a dozen items of one question type is exactly the reading the
minimum-evidence gate exists to refuse.
"""

from collections.abc import Mapping, Sequence

from llb.eval.context_ablation.derived import contamination_report, paired_deltas
from llb.eval.context_ablation.models import (
    ContaminationReport,
    SliceReading,
)
from llb.eval.context_ablation.verdict import decide_population
from llb.eval.paired_cases import CaseRows
from llb.rag.fusion_evidence.slices import MetricVectors


def _slice_contamination(
    baseline: str, rows: CaseRows, item_ids: Sequence[str], positions: Sequence[int]
) -> ContaminationReport:
    return contamination_report(baseline, rows, [item_ids[i] for i in positions])


def _slice_skipped(
    skipped_by_lane: Mapping[str, list[str]], sliced_ids: set[str]
) -> dict[str, int]:
    """How many of THIS slice's items each lane skipped."""
    return {
        label: sum(1 for item_id in skipped if item_id in sliced_ids)
        for label, skipped in skipped_by_lane.items()
    }


def slice_readings(
    grouped: Mapping[str, list[int]],
    per_slice_sets: Mapping[str, list[list[int]]],
    *,
    by_lane: Mapping[str, MetricVectors],
    item_ids: Sequence[str],
    baseline: str,
    baseline_rows: CaseRows,
    skipped_by_lane: Mapping[str, list[str]],
    confidence: float,
    resamples: int,
    seed: int,
) -> list[SliceReading]:
    """One derived table and one reading per question type, in slice-name order."""
    readings: list[SliceReading] = []
    for name in sorted(grouped):
        positions = grouped[name]
        sliced_ids = {item_ids[i] for i in positions}
        contamination = _slice_contamination(baseline, baseline_rows, item_ids, positions)
        derived = paired_deltas(
            by_lane,
            item_ids,
            skipped_by_lane,
            positions,
            per_slice_sets[name],
            confidence,
            resamples,
            seed,
        )
        readings.append(
            {
                "slice": name,
                "n": len(positions),
                "derived": derived,
                "contamination": contamination,
                "verdict": decide_population(
                    derived,
                    contamination,
                    baseline=baseline,
                    n=len(positions),
                    skipped=_slice_skipped(skipped_by_lane, sliced_ids),
                    confidence=confidence,
                ),
            }
        )
    return readings


__all__ = ["slice_readings"]
