"""Per-case row alignment shared by every lane that compares scored run bundles item by item.

Two lanes read the same canonical `scores.jsonl` rows and ask a paired question about them -- the
retrieval-lane answer comparison (`llb/eval/answer_quality/`) and the context ablation
(`llb/eval/context_ablation/`). What they share is everything BEFORE the statistics: agree on the
item set, refuse a set that is not actually shared, and project each lane's rows onto per-metric
vectors in one aligned order. That lives here so the two cannot drift into pairing rows two
slightly different ways.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llb.board.io import read_case_rows
from llb.rag.fusion_evidence.slices import MetricVectors

CaseRows = list[Mapping[str, Any]]

# What a recorded lane is made of on disk: one ordinary `run-eval` bundle per (lane, split), each
# holding the canonical per-case rows the lane was compared on.
SCORES_FILENAME = "scores.jsonl"


def shared_item_ids(lanes: Mapping[str, CaseRows]) -> list[str]:
    """The sorted item ids every lane scored; raises when the lanes disagree.

    A paired comparison over different item sets is not a comparison, so a lane that dropped or
    added a case fails loudly instead of being silently intersected away.
    """
    if not lanes:
        raise ValueError("the comparison needs at least one scored lane")
    per_lane = {label: [str(row["item_id"]) for row in rows] for label, rows in lanes.items()}
    for label, ids in per_lane.items():
        if len(set(ids)) != len(ids):
            raise ValueError(f"lane {label!r} scored an item id more than once")
    reference_label, reference = next(iter(per_lane.items()))
    expected = set(reference)
    for label, ids in per_lane.items():
        if set(ids) != expected:
            missing = sorted(expected - set(ids))
            extra = sorted(set(ids) - expected)
            raise ValueError(
                f"lanes {reference_label!r} and {label!r} scored different item sets "
                f"(missing {missing[:3]}, extra {extra[:3]})"
            )
    return sorted(expected)


def lane_vectors(rows: CaseRows, item_ids: Sequence[str], metrics: Sequence[str]) -> MetricVectors:
    """Per-metric values aligned to the shared item order."""
    by_id = {str(row["item_id"]): row for row in rows}
    return {
        metric: [float(by_id[item_id].get(metric, 0.0) or 0.0) for item_id in item_ids]
        for metric in metrics
    }


def recorded_lane_rows(run_dirs: Sequence[str | Path]) -> CaseRows:
    """One lane's per-case rows read back from the run bundles a comparison recorded for it.

    Every lane that re-reads a finished comparison -- the answer-quality re-render, the paired
    reading audit's adoption artifact -- reconstitutes a lane exactly this way, so the two cannot
    drift into two notions of what a recorded lane is. A named bundle with no `scores.jsonl` is an
    error rather than an empty lane, which would silently shrink the compared item set.
    """
    rows: CaseRows = []
    for run_dir in run_dirs:
        scores = Path(run_dir) / SCORES_FILENAME
        if not scores.is_file():
            raise FileNotFoundError(f"recorded run bundle {run_dir} has no {SCORES_FILENAME}")
        rows.extend(read_case_rows(scores))
    return rows


def rows_by_item(rows: CaseRows) -> dict[str, Mapping[str, Any]]:
    """One lane's rows keyed by item id."""
    return {str(row["item_id"]): row for row in rows}


__all__ = [
    "CaseRows",
    "SCORES_FILENAME",
    "lane_vectors",
    "recorded_lane_rows",
    "rows_by_item",
    "shared_item_ids",
]
