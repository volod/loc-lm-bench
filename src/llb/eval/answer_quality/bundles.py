"""Reconstitute the lanes of a recorded answer-quality comparison from the bundles it named.

An answer-quality comparison costs hours of generation, and the artifact it produced is locked to
the report format it was rendered under. The comparison itself is PURE over the per-case rows,
though, and every lane's run bundles are recorded in its own `comparison.json` -- so the recorded
lanes can be read back from disk and compared again with no model call
(`llb.eval.answer_quality.rerender`).

Reading a lane back is the same operation the paired-reading audit performs on a recorded artifact,
so both go through `recorded_lane_rows` and cannot drift into two notions of what a recorded lane
is. What this module adds on top is what only an answer-quality lane needs: the multi-span coverage
columns, recomputed at the lane's OWN retrieval budget exactly as the original run computed them,
and the refusal that fires before any of it when a bundle no longer describes its lane
(`llb.eval.answer_quality.bundle_match`).
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.eval.answer_quality.bundle_match import BundleMismatch, lane_budget, refusal
from llb.eval.answer_quality.coverage import read_case_coverage, with_coverage
from llb.eval.answer_quality.lanes import parse_lane_label
from llb.eval.answer_quality.models import LaneSpec
from llb.eval.paired_cases import CaseRows, recorded_lane_rows


@dataclass(frozen=True)
class RecordedComparison:
    """One recorded `comparison.json`, split into the parts a re-render needs."""

    path: Path
    payload: Mapping[str, Any]
    metadata: Mapping[str, Any]
    lanes: dict[str, LaneSpec]
    run_dirs: dict[str, list[str]]


def read_recorded(path: Path) -> RecordedComparison:
    """Load a recorded answer-quality comparison and parse each lane label back into its knobs."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    lanes = payload.get("lanes") if isinstance(payload, dict) else None
    if not isinstance(lanes, dict) or not lanes or "item_ids" not in payload:
        raise BundleMismatch(f"{path}: not a compare-answer-quality comparison")
    specs: dict[str, LaneSpec] = {}
    run_dirs: dict[str, list[str]] = {}
    for label, lane in lanes.items():
        try:
            specs[label] = parse_lane_label(str(label))
        except ValueError as exc:
            raise BundleMismatch(f"{path}: lane {label!r} no longer parses: {exc}") from None
        recorded = list(lane.get("run_dirs") or [])
        if not recorded:
            raise BundleMismatch(f"{path}: lane {label!r} recorded no run bundle to re-read")
        run_dirs[label] = [str(run_dir) for run_dir in recorded]
    return RecordedComparison(
        Path(path), payload, dict(payload.get("metadata") or {}), specs, run_dirs
    )


def resolve_lane_rows(recorded: RecordedComparison) -> dict[str, CaseRows]:
    """Every recorded lane's per-case rows plus its coverage columns, or one listed refusal.

    Coverage is recomputed at the LANE's own budget from each bundle's retrieval sidecar, exactly
    as the original run did, so a budget sweep's cells keep the thing they measured. Every lane is
    checked before any is trusted: the refusal lists what drifted across the whole bundle set.
    """
    rows: dict[str, CaseRows] = {}
    mismatches: list[str] = []
    for label, run_dirs in recorded.run_dirs.items():
        budget, lane_mismatches = lane_budget(recorded.lanes[label], recorded.metadata, run_dirs)
        mismatches += lane_mismatches
        if lane_mismatches:
            continue
        lane_rows: CaseRows = []
        for run_dir in run_dirs:
            try:
                bundle_rows = recorded_lane_rows([run_dir])
            except (FileNotFoundError, ValueError) as exc:
                mismatches.append(f"lane {label!r}: {exc}")
                continue
            coverage = read_case_coverage(Path(run_dir), budget)
            lane_rows.extend(with_coverage(list(bundle_rows), coverage))
        rows[label] = lane_rows
    if mismatches:
        raise BundleMismatch(refusal(recorded.path, mismatches))
    return rows


__all__ = [
    "BundleMismatch",
    "RecordedComparison",
    "read_recorded",
    "resolve_lane_rows",
]
