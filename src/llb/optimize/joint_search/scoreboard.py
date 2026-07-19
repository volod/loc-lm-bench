"""Scoreboard assembly from finalist final-split results."""

from typing import Any, Sequence

from llb.core.contracts.runs import EvalResult
from llb.optimize.joint_search.models import FinalistTuneResult
from llb.optimize.joint_search.report import assert_final_split
from llb.optimize.tuning_space import FINAL_SPLIT


def scoreboard_entries(
    finalists: Sequence[FinalistTuneResult],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Build scoreboard rows + recommended pick; enforces final-split only."""
    entries: list[dict[str, Any]] = []
    for finalist in finalists:
        for pick, result in finalist.finals.items():
            entry = {
                "model": finalist.name,
                "backend": finalist.backend,
                "source": finalist.source,
                "pick": pick,
                "quality": quality_from_result(result),
                "overrides": finalist.overrides_by_pick.get(pick, {}),
                "split": split_from_result(result),
                "study_name": finalist.study_name,
            }
            assert_final_split(entry)
            entries.append(entry)
    recommended = None
    if entries:
        ranked = sorted(
            entries,
            key=lambda e: (
                -(e["quality"] if isinstance(e["quality"], (int, float)) else -1.0),
                e["model"],
                e["pick"],
            ),
        )
        recommended = dict(ranked[0])
    return entries, recommended


def quality_from_result(result: EvalResult) -> float | None:
    rows = result.get("rows") or []
    if rows and isinstance(rows[0], dict) and "quality" in rows[0]:
        return float(rows[0]["quality"])
    metrics = result.get("metrics") or {}
    if isinstance(metrics, dict) and "objective_score" in metrics:
        return float(metrics["objective_score"])
    return None


def split_from_result(result: EvalResult) -> str:
    """Read the eval split from the manifest; default to final for injectable fakes."""
    manifest = result.get("manifest")
    if isinstance(manifest, dict):
        split = manifest.get("split")
        if isinstance(split, str) and split:
            return split
    return FINAL_SPLIT
