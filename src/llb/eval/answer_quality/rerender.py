"""Re-render a recorded answer-quality comparison under the CURRENT report, with no model call.

A comparison is pure over the per-case rows its lanes recorded, so an improvement to the artifact --
a new column, a new section, a corrected reading -- can reach a finished run without paying for
every generation again. `bundles.py` reconstitutes the recorded lanes and refuses a drifted bundle
set; this module recomputes the comparison over them and writes the artifact.

A second refusal fires here, after the recompute and before anything is written: the rebuilt
comparison must still cover the recorded item set, the recorded lanes, the recorded question-type
slices, and every metric column the artifact recorded. That is what catches a deleted retrieval
sidecar or a gold set whose question-type sidecar no longer labels the items -- drift the manifests
alone cannot see. Gaining a column the recorded run never had is the opposite of drift: that is the
report improvement reaching the old run, which is the whole reason this path exists.

Nothing here re-scores an answer or edits a recorded bundle: the run bundles stay exactly as the
generations left them, and the re-rendered artifact records which comparison it was rebuilt from.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llb.core.config import RunConfig
from llb.core.store_generations import generation_timestamp
from llb.eval.answer_quality.bundle_match import BundleMismatch, refusal
from llb.eval.answer_quality.bundles import RecordedComparison, read_recorded, resolve_lane_rows
from llb.eval.answer_quality.compare import compare_answer_quality
from llb.eval.answer_quality.conversion import budget_conversion
from llb.eval.answer_quality.models import AnswerQualityReport
from llb.eval.answer_quality.run import AnswerQualityRun, default_out_dir, write_artifacts
from llb.eval.paired_cases import CaseRows
from llb.rag.question_types import load_question_types

RERENDER_SOURCE_KEY = "rerendered_from"
RERENDER_TIMESTAMP_KEY = "rerendered_at"


def _shape_mismatches(
    recorded: RecordedComparison, rows: Mapping[str, CaseRows], report: AnswerQualityReport
) -> list[str]:
    """Ways the rebuilt comparison covers something other than what the artifact recorded.

    A metric column the bundles measure and the artifact does NOT is not drift -- it is the whole
    point: the recorded run predates a column the report has since gained, and the re-render is
    what carries it there. Only a column the bundles can no longer produce is a refusal.
    """
    mismatches: list[str] = []
    expected_ids = [str(item_id) for item_id in recorded.payload["item_ids"]]
    if report["item_ids"] != expected_ids:
        mismatches.append(
            f"the bundles score {len(report['item_ids'])} item(s), "
            f"the comparison recorded {len(expected_ids)}"
        )
    lost = [
        str(metric)
        for metric in recorded.payload.get("metrics") or []
        if metric not in report["metrics"]
    ]
    if lost:
        mismatches.append(
            f"the bundles no longer measure {lost} -- check that every recorded run bundle still "
            "has its retrieval sidecar"
        )
    for label, lane in recorded.payload["lanes"].items():
        expected_slices = sorted(lane.get("slices") or {})
        rebuilt = sorted(report["lanes"][label]["slices"])
        if expected_slices and rebuilt != expected_slices:
            mismatches.append(
                f"lane {label!r} slices into {rebuilt}, the comparison recorded "
                f"{expected_slices} -- check the gold set's question-type sidecar"
            )
    unresolved = sorted(set(recorded.payload["lanes"]) - set(rows))
    if unresolved:
        mismatches.append(f"no rows resolved for lane(s) {unresolved}")
    return mismatches


def rebuild_report(recorded: RecordedComparison) -> AnswerQualityReport:
    """The recorded comparison recomputed from its own run bundles under the current report."""
    rows = resolve_lane_rows(recorded)
    payload = recorded.payload
    cross = {
        label: str(reading["base_lane"])
        for label, reading in (payload.get("cross_readings") or {}).items()
    }
    report = compare_answer_quality(
        rows,
        load_question_types(Path(str(recorded.metadata.get("goldset", "")))),
        baseline=str(payload["baseline"]),
        run_dirs=recorded.run_dirs,
        focus_slice=str(payload["focus_slice"]),
        cross_baselines=cross,
        resamples=int(payload["resamples"]),
        confidence=float(payload["confidence"]),
        seed=int(payload["seed"]),
    )
    conversion = payload.get("budget_conversion")
    if cross and conversion is not None:
        report["budget_conversion"] = budget_conversion(
            report["cross_readings"],
            budgets=[int(budget) for budget in conversion["budgets"]],
            focus_slice=report["focus_slice"],
            coverage=report["verdict"]["coverage_metric"],
            confidence=float(payload["confidence"]),
        )
    mismatches = _shape_mismatches(recorded, rows, report)
    if mismatches:
        raise BundleMismatch(refusal(recorded.path, mismatches))
    return report


def rerender_metadata(recorded: RecordedComparison, *, timestamp: str) -> dict[str, Any]:
    """The recorded metadata, in its recorded order, plus where this re-render came from.

    Appending rather than rewriting is what makes the re-render checkable: strip the two added keys
    and the artifact is byte-identical to the one the generations produced.
    """
    return {
        **recorded.metadata,
        RERENDER_SOURCE_KEY: str(recorded.path),
        RERENDER_TIMESTAMP_KEY: timestamp,
    }


def rerender_from_bundles(
    comparison: Path,
    *,
    config: RunConfig | None = None,
    out_dir: Path | None = None,
    timestamp: str | None = None,
) -> AnswerQualityRun:
    """Re-render `comparison` from the run bundles it recorded, into a NEW artifact directory.

    The recorded artifact is never written to: a re-render is a new reading of the same
    generations, and overwriting the one the run produced would destroy the thing it is checked
    against. `out_dir` defaults to a fresh generation directory beside the other comparisons.
    """
    recorded = read_recorded(Path(comparison))
    report = rebuild_report(recorded)
    target = Path(out_dir) if out_dir is not None else default_out_dir(config or RunConfig())
    metadata = rerender_metadata(recorded, timestamp=timestamp or generation_timestamp())
    paths = write_artifacts(report, target, metadata=metadata)
    return AnswerQualityRun(report, target, paths)


__all__ = [
    "RERENDER_SOURCE_KEY",
    "RERENDER_TIMESTAMP_KEY",
    "rebuild_report",
    "rerender_from_bundles",
    "rerender_metadata",
]
