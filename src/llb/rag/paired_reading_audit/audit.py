"""Read-only re-decision of vector-backed paired artifacts under calibrated randomization."""

from pathlib import Path
from typing import Any, cast

from typing_extensions import TypedDict

from llb.rag.fusion_evidence.paired import PairedComparison, reading_of
from llb.rag.fusion_evidence.randomization import randomization_separates
from llb.rag.fusion_evidence.selection import SelectionAdjustment
from llb.rag.fusion_evidence.verdict import GAIN_METRICS
from llb.rag.paired_reading_audit.artifacts import (
    RebuiltArtifact,
    SkippedArtifact,
    artifacts as iter_artifacts,
    skipped_artifacts,
)


class ReadingChange(TypedDict):
    artifact: str
    comparison: str
    previous: str
    calibrated: str
    randomization_p: float


class VerdictReading(TypedDict):
    artifact: str
    lane: str
    previous: str
    calibrated: str
    selection_survives: bool


class SelectionReading(TypedDict):
    artifact: str
    lane: str
    hypothesis: str
    unadjusted_p: float
    adjusted_p: float
    survives: bool


class PairedReadingAudit(TypedDict):
    artifacts: int
    comparisons: int
    reading_changes: list[ReadingChange]
    verdicts: list[VerdictReading]
    selection_readings: list[SelectionReading]
    skipped_artifacts: list[SkippedArtifact]


def _comparison_map(value: Any, path: str = "") -> dict[str, PairedComparison]:
    """Find every persisted PairedComparison by structural shape and stable JSON path."""
    found: dict[str, PairedComparison] = {}
    if isinstance(value, dict):
        if {"delta", "wins", "losses", "ties", "sign_test_p"} <= value.keys():
            found[path or "$"] = cast(PairedComparison, value)
            return found
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            found.update(_comparison_map(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(_comparison_map(child, f"{path}[{index}]"))
    return found


def _stored_reading(comparison: PairedComparison, confidence: float) -> str:
    stability = comparison.get("stability")
    return stability["reading"] if stability is not None else reading_of(comparison, confidence)


def _selection_adjustment(entry: RebuiltArtifact) -> SelectionAdjustment | None:
    value = (entry.calibrated.get("verdict") or {}).get("selection_adjustment")
    return cast(SelectionAdjustment, value) if value is not None else None


def _selected_hypotheses(entry: RebuiltArtifact, adjustment: SelectionAdjustment) -> list[str]:
    if entry.lane != "fusion sweep":
        return list(adjustment["p_values"])
    best = str(entry.calibrated["verdict"].get("best_row") or "")
    return [
        key
        for key in adjustment["p_values"]
        if key.startswith(f"{best} :: ") and any(key.endswith(metric) for metric in GAIN_METRICS)
    ]


def _selection_readings(entry: RebuiltArtifact) -> list[SelectionReading]:
    adjustment = _selection_adjustment(entry)
    if adjustment is None:
        return []
    confidence = float(entry.previous.get("confidence", 0.95))
    if "uncertainty" in entry.previous:
        confidence = float(entry.previous["uncertainty"]["confidence"])
    return [
        {
            "artifact": str(entry.path),
            "lane": entry.lane,
            "hypothesis": key,
            "unadjusted_p": adjustment["p_values"][key]["unadjusted_p"],
            "adjusted_p": adjustment["p_values"][key]["adjusted_p"],
            "survives": randomization_separates(
                adjustment["p_values"][key]["adjusted_p"], confidence
            ),
        }
        for key in _selected_hypotheses(entry, adjustment)
    ]


def _verdict(entry: RebuiltArtifact, selections: list[SelectionReading]) -> VerdictReading:
    previous = entry.previous.get("verdict") or {}
    calibrated = entry.calibrated.get("verdict") or {}
    return {
        "artifact": str(entry.path),
        "lane": entry.lane,
        "previous": str(previous.get("decision", "unavailable")),
        "calibrated": str(calibrated.get("decision", "unavailable")),
        "selection_survives": any(reading["survives"] for reading in selections),
    }


def audit_paired_readings(data_dir: Path) -> PairedReadingAudit:
    """Rebuild supported vector-backed artifacts and list every changed reading."""
    changes: list[ReadingChange] = []
    verdicts: list[VerdictReading] = []
    selection_readings: list[SelectionReading] = []
    artifact_count = 0
    comparisons = 0
    for entry in iter_artifacts(data_dir):
        artifact_count += 1
        old = _comparison_map(entry.previous)
        new = _comparison_map(entry.calibrated)
        confidence = float(entry.previous.get("confidence", 0.95))
        if "uncertainty" in entry.previous:
            confidence = float(entry.previous["uncertainty"]["confidence"])
        for key in sorted(old.keys() & new.keys()):
            comparisons += 1
            previous = _stored_reading(old[key], confidence)
            calibrated = reading_of(new[key], confidence)
            if previous != calibrated:
                changes.append(
                    {
                        "artifact": str(entry.path),
                        "comparison": key,
                        "previous": previous,
                        "calibrated": calibrated,
                        "randomization_p": new[key]["randomization_p"],
                    }
                )
        selections = _selection_readings(entry)
        selection_readings.extend(selections)
        verdicts.append(_verdict(entry, selections))
    return {
        "artifacts": artifact_count,
        "comparisons": comparisons,
        "reading_changes": changes,
        "verdicts": verdicts,
        "selection_readings": selection_readings,
        "skipped_artifacts": skipped_artifacts(data_dir),
    }
