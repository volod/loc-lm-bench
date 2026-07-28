"""Read-only re-decision of vector-backed paired artifacts under calibrated randomization."""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from typing_extensions import TypedDict

from llb.board.io import read_case_rows
from llb.eval.context_ablation.compare import compare_context_strategies
from llb.eval.embedder_adoption.compare import CellRows, compare_cells
from llb.eval.embedder_adoption.models import CellSpec
from llb.eval.paired_cases import CaseRows
from llb.rag.embedding_bakeoff_uncertainty import paired_rows
from llb.rag.embedding_bakeoff_verdict import decide_verdict
from llb.rag.fusion_evidence.paired import PairedComparison, reading_of


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


class PairedReadingAudit(TypedDict):
    artifacts: int
    comparisons: int
    reading_changes: list[ReadingChange]
    verdicts: list[VerdictReading]


@dataclass(frozen=True)
class _RebuiltArtifact:
    path: Path
    lane: str
    previous: Mapping[str, Any]
    calibrated: Mapping[str, Any]


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


def _embedding_artifact(path: Path, payload: Mapping[str, Any]) -> _RebuiltArtifact:
    settings = payload["uncertainty"]
    baseline = settings["baseline"]
    vectors: dict[str, dict[str, list[float]]] = {}
    for item in payload["paired_items"]:
        for model, metrics in item["models"].items():
            target = vectors.setdefault(model, {metric: [] for metric in metrics})
            for metric, value in metrics.items():
                target[metric].append(float(value))
    rebuilt = paired_rows(
        vectors,
        baseline,
        resamples=int(settings["resamples"]),
        confidence=float(settings["confidence"]),
        seed=int(settings["seed"]),
    )
    candidates = []
    for row in payload["candidates"]:
        candidates.append({**row, "paired_vs_baseline": rebuilt.get(row["model"])})
    verdict = decide_verdict(
        rebuilt,
        baseline,
        tuple(settings["bars"]),
        float(settings["confidence"]),
    )
    return _RebuiltArtifact(
        path,
        "embedder bake-off",
        payload,
        {**payload, "candidates": candidates, "verdict": verdict},
    )


def _context_artifact(path: Path, payload: Mapping[str, Any]) -> _RebuiltArtifact:
    lane_names = list(payload["lanes"])
    rows: dict[str, CaseRows] = {}
    for lane in lane_names:
        rows[lane] = [
            {"item_id": item["item_id"], **item["lanes"][lane]}
            for item in payload["items"]
            if lane in item["lanes"]
        ]
    question_types = {
        item["item_id"]: item["question_type"]
        for item in payload["items"]
        if item.get("question_type") is not None
    }
    rebuilt = compare_context_strategies(
        rows,
        question_types,
        baseline=payload["baseline"],
        run_dirs={lane: payload["lanes"][lane]["run_dirs"] for lane in lane_names},
        resamples=int(payload["resamples"]),
        confidence=float(payload["confidence"]),
        seed=int(payload["seed"]),
    )
    return _RebuiltArtifact(path, "context ablation", payload, rebuilt)


def _adoption_artifact(path: Path, payload: Mapping[str, Any]) -> _RebuiltArtifact:
    cells: list[tuple[CellSpec, CellRows]] = []
    run_dirs: dict[str, dict[str, list[str]]] = {}
    for cell in payload["cells"]:
        rows: dict[str, CaseRows] = {}
        run_dirs[cell["label"]] = {}
        for model, lane in cell["lanes"].items():
            directories = list(lane["run_dirs"])
            run_dirs[cell["label"]][model] = directories
            rows[model] = [
                row
                for directory in directories
                for row in read_case_rows(Path(directory) / "scores.jsonl")
            ]
        cells.append((CellSpec(cell["top_k"], cell["reranker"]), rows))
    rebuilt = compare_cells(
        cells,
        run_dirs,
        baseline=payload["baseline"],
        candidate=payload["candidate"],
        metrics=payload["metrics"],
        resamples=int(payload["resamples"]),
        confidence=float(payload["confidence"]),
        seed=int(payload["seed"]),
    )
    return _RebuiltArtifact(path, "adoption bar", payload, rebuilt)


def _artifacts(data_dir: Path) -> Iterator[_RebuiltArtifact]:
    for path in sorted((data_dir / "compare-embeddings").glob("*/report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("paired_items") and payload.get("uncertainty"):
            yield _embedding_artifact(path, payload)
    for path in sorted((data_dir / "context-ablation").glob("*/comparison.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("items"):
            yield _context_artifact(path, payload)
    for path in sorted((data_dir / "embedder-adoption-bar").glob("*/comparison.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("cells"):
            yield _adoption_artifact(path, payload)


def _verdict(entry: _RebuiltArtifact) -> VerdictReading:
    previous = entry.previous.get("verdict") or {}
    calibrated = entry.calibrated.get("verdict") or {}
    return {
        "artifact": str(entry.path),
        "lane": entry.lane,
        "previous": str(previous.get("decision", "unavailable")),
        "calibrated": str(calibrated.get("decision", "unavailable")),
    }


def audit_paired_readings(data_dir: Path) -> PairedReadingAudit:
    """Rebuild supported vector-backed artifacts and list every changed reading."""
    changes: list[ReadingChange] = []
    verdicts: list[VerdictReading] = []
    artifacts = 0
    comparisons = 0
    for entry in _artifacts(data_dir):
        artifacts += 1
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
        verdicts.append(_verdict(entry))
    return {
        "artifacts": artifacts,
        "comparisons": comparisons,
        "reading_changes": changes,
        "verdicts": verdicts,
    }
