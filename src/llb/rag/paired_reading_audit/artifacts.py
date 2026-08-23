"""Vector-backed artifact adapters for the paired-reading audit."""

import json
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typing_extensions import TypedDict

from llb.eval.context_ablation.compare import compare_context_strategies
from llb.eval.embedder_adoption.compare import CellRows, compare_cells
from llb.eval.embedder_adoption.models import CellSpec
from llb.eval.paired_cases import CaseRows, recorded_lane_rows
from llb.rag.embedding_bakeoff.selection import adjust_bakeoff_selection
from llb.rag.embedding_bakeoff.uncertainty import paired_rows
from llb.rag.embedding_bakeoff.verdict import decide_verdict
from llb.rag.fusion_evidence.models import METRICS
from llb.rag.fusion_evidence.paired import paired_comparison
from llb.rag.fusion_evidence.selection_family import adjust_fusion_selection
from llb.rag.fusion_evidence.stats import bootstrap_index_sets
from llb.rag.fusion_evidence.verdict import decide as decide_fusion


class SkippedArtifact(TypedDict):
    artifact: str
    lane: str
    reason: str


@dataclass(frozen=True)
class RebuiltArtifact:
    path: Path
    lane: str
    previous: Mapping[str, Any]
    calibrated: Mapping[str, Any]


def _embedding_artifact(path: Path, payload: Mapping[str, Any]) -> RebuiltArtifact:
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
    candidates = [
        {**row, "paired_vs_baseline": rebuilt.get(row["model"])} for row in payload["candidates"]
    ]
    bars = tuple(settings["bars"])
    adjustment = adjust_bakeoff_selection(
        vectors,
        baseline,
        bars,
        resamples=int(settings["resamples"]),
        seed=int(settings["seed"]),
    )
    verdict = decide_verdict(
        rebuilt,
        baseline,
        bars,
        float(settings["confidence"]),
        adjustment=adjustment,
    )
    return RebuiltArtifact(
        path,
        "embedder bake-off",
        payload,
        {**payload, "candidates": candidates, "verdict": verdict},
    )


def _context_artifact(path: Path, payload: Mapping[str, Any]) -> RebuiltArtifact:
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
    return RebuiltArtifact(path, "context ablation", payload, rebuilt)


def _adoption_artifact(path: Path, payload: Mapping[str, Any]) -> RebuiltArtifact:
    cells: list[tuple[CellSpec, CellRows]] = []
    run_dirs: dict[str, dict[str, list[str]]] = {}
    for cell in payload["cells"]:
        rows: dict[str, CaseRows] = {}
        run_dirs[cell["label"]] = {}
        for model, lane in cell["lanes"].items():
            directories = list(lane["run_dirs"])
            run_dirs[cell["label"]][model] = directories
            rows[model] = recorded_lane_rows(directories)
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
    return RebuiltArtifact(path, "adoption bar", payload, rebuilt)


def _fusion_artifact(path: Path, payload: Mapping[str, Any]) -> RebuiltArtifact:
    focus_items = payload["focus_items"]
    baseline = payload["baseline"]
    vectors = {
        label: {
            metric: [float(item["rows"][label][metric]) for item in focus_items]
            for metric in METRICS
        }
        for label in payload["rows"]
    }
    resamples = int(payload["resamples"])
    confidence = float(payload["confidence"])
    index_sets = bootstrap_index_sets(len(focus_items), resamples, int(payload["seed"]))
    rebuilt: dict[str, Any] = deepcopy(dict(payload))
    focus = payload["focus_slice"]
    for label, row_vectors in vectors.items():
        for metric in METRICS:
            rebuilt["rows"][label]["slices"][focus]["paired_vs_baseline"][metric] = (
                paired_comparison(
                    row_vectors[metric],
                    vectors[baseline][metric],
                    index_sets,
                    confidence,
                )
            )
    adjustment = adjust_fusion_selection(
        vectors,
        baseline=baseline,
        indexes=list(range(len(focus_items))),
        resamples=resamples,
        index_sets=index_sets,
    )
    rebuilt["verdict"] = decide_fusion(
        rebuilt["rows"],
        baseline=baseline,
        focus_slice=focus,
        confidence=confidence,
        adjustment=adjustment,
    )
    return RebuiltArtifact(path, "fusion sweep", payload, rebuilt)


def artifacts(data_dir: Path) -> Iterator[RebuiltArtifact]:
    for path in sorted((data_dir / "compare-embeddings").rglob("report.json")):
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
    for path in sorted((data_dir / "graph-vector-fusion-multihop").glob("*/comparison.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("focus_items"):
            yield _fusion_artifact(path, payload)


def skipped_artifacts(data_dir: Path) -> list[SkippedArtifact]:
    """Legacy grid reports whose aggregate ledgers cannot recover joint item correlation."""
    skipped: list[SkippedArtifact] = []
    for path in sorted((data_dir / "compare-embeddings").rglob("report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("uncertainty") and not payload.get("paired_items"):
            skipped.append(
                {
                    "artifact": str(path),
                    "lane": "embedder bake-off",
                    "reason": (
                        "legacy report has aggregate paired rows but no aligned paired_items; "
                        "use its vector-backed corpus re-run"
                    ),
                }
            )
    return skipped
