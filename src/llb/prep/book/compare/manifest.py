"""The comparison manifest: where the reference, the accepts, and each lane live.

Paths are relative to the manifest file. A measured lane is loaded and retagged
with the lane id the manifest names, so a variant policy does not collide with
the original reading of the same engine. An over-VRAM lane is not loaded.
"""

from dataclasses import replace
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from llb.core.contracts.data_prep.corpus import DataPrepRow
from llb.prep.book.compare.assemble import LaneInput, compare_submissions
from llb.prep.book.compare.baseline import recorded_language
from llb.prep.book.compare.constants import BASELINE_ENGINE, BASELINE_VRAM_MIB, STATUS_MEASURED
from llb.prep.book.compare.inputs import load_accepts
from llb.prep.book.compare.report import ComparisonReport
from llb.prep.book.engines.protocol import EngineUnavailable
from llb.prep.book.engines.registry import default_adapter_registry
from llb.prep.book.metrics.constants import LANE_EXISTING
from llb.prep.book.metrics.reference import load_reference
from llb.prep.book.metrics.submission import load_citations, load_run


def _as_float(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, int):
        return value
    return float(value)


class ManifestLane(DataPrepRow):
    """One candidate. ``run`` is required only when the lane was measured."""

    lane: str = Field(min_length=1)
    engine: str = Field(min_length=1)
    role: Literal["page-reader", "variant-policy"]
    vram_mb: int = Field(ge=0)
    run: str | None = None
    variant: str = "original"
    dewarp: str | None = None
    seconds_per_page: float | None = None
    status: Literal["measured", "over-vram", "unavailable"] = STATUS_MEASURED
    detail: str = ""

    @model_validator(mode="before")
    @classmethod
    def _widen_seconds(cls, data: object) -> object:
        if not isinstance(data, dict) or "seconds_per_page" not in data:
            return data
        return {**data, "seconds_per_page": _as_float(data["seconds_per_page"])}


class ComparisonManifest(DataPrepRow):
    """Everything ``llb book-ocr compare`` needs, and no recovered page text."""

    language: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    predraft: str = Field(min_length=1)
    accepts: str = Field(min_length=1)
    citations: str = Field(min_length=1)
    baseline_vram_mb: int = Field(default=BASELINE_VRAM_MIB, ge=0)
    baseline_seconds_per_page: float | None = None
    class_counts: dict[str, int] = Field(default_factory=dict)
    source_sha256: str | None = None
    lanes: list[ManifestLane] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _widen_seconds(cls, data: object) -> object:
        if not isinstance(data, dict) or "baseline_seconds_per_page" not in data:
            return data
        widened = _as_float(data["baseline_seconds_per_page"])
        return {**data, "baseline_seconds_per_page": widened}


def run_manifest(path: Path) -> tuple[ComparisonReport, dict[str, object]]:
    """Score the manifest and return the report plus the book-ocr config."""
    manifest = _read(path)
    root = path.parent
    reference = load_reference(_path(root, manifest.reference))
    predraft = load_reference(_path(root, manifest.predraft))
    accepts = load_accepts(_path(root, manifest.accepts))
    citations = _path(root, manifest.citations)
    submissions = load_citations(citations)
    lanes: list[LaneInput] = []
    for row in manifest.lanes:
        lanes.append(_lane(row))
        if row.status != STATUS_MEASURED:
            continue
        if not row.run:
            raise ValueError(f"lane {row.lane} is measured and has no run")
        loaded = load_run(_path(root, row.run), row.engine)
        submissions.extend(replace(page, lane=row.lane) for page in loaded)
    return compare_submissions(
        reference,
        predraft,
        submissions,
        lanes,
        language=manifest.language,
        accepts=accepts,
        baseline_vram_mb=manifest.baseline_vram_mb,
        baseline_seconds_per_page=manifest.baseline_seconds_per_page,
        class_page_counts=manifest.class_counts,
        serves=registry_serves,
        baseline_lane=LANE_EXISTING,
        baseline_engine=BASELINE_ENGINE,
        recorded_language=recorded_language(citations),
        source_sha256=manifest.source_sha256,
    )


def registry_serves(engine: str, stage: str) -> bool:
    """True when the registered adapter lists the stage. Availability is not consulted."""
    try:
        adapter = default_adapter_registry().require(engine)
    except EngineUnavailable:
        return False
    return stage in adapter.stages


def _read(path: Path) -> ComparisonManifest:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return ComparisonManifest.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ValueError(f"the comparison cannot start: {path} is not a manifest: {exc}") from exc


def _lane(row: ManifestLane) -> LaneInput:
    return LaneInput(
        row.lane,
        row.engine,
        row.role,
        row.vram_mb,
        row.variant,
        row.dewarp,
        row.seconds_per_page,
        row.status,
        row.detail,
    )


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


__all__ = ["ComparisonManifest", "ManifestLane", "registry_serves", "run_manifest"]
