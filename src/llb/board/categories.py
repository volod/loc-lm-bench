"""Category board and guarded composite loading."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from llb.core.contracts import JsonObject
from llb.scoring.aggregate import ModelResult
from llb.scoring.composite_builder import build_category_composite_rows
from llb.scoring.composite_types import CompositeComponent, CompositeIssue

from llb.board.io import read_case_series

_LOG = logging.getLogger(__name__)

AGENTIC_METHOD = "agentic"
CATEGORY_METHODS = (
    "security",
    "tooling",
    AGENTIC_METHOD,
    "summarization",
    "structured",
    "text-analysis",
)
CATEGORY_OBJECTIVE_COLUMNS: dict[str, tuple[str, ...]] = {
    "security": ("objective_score", "defended"),
    "tooling": ("objective_score", "correct"),
    AGENTIC_METHOD: ("objective_score", "success"),
    "summarization": ("objective_score", "coverage"),
    "structured": ("objective_score", "score"),
    "text_analysis": ("objective_score",),
    "text-analysis": ("objective_score",),
}


@dataclass
class CategoryRunRecord:
    result: ModelResult
    config: JsonObject
    run_dir: str
    created_at: str
    data_verified: bool
    verification_ref: str | None
    verification_error: str | None = None


def category_case_objectives(config: JsonObject, run_dir: Path) -> list[float]:
    category = str(config.get("category", ""))
    columns = CATEGORY_OBJECTIVE_COLUMNS.get(category, ("objective_score",))
    for column in columns:
        values = read_case_series(run_dir, column)
        if values:
            return values
    return []


def category_record_from_manifest(manifest: JsonObject, run_dir: Path) -> CategoryRunRecord | None:
    """Build a category run record from one persisted run bundle."""
    config = manifest.get("config") or {}
    tier = config.get("tier")
    model = config.get("model")
    if not tier or not model:
        return None
    metrics = manifest.get("metrics") or {}
    result = ModelResult(
        model=str(model),
        backend=str(config.get("backend", "?")),
        objective_score=float(metrics.get("objective_score", 0.0)),
        n_cases=int(manifest.get("n_cases", 0)),
        reliability=float(metrics.get("reliability", 1.0)),
        tokens_per_s=float(metrics.get("tokens_per_s", 0.0)),
        tier=str(tier),
        case_objectives=category_case_objectives(config, run_dir),
    )
    verification_ref = config.get("verification_ref")
    verification_error = _verification_error(config, verification_ref, run_dir)
    return CategoryRunRecord(
        result=result,
        config=config,
        run_dir=str(run_dir),
        created_at=str(manifest.get("created_at", "")),
        data_verified=bool(config.get("data_verified", False)),
        verification_ref=str(verification_ref) if verification_ref else None,
        verification_error=verification_error,
    )


def _verification_error(config: JsonObject, verification_ref: object, run_dir: Path) -> str | None:
    if not bool(config.get("data_verified", False)):
        return None
    if not verification_ref:
        return "missing verification_ref"
    from llb.goldset.verify import check_verification_ref

    status = check_verification_ref(str(verification_ref), base_dir=run_dir)
    return status.reason if not status.valid else None


def _category_manifest_paths(data_dir: Path | str) -> list[Path]:
    paths: list[Path] = []
    for method in CATEGORY_METHODS:
        root = Path(data_dir) / method
        if root.exists():
            paths.extend(path for path in sorted(root.glob("*/manifest.json")) if _is_run_dir(path))
    return paths


def _is_run_dir(manifest_path: Path) -> bool:
    return not manifest_path.parent.name.startswith(".")


def _read_category_record(manifest_path: Path) -> CategoryRunRecord | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return category_record_from_manifest(manifest, manifest_path.parent)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        _LOG.warning("[board] unreadable category bundle %s: %s", manifest_path.parent, exc)
        return None


def _keep_best_record(
    by_tier: dict[str, dict[str, CategoryRunRecord]], record: CategoryRunRecord
) -> None:
    best = by_tier.setdefault(record.result.tier, {})
    current = best.get(record.result.model)
    if current is None or record.result.objective_score > current.result.objective_score:
        best[record.result.model] = record


def load_category_run_records(data_dir: Path | str) -> dict[str, list[CategoryRunRecord]]:
    """Load category run bundles grouped by tier, keeping the best run per model within a tier."""
    by_tier: dict[str, dict[str, CategoryRunRecord]] = {}
    for manifest_path in _category_manifest_paths(data_dir):
        record = _read_category_record(manifest_path)
        if record is not None:
            _keep_best_record(by_tier, record)
    return {tier: list(models.values()) for tier, models in by_tier.items()}


def load_category_records(data_dir: Path | str) -> dict[str, list[ModelResult]]:
    """Load category run bundles grouped by tier for separate category boards."""
    return {
        tier: [record.result for record in records]
        for tier, records in load_category_run_records(data_dir).items()
    }


def load_category_composite(
    data_dir: Path | str,
    *,
    require_verified: bool = True,
    require_ci: bool = True,
) -> tuple[list[JsonObject], list[CompositeIssue]]:
    """Load the guarded category-suite composite headline from persisted category runs."""
    components_by_tier = {
        tier: [
            CompositeComponent(
                result=record.result,
                data_verified=record.data_verified,
                verification_ref=record.verification_ref,
                verification_error=record.verification_error,
            )
            for record in records
        ]
        for tier, records in load_category_run_records(data_dir).items()
    }
    return build_category_composite_rows(
        components_by_tier,
        require_verified=require_verified,
        require_ci=require_ci,
    )
