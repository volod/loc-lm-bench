"""Final-split RAG run loading for the board."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from llb.contracts import JsonObject, ScreenReport
from llb.scoring.aggregate import DEFAULT_WEIGHT_JUDGE, ModelResult, headline_quality

from llb.board.io import mean_or_none, read_case_objectives, read_case_series, read_case_splits

_LOG = logging.getLogger(__name__)

CONFIG_KEYS = ("strategy", "chunk_size", "chunk_overlap", "top_k", "retrieval_mode")
FINAL_SPLIT = "final"


@dataclass
class RunRecord:
    result: ModelResult
    config: JsonObject
    run_dir: str
    created_at: str
    split: str


def record_from_manifest(manifest: JsonObject, run_dir: Path) -> RunRecord | None:
    """Build a final-split RunRecord, or None for incomplete/non-leaderboard manifests."""
    config = manifest.get("config") or {}
    model = config.get("model")
    if not model:
        return None
    split = _declared_or_legacy_split(manifest, run_dir)
    if split != FINAL_SPLIT:
        return None
    metrics = manifest.get("metrics") or {}
    telemetry = manifest.get("telemetry") or {}
    case_semantic = read_case_series(run_dir, "semantic")
    case_judge = read_case_series(run_dir, "judge_score")
    result = ModelResult(
        model=str(model),
        backend=str(config.get("backend", "?")),
        objective_score=float(metrics.get("objective_score", 0.0)),
        n_cases=int(manifest.get("n_cases", 0)),
        reliability=float(metrics.get("reliability", 1.0)),
        tokens_per_s=float(metrics.get("tokens_per_s", 0.0)),
        peak_vram_mb=telemetry.get("peak_vram_mb"),
        judge_score=mean_or_none(case_judge),
        semantic_score=mean_or_none(case_semantic),
        case_objectives=read_case_objectives(run_dir),
        case_semantic=case_semantic,
        case_judge=case_judge,
    )
    return RunRecord(
        result=result,
        config=config,
        run_dir=str(run_dir),
        created_at=str(manifest.get("created_at", "")),
        split=split,
    )


def _declared_or_legacy_split(manifest: JsonObject, run_dir: Path) -> str | None:
    declared_split = manifest.get("split")
    if declared_split is not None:
        return str(declared_split)
    legacy_splits = read_case_splits(run_dir)
    if len(legacy_splits) > 1:
        _LOG.warning("[board] mixed splits in legacy run bundle: %s", run_dir)
    if legacy_splits == {FINAL_SPLIT}:
        return FINAL_SPLIT
    return None


def load_run_records(run_root: Path | str) -> list[RunRecord]:
    """Load published final-split bundles under `run_root`."""
    root = Path(run_root)
    records: list[RunRecord] = []
    if not root.exists():
        return records
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record = record_from_manifest(manifest, manifest_path.parent)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            _LOG.warning("[board] unreadable run bundle %s: %s", manifest_path.parent, exc)
            continue
        if record is not None:
            records.append(record)
    return records


def best_per_model(
    records: list[RunRecord],
    *,
    judge_trusted: bool = False,
    weight_judge: float = DEFAULT_WEIGHT_JUDGE,
) -> list[RunRecord]:
    """Keep the best run per model under the declared ranking policy."""

    def score(rec: RunRecord) -> float:
        return headline_quality(rec.result, judge_trusted, weight_judge)

    best: dict[str, RunRecord] = {}
    for rec in records:
        current = best.get(rec.result.model)
        if current is None or score(rec) > score(current):
            best[rec.result.model] = rec
    return list(best.values())


def load_screen_reports(screen_root: Path | str) -> list[ScreenReport]:
    """Load Tier-1 public-screen reports, separate from private run bundles."""
    root = Path(screen_root)
    reports: list[ScreenReport] = []
    if not root.exists():
        return reports
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _LOG.warning("[board] unreadable screen report: %s", path)
            continue
        if isinstance(data, dict) and data.get("track") and "results" in data:
            reports.append(data)  # type: ignore[arg-type]
    return reports


def config_summary(config: JsonObject) -> dict[str, object]:
    """The subset of a config shown as the model's best configuration."""
    return {key: config.get(key) for key in CONFIG_KEYS}
