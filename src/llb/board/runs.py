"""Final-split RAG run loading for the board."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from llb.core.contracts import JsonObject, ScreenReport
from llb.finetune.registry.io import load_registry, registry_path
from llb.finetune.registry.model import AdapterEntry
from llb.finetune.registry.staleness import staleness
from llb.scoring.leaderboard import DEFAULT_WEIGHT_JUDGE, ModelResult, headline_quality

from llb.board.io import mean_or_none, read_case_objectives, read_case_series, read_case_splits

_LOG = logging.getLogger(__name__)

CONFIG_KEYS = ("strategy", "chunk_size", "chunk_overlap", "top_k", "retrieval_mode")
FINAL_SPLIT = "final"
# An adapter-backed row is only reproducible through the registry, so an unregistered adapter's
# bundle never renders and a stale one is stamped in the model label before it can be compared.
STALE_STAMP = "stale"


@dataclass
class RunRecord:
    result: ModelResult
    config: JsonObject
    run_dir: str
    created_at: str
    split: str


def record_from_manifest(
    manifest: JsonObject,
    run_dir: Path,
    *,
    registry: dict[str, AdapterEntry] | None = None,
) -> RunRecord | None:
    """Build a final-split RunRecord, or None for incomplete/non-leaderboard manifests.

    An adapter-backed bundle additionally needs a registry entry: without one its tuned number
    cannot be traced back to a dataset digest and source run, so the row is dropped rather than
    ranked. A registered-but-stale adapter renders with a `[stale]` stamp on its model label.
    """
    config = manifest.get("config") or {}
    model = config.get("model")
    if not model:
        return None
    split = _declared_or_legacy_split(manifest, run_dir)
    if split != FINAL_SPLIT:
        return None
    model = _adapter_model_label(str(model), config, registry or {}, run_dir)
    if model is None:
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


def _adapter_model_label(
    model: str, config: JsonObject, registry: dict[str, AdapterEntry], run_dir: Path
) -> str | None:
    """Stamp or reject an adapter-backed row; plain base-model rows pass through untouched."""
    adapter = config.get("adapter")
    if not isinstance(adapter, dict):
        return model
    digest = str(adapter.get("adapter_digest") or "")
    entry = registry.get(digest)
    if entry is None:
        _LOG.warning(
            "[board] skipping %s: adapter %s is not registered (`llb list-adapters`)",
            run_dir,
            digest[:12] or "?",
        )
        return None
    return f"{model} [{STALE_STAMP}]" if staleness(entry).is_stale else model


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


def load_run_records(
    run_root: Path | str, *, data_dir: Path | str | None = None
) -> list[RunRecord]:
    """Load published final-split bundles under `run_root`, resolving adapters through the registry.

    `data_dir` locates `adapters/registry.jsonl`; it defaults to the parent of the canonical
    `$DATA_DIR/run-eval/` bundle root.
    """
    root = Path(run_root)
    records: list[RunRecord] = []
    if not root.exists():
        return records
    registry = load_registry(registry_path(Path(data_dir) if data_dir is not None else root.parent))
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record = record_from_manifest(manifest, manifest_path.parent, registry=registry)
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
