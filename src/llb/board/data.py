"""Board data loading (M3.7) -- pure, so the Streamlit page stays a thin view.

Reads FINAL-split canonical run bundles under ``$DATA_DIR/run-eval/<ts>/`` (the immutable
`manifest.json` + per-case `scores`) into the `ModelResult` rows the M3.6 ranker consumes,
keeps the best config per model, and remembers each row's config + run dir so the page can
show "best config per model". Tuning and calibration runs are excluded: allowing an Optuna
trial onto the board would leak stage-1 results into the final leaderboard. No Streamlit here
-- this half is unit-tested.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from llb.contracts import JsonObject, ScreenReport
from llb.scoring.aggregate import DEFAULT_WEIGHT_JUDGE, ModelResult, headline_quality

_LOG = logging.getLogger(__name__)

# The config knobs shown as "best config per model" on the board.
CONFIG_KEYS = ("strategy", "chunk_size", "chunk_overlap", "top_k", "retrieval_mode")
FINAL_SPLIT = "final"


@dataclass
class RunRecord:
    result: ModelResult
    config: JsonObject
    run_dir: str
    created_at: str
    split: str


def read_case_splits(run_dir: Path) -> set[str]:
    """Read represented splits for legacy manifests that predate the manifest `split` field."""
    jsonl = run_dir / "scores.jsonl"
    if jsonl.exists():
        splits: set[str] = set()
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line).get("split")
            if isinstance(value, str):
                splits.add(value)
        return splits
    parquet = run_dir / "scores.parquet"
    if parquet.exists():
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(parquet, columns=["split"])
            return {str(value) for value in table.column("split").to_pylist() if value is not None}
        except Exception:  # pragma: no cover - optional dep / legacy schema drift
            return set()
    return set()


def read_case_series(run_dir: Path, column: str) -> list[float]:
    """Per-case values of one score column (JSONL preferred, Parquet fallback). Missing -> []."""
    jsonl = run_dir / "scores.jsonl"
    if jsonl.exists():
        out: list[float] = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line).get(column)
            if value is not None:
                out.append(float(value))
        return out
    parquet = run_dir / "scores.parquet"
    if parquet.exists():
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(parquet)
            if column not in table.column_names:
                return []
            return [float(v) for v in table.column(column).to_pylist() if v is not None]
        except Exception:  # pragma: no cover - optional dep / schema drift
            return []
    return []


def read_case_objectives(run_dir: Path) -> list[float]:
    """Per-case objective scores for the bootstrap CI (JSONL preferred, Parquet fallback)."""
    return read_case_series(run_dir, "objective_score")


def record_from_manifest(manifest: JsonObject, run_dir: Path) -> RunRecord | None:
    """Build a final-split RunRecord, or None for incomplete/non-leaderboard manifests."""
    config = manifest.get("config") or {}
    model = config.get("model")
    if not model:
        return None
    declared_split = manifest.get("split")
    if declared_split is None:
        legacy_splits = read_case_splits(run_dir)
        if len(legacy_splits) > 1:
            _LOG.warning("[board] mixed splits in legacy run bundle: %s", run_dir)
        if legacy_splits != {FINAL_SPLIT}:
            return None
        split = FINAL_SPLIT
    else:
        split = str(declared_split)
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
        judge_score=_mean_or_none(case_judge),
        semantic_score=_mean_or_none(case_semantic),
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


def load_run_records(run_root: Path | str) -> list[RunRecord]:
    """Load published final-split bundles under `run_root` (skips staging and tune runs)."""
    root = Path(run_root)
    records: list[RunRecord] = []
    if not root.exists():
        return records
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _LOG.warning("[board] unreadable manifest: %s", manifest_path)
            continue
        try:
            record = record_from_manifest(manifest, manifest_path.parent)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            _LOG.warning("[board] unreadable run bundle %s: %s", manifest_path.parent, exc)
            continue
        if record is not None:
            records.append(record)
    return records


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def best_per_model(
    records: list[RunRecord],
    *,
    judge_trusted: bool = False,
    weight_judge: float = DEFAULT_WEIGHT_JUDGE,
) -> list[RunRecord]:
    """Keep the best run per model under the DECLARED ranking policy (headline quality =
    objective, blended with the judge when trusted) -- not objective score alone, so the board's
    per-model pick matches how the board ranks."""

    def score(rec: RunRecord) -> float:
        return headline_quality(rec.result, judge_trusted, weight_judge)

    best: dict[str, RunRecord] = {}
    for rec in records:
        cur = best.get(rec.result.model)
        if cur is None or score(rec) > score(cur):
            best[rec.result.model] = rec
    return list(best.values())


def load_screen_reports(screen_root: Path | str) -> list[ScreenReport]:
    """Load Tier-1 public-screen reports (separate from Tier-2 private bundles; never mixed)."""
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


# --- M5 category boards (each its OWN Tier, never cross-ranked with the RAG board) ----------

# The per-category run-bundle method dirs (under $DATA_DIR/<method>/<ts>/) the board surfaces.
CATEGORY_METHODS = (
    "security",
    "tooling",
    "agentic",
    "summarization",
    "structured",
    "text-analysis",
)


def _category_result(manifest: JsonObject, run_dir: Path) -> ModelResult | None:
    """Build a `ModelResult` from one M5 category run bundle (its config carries the Tier)."""
    config = manifest.get("config") or {}
    tier = config.get("tier")
    model = config.get("model")
    if not tier or not model:
        return None
    metrics = manifest.get("metrics") or {}
    return ModelResult(
        model=str(model),
        backend=str(config.get("backend", "?")),
        objective_score=float(metrics.get("objective_score", 0.0)),
        n_cases=int(manifest.get("n_cases", 0)),
        reliability=float(metrics.get("reliability", 1.0)),
        tokens_per_s=float(metrics.get("tokens_per_s", 0.0)),
        tier=str(tier),
        case_objectives=read_case_objectives(run_dir),  # [] for categories without that column
    )


def load_category_records(data_dir: Path | str) -> dict[str, list[ModelResult]]:
    """Load the M5 category run bundles grouped BY TIER (so each renders on its own board).

    Keeps the best run per model within each tier (highest objective score), mirroring the RAG
    board's best-per-model pick. Never merges tiers -- the `aggregate` guard refuses a mixed board.
    """
    by_tier: dict[str, dict[str, ModelResult]] = {}
    for method in CATEGORY_METHODS:
        root = Path(data_dir) / method
        if not root.exists():
            continue
        for manifest_path in sorted(root.glob("*/manifest.json")):
            if manifest_path.parent.name.startswith("."):
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                result = _category_result(manifest, manifest_path.parent)
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
                _LOG.warning("[board] unreadable category bundle %s: %s", manifest_path.parent, exc)
                continue
            if result is None:
                continue
            best = by_tier.setdefault(result.tier, {})
            current = best.get(result.model)
            if current is None or result.objective_score > current.objective_score:
                best[result.model] = result
    return {tier: list(models.values()) for tier, models in by_tier.items()}
