"""Board data loading (M3.7) -- pure, so the Streamlit page stays a thin view.

Reads the canonical run bundles under ``$DATA_DIR/run-eval/<ts>/`` (the immutable
`manifest.json` + per-case `scores`) into the `ModelResult` rows the M3.6 ranker consumes,
keeps the BEST config per model, and remembers each row's config + run dir so the page can
show "best config per model". No Streamlit here -- this half is unit-tested.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from llb.contracts import JsonObject
from llb.scoring.aggregate import ModelResult

_LOG = logging.getLogger(__name__)

# The config knobs shown as "best config per model" on the board.
CONFIG_KEYS = ("strategy", "chunk_size", "chunk_overlap", "top_k", "retrieval_mode")


@dataclass
class RunRecord:
    result: ModelResult
    config: JsonObject
    run_dir: str
    created_at: str


def read_case_objectives(run_dir: Path) -> list[float]:
    """Per-case objective scores for the bootstrap CI (JSONL preferred, Parquet fallback)."""
    jsonl = run_dir / "scores.jsonl"
    if jsonl.exists():
        out: list[float] = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(float(json.loads(line).get("objective_score", 0.0)))
        return out
    parquet = run_dir / "scores.parquet"
    if parquet.exists():
        try:
            import pyarrow.parquet as pq

            col = pq.read_table(parquet, columns=["objective_score"]).column("objective_score")
            return [float(v) for v in col.to_pylist()]
        except Exception:  # pragma: no cover - optional dep / schema drift
            return []
    return []


def record_from_manifest(manifest: JsonObject, run_dir: Path) -> RunRecord | None:
    """Build one RunRecord from a parsed manifest.json; None if it lacks a model/config."""
    config = manifest.get("config") or {}
    model = config.get("model")
    if not model:
        return None
    metrics = manifest.get("metrics") or {}
    telemetry = manifest.get("telemetry") or {}
    result = ModelResult(
        model=str(model),
        backend=str(config.get("backend", "?")),
        objective_score=float(metrics.get("objective_score", 0.0)),
        n_cases=int(manifest.get("n_cases", 0)),
        reliability=float(metrics.get("reliability", 1.0)),
        tokens_per_s=float(metrics.get("tokens_per_s", 0.0)),
        peak_vram_mb=telemetry.get("peak_vram_mb"),
        case_objectives=read_case_objectives(run_dir),
    )
    return RunRecord(
        result=result,
        config=config,
        run_dir=str(run_dir),
        created_at=str(manifest.get("created_at", "")),
    )


def load_run_records(run_root: Path | str) -> list[RunRecord]:
    """Load every published run bundle under `run_root` (skips staging `.tmp` dirs)."""
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
        record = record_from_manifest(manifest, manifest_path.parent)
        if record is not None:
            records.append(record)
    return records


def best_per_model(records: list[RunRecord]) -> list[RunRecord]:
    """Keep the highest-objective run per model (the leaderboard is per-model best config)."""
    best: dict[str, RunRecord] = {}
    for rec in records:
        cur = best.get(rec.result.model)
        if cur is None or rec.result.objective_score > cur.result.objective_score:
            best[rec.result.model] = rec
    return list(best.values())


def config_summary(config: JsonObject) -> dict[str, object]:
    """The subset of a config shown as the model's best configuration."""
    return {key: config.get(key) for key in CONFIG_KEYS}
