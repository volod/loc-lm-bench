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
from dataclasses import dataclass, replace
from pathlib import Path

from llb.bench.agentic import HARNESS_LOOP
from llb.contracts import BoardRow, JsonObject, ScreenReport
from llb.scoring.aggregate import (
    DEFAULT_WEIGHT_JUDGE,
    TIER_AGENTIC,
    ModelResult,
    format_board,
    headline_quality,
    rank_board,
    ranking_policy_note,
)
from llb.scoring.composite import CompositeComponent, CompositeIssue, build_m5_composite_rows

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


@dataclass
class CategoryRunRecord:
    result: ModelResult
    config: JsonObject
    run_dir: str
    created_at: str
    data_verified: bool
    verification_ref: str | None
    verification_error: str | None = None


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

CATEGORY_OBJECTIVE_COLUMNS: dict[str, tuple[str, ...]] = {
    "security": ("objective_score", "defended"),
    "tooling": ("objective_score", "correct"),
    "agentic": ("objective_score", "success"),
    "summarization": ("objective_score", "coverage"),
    "structured": ("objective_score", "score"),
    "text_analysis": ("objective_score",),
    "text-analysis": ("objective_score",),
}


def _category_case_objectives(config: JsonObject, run_dir: Path) -> list[float]:
    category = str(config.get("category", ""))
    columns = CATEGORY_OBJECTIVE_COLUMNS.get(category, ("objective_score",))
    for column in columns:
        values = read_case_series(run_dir, column)
        if values:
            return values
    return []


def _category_record(manifest: JsonObject, run_dir: Path) -> CategoryRunRecord | None:
    """Build a category run record from one M5 run bundle (its config carries the Tier)."""
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
        case_objectives=_category_case_objectives(config, run_dir),
    )
    verification_ref = config.get("verification_ref")
    verification_error: str | None = None
    data_verified = bool(config.get("data_verified", False))
    if data_verified:
        if not verification_ref:
            verification_error = "missing verification_ref"
        else:
            from llb.goldset.verify import check_verification_ref

            status = check_verification_ref(str(verification_ref), base_dir=run_dir)
            if not status.valid:
                verification_error = status.reason
    return CategoryRunRecord(
        result=result,
        config=config,
        run_dir=str(run_dir),
        created_at=str(manifest.get("created_at", "")),
        data_verified=data_verified,
        verification_ref=str(verification_ref) if verification_ref else None,
        verification_error=verification_error,
    )


def load_category_run_records(data_dir: Path | str) -> dict[str, list[CategoryRunRecord]]:
    """Load M5 category run bundles grouped BY TIER, preserving config/gate metadata.

    Keeps the best run per model within each tier (highest objective score), mirroring the RAG
    board's best-per-model pick. Never merges tiers -- the `aggregate` guard refuses a mixed board.
    """
    by_tier: dict[str, dict[str, CategoryRunRecord]] = {}
    for method in CATEGORY_METHODS:
        root = Path(data_dir) / method
        if not root.exists():
            continue
        for manifest_path in sorted(root.glob("*/manifest.json")):
            if manifest_path.parent.name.startswith("."):
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                record = _category_record(manifest, manifest_path.parent)
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
                _LOG.warning("[board] unreadable category bundle %s: %s", manifest_path.parent, exc)
                continue
            if record is None:
                continue
            best = by_tier.setdefault(record.result.tier, {})
            current = best.get(record.result.model)
            if current is None or record.result.objective_score > current.result.objective_score:
                best[record.result.model] = record
    return {tier: list(models.values()) for tier, models in by_tier.items()}


def load_category_records(data_dir: Path | str) -> dict[str, list[ModelResult]]:
    """Load the M5 category run bundles grouped BY TIER (so each renders on its own board)."""
    return {
        tier: [record.result for record in records]
        for tier, records in load_category_run_records(data_dir).items()
    }


def load_m5_composite(
    data_dir: Path | str,
    *,
    require_verified: bool = True,
    require_ci: bool = True,
) -> tuple[list[JsonObject], list[CompositeIssue]]:
    """Load the guarded M5 composite headline from persisted category runs."""
    records_by_tier = load_category_run_records(data_dir)
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
        for tier, records in records_by_tier.items()
    }
    return build_m5_composite_rows(
        components_by_tier,
        require_verified=require_verified,
        require_ci=require_ci,
    )


# --- M7.1 agentic harness comparison (LangGraph vs CrewAI, one model, TIER_AGENTIC) ----------

AGENTIC_METHOD = "agentic"


@dataclass
class HarnessRunRecord:
    """One agentic run tagged by its harness (the M7.1 comparison axis)."""

    model: str
    harness: str
    result: ModelResult  # tier=TIER_AGENTIC, with the per-case objective series for CIs
    run_dir: str
    created_at: str


def load_agentic_harness_records(data_dir: Path | str) -> list[HarnessRunRecord]:
    """Load agentic run bundles tagged BY HARNESS, keeping the best run per (model, harness).

    The harness is never silently mixed into the per-model best pick (that would hide the very
    effect we measure): records stay keyed by (model, harness) so a comparison can rank one model
    across `{loop, langgraph, crewai}`."""
    root = Path(data_dir) / AGENTIC_METHOD
    best: dict[tuple[str, str], HarnessRunRecord] = {}
    if not root.exists():
        return []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _LOG.warning("[board] unreadable agentic manifest: %s", manifest_path)
            continue
        record = _harness_record(manifest, manifest_path.parent)
        if record is None:
            continue
        key = (record.model, record.harness)
        current = best.get(key)
        if current is None or record.result.objective_score > current.result.objective_score:
            best[key] = record
    return list(best.values())


def _harness_record(manifest: JsonObject, run_dir: Path) -> HarnessRunRecord | None:
    config = manifest.get("config") or {}
    if config.get("tier") != TIER_AGENTIC:
        return None
    model = config.get("model")
    if not model:
        return None
    metrics = manifest.get("metrics") or {}
    result = ModelResult(
        model=str(model),
        backend=str(config.get("backend", "?")),
        objective_score=float(metrics.get("objective_score", 0.0)),
        n_cases=int(manifest.get("n_cases", 0)),
        reliability=float(metrics.get("reliability", 1.0)),
        tokens_per_s=float(metrics.get("tokens_per_s", 0.0)),
        tier=TIER_AGENTIC,
        case_objectives=_category_case_objectives(config, run_dir),
    )
    return HarnessRunRecord(
        model=str(model),
        harness=str(config.get("harness", HARNESS_LOOP)),
        result=result,
        run_dir=str(run_dir),
        created_at=str(manifest.get("created_at", "")),
    )


def harness_comparison(data_dir: Path | str, model: str) -> tuple[list[BoardRow], str, list[str]]:
    """Rank ONE model's agentic runs across its harnesses under TIER_AGENTIC.

    Each harness becomes its own board row (the `model` cell carries the harness id, the candidate
    model being held fixed), so the existing average-rank + bootstrap-CI ranker compares harness
    effect without ever cross-ranking models or tiers. Returns (rows, ascii table, harness ids)."""
    records = [r for r in load_agentic_harness_records(data_dir) if r.model == model]
    if not records:
        return [], "", []
    results = [replace(r.result, model=r.harness) for r in sorted(records, key=lambda r: r.harness)]
    rows = rank_board(results)
    table = format_board(rows, policy=ranking_policy_note(results, judge_trusted=False))
    return rows, table, [r.harness for r in records]


# --- M7.3 prompt-system comparison ---------------------------------------------------------


@dataclass
class PromptSystemRunRecord:
    """One agentic run tagged by its prompt-system id (the M7.3 comparison axis)."""

    model: str
    harness: str
    prompt_system: str
    result: ModelResult
    run_dir: str


def load_prompt_system_records(data_dir: Path | str) -> list[PromptSystemRunRecord]:
    """Load agentic run bundles that carry a prompt-system id, best per (model, harness, system)."""
    root = Path(data_dir) / AGENTIC_METHOD
    best: dict[tuple[str, str, str], PromptSystemRunRecord] = {}
    if not root.exists():
        return []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        config = manifest.get("config") or {}
        prompt_system = config.get("prompt_system")
        if config.get("tier") != TIER_AGENTIC or not config.get("model") or not prompt_system:
            continue
        record = _harness_record(manifest, manifest_path.parent)
        if record is None:
            continue
        key = (record.model, record.harness, str(prompt_system))
        current = best.get(key)
        if current is None or record.result.objective_score > current.result.objective_score:
            best[key] = PromptSystemRunRecord(
                model=record.model,
                harness=record.harness,
                prompt_system=str(prompt_system),
                result=record.result,
                run_dir=record.run_dir,
            )
    return list(best.values())


def prompt_system_comparison(
    data_dir: Path | str, model: str, harness: str | None = None
) -> tuple[list[BoardRow], str, list[str]]:
    """Rank ONE model (optionally one harness) across prompt-system ids under TIER_AGENTIC.

    Each prompt-system id becomes its own board row (the `model` cell carries the id), so the
    average-rank + bootstrap-CI ranker isolates whether the additional prompt system helps -- without
    mixing prompt systems or cross-ranking models. Returns (rows, ascii table, prompt-system ids)."""
    records = [
        r
        for r in load_prompt_system_records(data_dir)
        if r.model == model and (harness is None or r.harness == harness)
    ]
    if not records:
        return [], "", []
    ordered = sorted(records, key=lambda r: r.prompt_system)
    results = [replace(r.result, model=r.prompt_system) for r in ordered]
    rows = rank_board(results)
    table = format_board(rows, policy=ranking_policy_note(results, judge_trusted=False))
    return rows, table, [r.prompt_system for r in ordered]


@dataclass
class RagPromptSystemRunRecord:
    """One final-split RAG run tagged by prompt-system id."""

    model: str
    prompt_system: str
    result: ModelResult
    run_dir: str


def load_rag_prompt_system_records(data_dir: Path | str) -> list[RagPromptSystemRunRecord]:
    """Load baseline `run-eval` bundles tagged with prompt-system provenance."""
    root = Path(data_dir) / "run-eval"
    best: dict[tuple[str, str], RagPromptSystemRunRecord] = {}
    if not root.exists():
        return []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        provenance = manifest.get("prompt_system_provenance") or {}
        if not isinstance(provenance, dict):
            continue
        prompt_system = provenance.get("prompt_system_id")
        if not prompt_system:
            continue
        record = record_from_manifest(manifest, manifest_path.parent)
        if record is None:
            continue
        key = (record.result.model, str(prompt_system))
        current = best.get(key)
        if current is None or record.result.objective_score > current.result.objective_score:
            best[key] = RagPromptSystemRunRecord(
                model=record.result.model,
                prompt_system=str(prompt_system),
                result=record.result,
                run_dir=record.run_dir,
            )
    return list(best.values())


def rag_prompt_system_comparison(
    data_dir: Path | str, model: str
) -> tuple[list[BoardRow], str, list[str]]:
    """Rank ONE baseline RAG model across prompt-system ids."""
    records = [r for r in load_rag_prompt_system_records(data_dir) if r.model == model]
    if not records:
        return [], "", []
    ordered = sorted(records, key=lambda r: r.prompt_system)
    results = [replace(r.result, model=r.prompt_system) for r in ordered]
    rows = rank_board(results)
    table = format_board(rows, policy=ranking_policy_note(results, judge_trusted=False))
    return rows, table, [r.prompt_system for r in ordered]
