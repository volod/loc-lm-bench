"""Agentic harness comparison loading for the board."""

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

from llb.bench.agentic.model import HARNESS_LOOP
from llb.core.contracts import BoardRow, JsonObject
from llb.scoring.aggregate import (
    TIER_AGENTIC,
    rank_board,
)
from llb.scoring.board_format import format_board, ranking_policy_note
from llb.scoring.leaderboard import ModelResult

from llb.board.categories import AGENTIC_METHOD, category_case_objectives

_LOG = logging.getLogger(__name__)


@dataclass
class HarnessRunRecord:
    """One agentic run tagged by its harness."""

    model: str
    harness: str
    result: ModelResult
    run_dir: str
    created_at: str


def load_agentic_harness_records(data_dir: Path | str) -> list[HarnessRunRecord]:
    """Load agentic run bundles tagged by harness, keeping the best run per model and harness."""
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
        record = harness_record_from_manifest(manifest, manifest_path.parent)
        if record is None:
            continue
        key = (record.model, record.harness)
        current = best.get(key)
        if current is None or record.result.objective_score > current.result.objective_score:
            best[key] = record
    return list(best.values())


def harness_record_from_manifest(manifest: JsonObject, run_dir: Path) -> HarnessRunRecord | None:
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
        case_objectives=category_case_objectives(config, run_dir),
    )
    return HarnessRunRecord(
        model=str(model),
        harness=str(config.get("harness", HARNESS_LOOP)),
        result=result,
        run_dir=str(run_dir),
        created_at=str(manifest.get("created_at", "")),
    )


def harness_comparison(data_dir: Path | str, model: str) -> tuple[list[BoardRow], str, list[str]]:
    """Rank one model's agentic runs across harnesses under the agentic tier."""
    records = [r for r in load_agentic_harness_records(data_dir) if r.model == model]
    if not records:
        return [], "", []
    results = [replace(r.result, model=r.harness) for r in sorted(records, key=lambda r: r.harness)]
    rows = rank_board(results)
    table = format_board(rows, policy=ranking_policy_note(results, judge_trusted=False))
    return rows, table, [r.harness for r in records]
