"""Context-policy comparison loading for the board (mirrors the agentic harness comparison).

Each context-policy run bundle is tagged with its policy; per fixed model, the best bundle per
policy is ranked under ``TIER_CHAIN_CONTEXT`` with the policy as the row label.
"""

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

from llb.bench.chain_context import METHOD
from llb.core.contracts.results import BoardRow
from llb.core.contracts.common import JsonObject
from llb.scoring.aggregate import (
    TIER_CHAIN_CONTEXT,
    rank_board,
)
from llb.scoring.board_format import format_board, ranking_policy_note
from llb.scoring.leaderboard import ModelResult

from llb.board.io import read_case_series

_LOG = logging.getLogger(__name__)


@dataclass
class PolicyRunRecord:
    """One context-policy run tagged by (model, policy)."""

    model: str
    policy: str
    result: ModelResult
    per_step_objective: float
    run_dir: str
    created_at: str


def _final_objectives(run_dir: Path) -> list[float]:
    """The last-step objective per chain -- the headline series behind the policy's CI."""
    steps = read_case_series(run_dir, "objective_score")
    finals = read_case_series(run_dir, "is_final")
    if steps and finals and len(steps) == len(finals):
        return [obj for obj, flag in zip(steps, finals) if flag]
    return steps


def policy_record_from_manifest(manifest: JsonObject, run_dir: Path) -> PolicyRunRecord | None:
    config = manifest.get("config") or {}
    if config.get("tier") != TIER_CHAIN_CONTEXT:
        return None
    model = config.get("model")
    policy = config.get("policy")
    if not model or not policy:
        return None
    metrics = manifest.get("metrics") or {}
    result = ModelResult(
        model=str(model),
        backend=str(config.get("backend", "?")),
        objective_score=float(metrics.get("objective_score", 0.0)),
        n_cases=len(_final_objectives(run_dir)),
        reliability=float(metrics.get("reliability", 1.0)),
        tokens_per_s=float(metrics.get("tokens_per_s", 0.0)),
        tier=TIER_CHAIN_CONTEXT,
        case_objectives=_final_objectives(run_dir),
    )
    return PolicyRunRecord(
        model=str(model),
        policy=str(policy),
        result=result,
        per_step_objective=float(config.get("per_step_objective", 0.0)),
        run_dir=str(run_dir),
        created_at=str(manifest.get("created_at", "")),
    )


def load_chain_context_records(data_dir: Path | str) -> list[PolicyRunRecord]:
    """Load context-policy run bundles, keeping the best run per (model, policy)."""
    root = Path(data_dir) / METHOD
    best: dict[tuple[str, str], PolicyRunRecord] = {}
    if not root.exists():
        return []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _LOG.warning("[board] unreadable chain-context manifest: %s", manifest_path)
            continue
        record = policy_record_from_manifest(manifest, manifest_path.parent)
        if record is None:
            continue
        key = (record.model, record.policy)
        current = best.get(key)
        if current is None or record.result.objective_score > current.result.objective_score:
            best[key] = record
    return list(best.values())


def chain_context_comparison(
    data_dir: Path | str, model: str
) -> tuple[list[BoardRow], str, list[str]]:
    """Rank one model's context-policy runs under the chain-context tier (policy = row label)."""
    records = [r for r in load_chain_context_records(data_dir) if r.model == model]
    if not records:
        return [], "", []
    results = [replace(r.result, model=r.policy) for r in sorted(records, key=lambda r: r.policy)]
    rows = rank_board(results)
    table = format_board(rows, policy=ranking_policy_note(results, judge_trusted=False))
    return rows, table, [r.policy for r in records]
