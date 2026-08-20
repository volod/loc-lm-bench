"""Agent context-policy comparison loading for the board (mirrors `board/chain_context.py`).

Each context-policy run bundle is tagged with its policy; per fixed model, the best bundle per
policy is ranked under `TIER_AGENTIC` with the policy as the row label. The bundles live under
their own `agentic-context` method root, so they are never mixed into the harness comparison or
into the category composite, both of which read the `agentic` root.
"""

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

from llb.bench.context_policy.report import METHOD
from llb.board.io import read_case_series
from llb.core.contracts.common import JsonObject
from llb.core.contracts.results import BoardRow
from llb.scoring.aggregate import TIER_AGENTIC, rank_board
from llb.scoring.board_format import format_board, ranking_policy_note
from llb.scoring.leaderboard import ModelResult

_LOG = logging.getLogger(__name__)


@dataclass
class ContextPolicyRunRecord:
    """One agent context-policy run tagged by (model, policy)."""

    model: str
    policy: str
    result: ModelResult
    mean_max_prompt_tokens: float
    n_context_overflow: int
    run_dir: str
    created_at: str


def policy_record_from_manifest(
    manifest: JsonObject, run_dir: Path
) -> ContextPolicyRunRecord | None:
    config = manifest.get("config") or {}
    if config.get("category") != METHOD:
        return None
    model, policy = config.get("model"), config.get("policy")
    if not model or not policy:
        return None
    metrics = manifest.get("metrics") or {}
    case_success = read_case_series(run_dir, "success") or read_case_series(
        run_dir, "objective_score"
    )
    result = ModelResult(
        model=str(model),
        backend=str(config.get("backend", "?")),
        objective_score=float(metrics.get("objective_score", 0.0)),
        n_cases=len(case_success),
        reliability=float(metrics.get("reliability", 1.0)),
        tokens_per_s=float(metrics.get("tokens_per_s", 0.0)),
        tier=TIER_AGENTIC,
        case_objectives=case_success,
    )
    return ContextPolicyRunRecord(
        model=str(model),
        policy=str(policy),
        result=result,
        mean_max_prompt_tokens=float(config.get("mean_max_prompt_tokens", 0.0)),
        n_context_overflow=int(config.get("n_context_overflow", 0)),
        run_dir=str(run_dir),
        created_at=str(manifest.get("created_at", "")),
    )


def load_agentic_context_records(data_dir: Path | str) -> list[ContextPolicyRunRecord]:
    """Load context-policy run bundles, keeping the best run per (model, policy)."""
    root = Path(data_dir) / METHOD
    best: dict[tuple[str, str], ContextPolicyRunRecord] = {}
    if not root.exists():
        return []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _LOG.warning("[board] unreadable agentic-context manifest: %s", manifest_path)
            continue
        record = policy_record_from_manifest(manifest, manifest_path.parent)
        if record is None:
            continue
        key = (record.model, record.policy)
        current = best.get(key)
        if current is None or record.result.objective_score > current.result.objective_score:
            best[key] = record
    return list(best.values())


def agentic_context_comparison(
    data_dir: Path | str, model: str
) -> tuple[list[BoardRow], str, list[str]]:
    """Rank one model's context-policy runs under the agentic tier (policy = row label)."""
    records = [r for r in load_agentic_context_records(data_dir) if r.model == model]
    if not records:
        return [], "", []
    ordered = sorted(records, key=lambda r: r.policy)
    results = [replace(r.result, model=r.policy) for r in ordered]
    rows = rank_board(results)
    table = format_board(rows, policy=ranking_policy_note(results, judge_trusted=False))
    return rows, table, [r.policy for r in ordered]
