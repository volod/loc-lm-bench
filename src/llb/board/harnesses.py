"""Agentic harness comparison loading for the board."""

import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from llb.bench.agentic.model import HARNESS_LOOP
from llb.core.contracts.results import BoardRow
from llb.core.contracts.common import JsonObject
from llb.scoring.aggregate import (
    TIER_AGENTIC,
    rank_board,
)
from llb.scoring.board_format import format_board, ranking_policy_note
from llb.scoring.leaderboard import ModelResult

from llb.board.categories import AGENTIC_METHOD, category_case_objectives
from llb.board.io import admitted_manifest

_LOG = logging.getLogger(__name__)


@dataclass
class HarnessRunRecord:
    """One agentic run tagged by its harness."""

    model: str
    harness: str
    result: ModelResult
    mean_max_prompt_tokens: float
    context_policy: str
    context_policy_supported: bool
    mean_steps: float
    run_dir: str
    created_at: str


def load_agentic_harness_records(data_dir: Path | str) -> list[HarnessRunRecord]:
    """Load agentic run bundles tagged by harness + context policy.

    Keeps the best run per (model, harness, context_policy) so a harness comparison never
    silently mixes two context policies on the same axis.
    """
    root = Path(data_dir) / AGENTIC_METHOD
    best: dict[tuple[str, str, str], HarnessRunRecord] = {}
    if not root.exists():
        return []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        manifest = admitted_manifest(manifest_path)
        if manifest is None:
            continue
        record = harness_record_from_manifest(manifest, manifest_path.parent)
        if record is None:
            continue
        key = (record.model, record.harness, record.context_policy)
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
    # Legacy bundles without prompt accounting default to unsupported+unknown so they are never
    # silently read as our `full` policy assembly.
    supported = config.get("context_policy_supported")
    return HarnessRunRecord(
        model=str(model),
        harness=str(config.get("harness", HARNESS_LOOP)),
        result=result,
        mean_max_prompt_tokens=float(config.get("mean_max_prompt_tokens", 0.0)),
        context_policy=str(config.get("context_policy", "unknown")),
        context_policy_supported=bool(supported) if supported is not None else False,
        mean_steps=float(config.get("mean_trajectory_steps", 0.0)),
        run_dir=str(run_dir),
        created_at=str(manifest.get("created_at", "")),
    )


def _select_comparison_policy(
    records: list[HarnessRunRecord], context_policy: str | None
) -> tuple[str, list[HarnessRunRecord]]:
    """Pick one context policy for the harness axis so the comparison varies only the harness.

    Explicit `context_policy` wins. Otherwise prefer the policy covering the most harnesses for
    this model; ties go to the newest `created_at`.
    """
    if context_policy is not None:
        matched = [r for r in records if r.context_policy == context_policy]
        return context_policy, matched
    by_policy: dict[str, list[HarnessRunRecord]] = defaultdict(list)
    for record in records:
        by_policy[record.context_policy].append(record)
    if not by_policy:
        return "unknown", []

    def sort_key(policy: str) -> tuple[int, str]:
        group = by_policy[policy]
        newest = max(r.created_at for r in group)
        return (len(group), newest)

    chosen = max(by_policy, key=sort_key)
    return chosen, by_policy[chosen]


def format_harness_context_table(records: list[HarnessRunRecord]) -> str:
    """Per-harness prompt-size and policy-support appendix beside the ranked board.

    Headline ranking stays completion-only; this table makes the context each harness actually
    sent visible, and names harnesses that could not apply the requested policy.
    """
    header = (
        f"{'harness':<12} {'completion':>10} {'steps':>7} {'prompt-tok':>11} "
        f"{'policy':<16} {'applied':>8}"
    )
    lines = [header, "-" * len(header)]
    for record in sorted(records, key=lambda r: r.harness):
        applied = "yes" if record.context_policy_supported else "no"
        policy = record.context_policy
        if not record.context_policy_supported:
            policy = f"{policy}*"
        lines.append(
            f"{record.harness:<12} "
            f"{record.result.objective_score:>10.3f} "
            f"{record.mean_steps:>7.2f} "
            f"{record.mean_max_prompt_tokens:>11.1f} "
            f"{policy:<16} "
            f"{applied:>8}"
        )
    lines.append(
        "* = harness did not apply the requested context policy (framework-native transcript); "
        "prompt-tok is what that harness actually sent"
    )
    return "\n".join(lines)


def harness_comparison(
    data_dir: Path | str,
    model: str,
    *,
    context_policy: str | None = None,
) -> tuple[list[BoardRow], str, list[str]]:
    """Rank one model's agentic runs across harnesses under ONE fixed context policy."""
    all_records = [r for r in load_agentic_harness_records(data_dir) if r.model == model]
    if not all_records:
        return [], "", []
    policy, records = _select_comparison_policy(all_records, context_policy)
    if not records:
        return [], "", []
    results = [replace(r.result, model=r.harness) for r in sorted(records, key=lambda r: r.harness)]
    rows = rank_board(results)
    table = format_board(rows, policy=ranking_policy_note(results, judge_trusted=False))
    note = f"context-policy={policy} (harness axis only; policy held fixed)"
    table = f"{note}\n{table}\n\n{format_harness_context_table(records)}"
    return rows, table, [r.harness for r in records]
