"""Result models and report helpers for the chain-context benchmark."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from llb.bench.chain_context_policy import (
    POLICY_HISTORY,
    POLICY_ROLES,
    POLICY_SUMMARY,
    PolicyReport,
    RECOMMENDATION_TEMPLATE,
    prompt_system_ids,
)
from llb.bench.common import mean
from llb.core.contracts.results import BoardRow
from llb.core.contracts.runs import RunMetrics
from llb.goldset.chains import ChainItem
from llb.prompts.registry import render_text
from llb.scoring.aggregate import TIER_CHAIN_CONTEXT

METHOD = "chain-context"


@dataclass(slots=True)
class ChainContextRun:
    """Outcome of one context-policy comparison for a fixed model."""

    model: str
    backend: str
    reports: list[PolicyReport]
    board: list[BoardRow]
    table: str
    recommendation: str
    chain_digest: str


def chain_set_digest(chains: list[ChainItem]) -> str:
    """Return an order-sensitive digest of the chain-set content."""
    payload = json.dumps(
        [chain.model_dump() for chain in chains], ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_recommendation(model: str, reports: list[PolicyReport], ranked: list[BoardRow]) -> str:
    """Render a recommendation naming the winning policy and its evidence."""
    by_policy = {report.policy: report for report in reports}
    order = [str(row["model"]) for row in ranked] or [report.policy for report in reports]
    winner = order[0]
    winning_report = by_policy[winner]
    ranking = ", ".join(
        f"{policy} {mean(by_policy[policy].final_objectives):.3f}" for policy in order
    )
    return render_text(
        RECOMMENDATION_TEMPLATE,
        {
            "model": model,
            "winner": winner,
            "winner_final": f"{mean(winning_report.final_objectives):.3f}",
            "winner_step": f"{mean(winning_report.step_objectives):.3f}",
            "ranking": ranking,
            "advice": _advice(winner),
        },
    )


def _advice(winner: str) -> str:
    if winner == POLICY_ROLES:
        return (
            "послідовність ролей librarian -> analyst -> answerer перемагає: розкладай крок "
            "на пошук факту, встановлення зв'язку та остаточну відповідь окремими системними промптами"
        )
    if winner == POLICY_HISTORY:
        return "накопичуй повну історію попередніх кроків у промпті наступного кроку"
    if winner == POLICY_SUMMARY:
        return "стискай історію попередніх кроків у короткий підсумок перед наступним кроком"
    return "свіжий пошук на кожному кроці без переносу історії достатній для цього набору"


def policy_config(
    report: PolicyReport, model: str, backend: str, chain_digest: str, policies: list[str]
) -> dict[str, Any]:
    """Build one persisted policy's provenance configuration."""
    return {
        "model": model,
        "backend": backend,
        "tier": TIER_CHAIN_CONTEXT,
        "category": METHOD,
        "policy": report.policy,
        "policies": policies,
        "chain_set_digest": chain_digest,
        "prompt_system_ids": prompt_system_ids(report.policy),
        "n_chains": len(report.final_objectives),
        "n_steps": len(report.step_rows),
        "final_objective": round(mean(report.final_objectives), 6),
        "per_step_objective": round(mean(report.step_objectives), 6),
        "final_ci": list(report.final_ci) if report.final_ci else None,
        "per_step_ci": list(report.step_ci) if report.step_ci else None,
    }


def policy_metrics(report: PolicyReport) -> RunMetrics:
    """Project a policy report into the common persisted metric contract."""
    return {
        "objective_score": round(mean(report.final_objectives), 6),
        "reliability": report.reliability,
        "tokens_per_s": 0.0,
    }
