"""Context-policy benchmark -- rank context-management policies for ONE fixed model.

A chain-of-questions item is a 2-4 step ordered walk where each step's answer depends on the
prior steps. This benchmark holds the model, the verified chain set, and the scoring fixed and
varies ONLY the context-management POLICY -- the row label -- exactly as the agentic
harness comparison holds everything fixed and varies the harness. Four policies:

  - ``fresh``   -- fresh retrieval per step, NO prior-step carryover (the naive baseline);
  - ``history`` -- fresh retrieval PLUS the accumulated full (question, answer) history;
  - ``summary`` -- fresh retrieval PLUS a running model-written summary of the prior steps;
  - ``roles``   -- staged role/system-prompt sequence (librarian -> analyst -> answerer) built
    from prompt-system role templates, PLUS the accumulated history.

Retrieval is reached through an injectable ``Retriever`` (``retrieve(query, k)``) and the model
through the same injectable ``complete`` (prompt -> raw text) every category uses, so the exact
context assembled per policy per step is provable over a FAKE endpoint with no GPU. Each step's
answer is scored objectively against its reference answer (``scoring.correctness`` token-F1);
per-step and final-answer correctness carry bootstrap CIs and the policies are ranked under
``TIER_CHAIN_CONTEXT``.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.bench.common import (
    Mirror,
    mean,
    persist_category_run,
    render_board,
    verified_data_config,
)
from llb.core.contracts.results import BoardRow
from llb.core.contracts.runs import RunMetrics
from llb.goldset.chains import ChainItem
from llb.prompts.registry import render_text
from llb.scoring.aggregate import TIER_CHAIN_CONTEXT
from llb.bench.chain_context_policy import (
    DEFAULT_K,
    POLICY_FRESH,
    POLICY_HISTORY,
    POLICY_ROLES,
    POLICY_SUMMARY,
    PolicyReport,
    RECOMMENDATION_TEMPLATE,
    Retriever,
    prompt_system_ids,
    run_policy,
)

_LOG = logging.getLogger(__name__)

METHOD = "chain-context"

CONTEXT_POLICIES: tuple[str, ...] = (POLICY_FRESH, POLICY_HISTORY, POLICY_SUMMARY, POLICY_ROLES)

# Prompt-system template ids for the role sequence and the default instruction. These are the
# reviewable prompt-system content recorded in provenance as `prompt_system_ids`.


# --- context assembly (pure, unit-tested) -------------------------------------------------


# --- per-policy run ------------------------------------------------------------------------


# --- run + persistence ---------------------------------------------------------------------


@dataclass(slots=True)
class ChainContextRun:
    """Outcome of one context-policy comparison for a fixed model over a verified chain set."""

    model: str
    backend: str
    reports: list[PolicyReport]
    board: list[BoardRow]
    table: str
    recommendation: str
    chain_digest: str


def chain_set_digest(chains: list[ChainItem]) -> str:
    """Stable digest of the chain set content (order-sensitive), recorded in provenance."""
    payload = json.dumps([c.model_dump() for c in chains], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_recommendation(model: str, reports: list[PolicyReport], ranked: list[BoardRow]) -> str:
    """A template-sourced recommendation naming the winning policy and its per-step evidence."""
    by_policy = {r.policy: r for r in reports}
    order = [str(row["model"]) for row in ranked] or [r.policy for r in reports]
    winner = order[0]
    win = by_policy[winner]
    ranking = ", ".join(
        f"{policy} {mean(by_policy[policy].final_objectives):.3f}" for policy in order
    )
    advice = _advice(winner)
    return render_text(
        RECOMMENDATION_TEMPLATE,
        {
            "model": model,
            "winner": winner,
            "winner_final": f"{mean(win.final_objectives):.3f}",
            "winner_step": f"{mean(win.step_objectives):.3f}",
            "ranking": ranking,
            "advice": advice,
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


def _policy_config(
    report: PolicyReport, model: str, backend: str, chain_digest: str, policies: list[str]
) -> dict[str, Any]:
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


def _policy_metrics(report: PolicyReport) -> RunMetrics:
    return {
        "objective_score": round(mean(report.final_objectives), 6),
        "reliability": report.reliability,
        "tokens_per_s": 0.0,
    }


def run_chain_context(
    chains: list[ChainItem],
    *,
    model: str,
    backend: str,
    retriever: Retriever,
    complete: Any,
    policies: list[str] | None = None,
    k: int = DEFAULT_K,
    data_dir: Path | str | None = None,
    run_name: str = "chain-context",
    persist: bool = True,
    mirror: Mirror | None = None,
    data_verified: bool = False,
    verification_ref: str | None = None,
) -> ChainContextRun:
    """Rank context-management policies for one fixed model over a verified chain set.

    Each policy is walked step by step, scored against the step references, and persisted as its
    OWN run bundle under ``$DATA_DIR/chain-context/`` tagged with the policy (mirroring the
    per-harness agentic bundles); the returned board ranks all policies together under
    ``TIER_CHAIN_CONTEXT``. ``data_verified`` follows the category-suite verification-gate rules.
    """
    if not chains:
        raise SystemExit("no chains provided")
    policies = policies or list(CONTEXT_POLICIES)
    unknown = [p for p in policies if p not in CONTEXT_POLICIES]
    if unknown:
        raise SystemExit(f"unknown context policies: {unknown}; choose from {CONTEXT_POLICIES}")
    verification_cfg = verified_data_config(
        data_verified=data_verified, verification_ref=verification_ref
    )
    digest = chain_set_digest(chains)

    reports = [
        run_policy(
            chains,
            policy,
            model=model,
            backend=backend,
            retriever=retriever,
            complete=complete,
            k=k,
        )
        for policy in policies
    ]
    board, table = render_board([r.result for r in reports])
    recommendation = build_recommendation(model, reports, board)

    if persist and data_dir is not None:
        for report in reports:
            config = {
                **_policy_config(report, model, backend, digest, policies),
                **verification_cfg,
            }
            report.paths = persist_category_run(
                method=METHOD,
                data_dir=data_dir,
                run_name=f"{run_name}-{report.policy}",
                config=config,
                metrics=_policy_metrics(report),
                case_rows=report.step_rows,
                mirror=mirror,
            )
        _log_run(model, reports)
    return ChainContextRun(
        model=model,
        backend=backend,
        reports=reports,
        board=board,
        table=table,
        recommendation=recommendation,
        chain_digest=digest,
    )


def _log_run(model: str, reports: list[PolicyReport]) -> None:
    for report in reports:
        manifest = report.paths["manifest"] if report.paths else "(not persisted)"
        _LOG.info(
            "[chain-context] %s policy=%s final=%.3f per-step=%.3f reliability=%.3f -> %s",
            model,
            report.policy,
            mean(report.final_objectives),
            mean(report.step_objectives),
            report.reliability,
            manifest,
        )


def load_chains_file(path: Path | str) -> list[ChainItem]:
    """Load a verified chain set (JSONL) for the benchmark."""
    from llb.goldset.chains import load_chains

    return load_chains(path)
