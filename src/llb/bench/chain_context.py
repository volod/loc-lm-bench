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
from typing import Any, Protocol

from llb.bench.common import (
    Mirror,
    category_result,
    mean,
    persist_category_run,
    render_board,
    verified_data_config,
)
from llb.core.contracts import BoardRow, ChunkRecord, RunMetrics, RunPaths
from llb.eval.common import classify_response, format_context
from llb.goldset.chains import ChainItem
from llb.prompts import render_text
from llb.scoring.aggregate import TIER_CHAIN_CONTEXT, ModelResult, bootstrap_mean_ci
from llb.scoring.correctness import answer_correctness

_LOG = logging.getLogger(__name__)

METHOD = "chain-context"

POLICY_FRESH = "fresh"
POLICY_HISTORY = "history"
POLICY_SUMMARY = "summary"
POLICY_ROLES = "roles"
CONTEXT_POLICIES: tuple[str, ...] = (POLICY_FRESH, POLICY_HISTORY, POLICY_SUMMARY, POLICY_ROLES)

# Prompt-system template ids for the role sequence and the default instruction. These are the
# reviewable prompt-system content recorded in provenance as `prompt_system_ids`.
INSTRUCTION_DEFAULT = "bench.chain_context.instruction_default"
ROLE_LIBRARIAN = "bench.chain_context.role_librarian"
ROLE_ANALYST = "bench.chain_context.role_analyst"
ROLE_ANSWERER = "bench.chain_context.role_answerer"
STEP_TEMPLATE = "bench.chain_context.step"
SUMMARY_TEMPLATE = "bench.chain_context.summary"
RECOMMENDATION_TEMPLATE = "bench.chain_context.recommendation"

DEFAULT_K = 4
_NO_CONTEXT = "(контекст не знайдено)"


class Retriever(Protocol):
    """Anything exposing the store retrieve seam (``rag.store.RagStore``, a graph store, a fake)."""

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]: ...


# --- context assembly (pure, unit-tested) -------------------------------------------------


def role_for_step(index: int, n_steps: int) -> str:
    """Role template id for a step in the ``roles`` sequence: first=librarian, last=answerer,
    middle=analyst. A 2-step chain is librarian -> answerer (no analyst)."""
    if index == 0:
        return ROLE_LIBRARIAN
    if index >= n_steps - 1:
        return ROLE_ANSWERER
    return ROLE_ANALYST


def instruction_for(policy: str, index: int, n_steps: int) -> tuple[str, str]:
    """(template_id, rendered instruction) for a step. ``roles`` rotates the role prompt; every
    other policy uses the one default answerer instruction."""
    template_id = role_for_step(index, n_steps) if policy == POLICY_ROLES else INSTRUCTION_DEFAULT
    return template_id, render_text(template_id)


def format_history(prior_qa: list[tuple[str, str]]) -> str:
    """Render prior (question, answer) pairs as a numbered UA transcript."""
    lines: list[str] = []
    for i, (question, answer) in enumerate(prior_qa, 1):
        lines.append(f"Крок {i} — Питання: {question}")
        lines.append(f"Крок {i} — Відповідь: {answer}")
    return "\n".join(lines)


def prior_block(policy: str, prior_qa: list[tuple[str, str]], running_summary: str) -> str:
    """The prior-context block spliced before the question, per policy. Empty (no carryover) for
    ``fresh`` and for the first step of any policy. Always ends with a blank line so it slots
    cleanly ahead of ``Питання:``."""
    if not prior_qa:
        return ""
    if policy == POLICY_SUMMARY:
        body = running_summary.strip()
        return f"Підсумок попередніх кроків:\n{body}\n\n" if body else ""
    if policy in (POLICY_HISTORY, POLICY_ROLES):
        return f"Попередні кроки:\n{format_history(prior_qa)}\n\n"
    return ""  # fresh: no carryover


def assemble_prompt(
    policy: str,
    *,
    index: int,
    n_steps: int,
    question: str,
    retrieved: list[ChunkRecord],
    prior_qa: list[tuple[str, str]],
    running_summary: str,
) -> tuple[str, str]:
    """Build the EXACT prompt string for one step under one policy.

    Returns ``(instruction_template_id, prompt)``. This is the unit-tested seam: given the same
    retrieved chunks and prior state, each policy assembles a deterministic prompt.
    """
    template_id, instruction = instruction_for(policy, index, n_steps)
    facts = format_context(retrieved) if retrieved else _NO_CONTEXT
    prompt = render_text(
        STEP_TEMPLATE,
        {
            "instruction": instruction,
            "facts": facts,
            "prior_block": prior_block(policy, prior_qa, running_summary),
            "question": question,
        },
    )
    return template_id, prompt


def summarize(complete: Any, prior_qa: list[tuple[str, str]]) -> str:
    """Ask the model for a running summary of the prior steps (the ``summary`` policy's memory)."""
    prompt = render_text(SUMMARY_TEMPLATE, {"prior": format_history(prior_qa)})
    return (complete(prompt) or "").strip()


def prompt_system_ids(policy: str) -> list[str]:
    """The distinct instruction/role prompt-system ids a policy uses, recorded in provenance."""
    if policy == POLICY_ROLES:
        return [ROLE_LIBRARIAN, ROLE_ANALYST, ROLE_ANSWERER]
    return [INSTRUCTION_DEFAULT]


# --- per-policy run ------------------------------------------------------------------------


@dataclass(slots=True)
class PolicyReport:
    """One context policy's scored outcome over the whole chain set."""

    policy: str
    result: ModelResult
    step_rows: list[dict[str, Any]]
    final_objectives: list[float]  # last-step token-F1 per chain (the headline)
    step_objectives: list[float]  # every step's token-F1 (per-step correctness)
    final_ci: tuple[float, float] | None
    step_ci: tuple[float, float] | None
    reliability: float
    paths: RunPaths | None = None


_RELIABILITY_OK = "ok"


def _step_row(
    chain: ChainItem, step_index: int, policy: str, answer: str, template_id: str
) -> dict[str, Any]:
    step = chain.steps[step_index]
    is_final = step_index == len(chain.steps) - 1
    scores = answer_correctness(answer, step.reference_answer)
    status = classify_response(answer, None)
    return {
        "item_id": f"{chain.chain_id}#s{step.order}",
        "chain_id": chain.chain_id,
        "step": step.order,
        "n_steps": len(chain.steps),
        "policy": policy,
        "is_final": 1 if is_final else 0,
        "status": status,
        "objective_score": round(float(scores["score"]), 6),
        "exact": scores["exact"],
        "token_f1": round(float(scores["token_f1"]), 6),
        "contains": scores["contains"],
        "prompt_system_id": template_id,
        "answer_preview": (answer or "")[:280],
    }


def run_policy(
    chains: list[ChainItem],
    policy: str,
    *,
    model: str,
    backend: str,
    retriever: Retriever,
    complete: Any,
    k: int = DEFAULT_K,
) -> PolicyReport:
    """Walk every chain step by step under one policy, scoring each answer against its reference."""
    step_rows: list[dict[str, Any]] = []
    final_objectives: list[float] = []
    for chain in chains:
        prior_qa: list[tuple[str, str]] = []
        running_summary = ""
        for index, step in enumerate(chain.steps):
            retrieved = retriever.retrieve(step.question, k)
            template_id, prompt = assemble_prompt(
                policy,
                index=index,
                n_steps=len(chain.steps),
                question=step.question,
                retrieved=retrieved,
                prior_qa=prior_qa,
                running_summary=running_summary,
            )
            answer = complete(prompt) or ""
            row = _step_row(chain, index, policy, answer, template_id)
            step_rows.append(row)
            if row["is_final"]:
                final_objectives.append(row["objective_score"])
            prior_qa.append((step.question, answer))
            if policy == POLICY_SUMMARY:
                running_summary = summarize(complete, prior_qa)

    step_objectives = [float(r["objective_score"]) for r in step_rows]
    n_ok = sum(1 for r in step_rows if r["status"] == _RELIABILITY_OK)
    reliability = n_ok / len(step_rows) if step_rows else 0.0
    result = category_result(
        model=policy,  # the policy IS the ranked row label (model is fixed)
        backend=backend,
        tier=TIER_CHAIN_CONTEXT,
        case_objectives=final_objectives,  # headline = final-answer correctness per chain
        reliability=reliability,
    )
    return PolicyReport(
        policy=policy,
        result=result,
        step_rows=step_rows,
        final_objectives=final_objectives,
        step_objectives=step_objectives,
        final_ci=bootstrap_mean_ci(final_objectives),
        step_ci=bootstrap_mean_ci(step_objectives),
        reliability=reliability,
    )


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
