"""Focused chain context policy implementation."""

from dataclasses import dataclass
from typing import Any, Protocol
from llb.bench.common import (
    category_result,
)
from llb.core.contracts import ChunkRecord, RunPaths
from llb.eval.common import classify_response, format_context
from llb.goldset.chains import ChainItem
from llb.prompts.registry import render_text
from llb.scoring.aggregate import TIER_CHAIN_CONTEXT
from llb.scoring.leaderboard import ModelResult, bootstrap_mean_ci
from llb.scoring.correctness import answer_correctness

POLICY_FRESH = "fresh"

POLICY_HISTORY = "history"

POLICY_SUMMARY = "summary"

POLICY_ROLES = "roles"

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
