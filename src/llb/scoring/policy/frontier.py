"""Frontier judge lane: litellm completion guarded by the cost ledger."""

import json
import logging
import re
from collections.abc import Callable

from llb.core.contracts.judging import JudgeInputRecord, JudgeScore
from llb.prep.frontier_telemetry import LLMComplete
from llb.scoring.judge.scorer import extract_scores
from llb.scoring.judge.template import UA_ANSWER_RELEVANCY_STEPS, UA_FAITHFULNESS_STEPS
from llb.scoring.policy.errors import BudgetExceeded
from llb.scoring.policy.ledger import CostLedger, LedgerEntry

_LOG = logging.getLogger(__name__)
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

FrontierComplete = Callable[[str], tuple[str, float, int, int]]
"""Return (text, cost_usd, prompt_tokens, completion_tokens)."""


def build_frontier_judge_prompt(record: JudgeInputRecord) -> str:
    """Compose a Ukrainian judge prompt from the registered G-Eval step lists."""
    context_block = "\n".join(f"[{index}] {text}" for index, text in enumerate(record["contexts"]))
    faithfulness = "\n".join(f"- {step}" for step in UA_FAITHFULNESS_STEPS)
    relevancy = "\n".join(f"- {step}" for step in UA_ANSWER_RELEVANCY_STEPS)
    return (
        "Оціни відповідь моделі за двома критеріями і поверни ТІЛЬКИ JSON:\n"
        '{"faithfulness": <0..1>, "answer_relevancy": <0..1>}\n\n'
        f"Критерії вірності (faithfulness):\n{faithfulness}\n\n"
        f"Критерії релевантності (answer_relevancy):\n{relevancy}\n\n"
        f"Питання: {record['question']}\n"
        f"Відповідь: {record['answer']}\n"
        f"Контекст:\n{context_block}\n"
    )


def parse_frontier_judge_response(text: str) -> JudgeScore:
    """Parse the frontier judge JSON into the canonical score contract."""
    match = _JSON_BLOCK.search(text)
    if match is None:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}
    if not isinstance(payload, dict):
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}
    return extract_scores([payload])[0]


def litellm_frontier_complete(model: str, temperature: float = 0.0) -> FrontierComplete:
    """Default frontier completer via litellm (needs the ``[prep]`` extra + provider key)."""

    def complete(prompt: str) -> tuple[str, float, int, int]:
        from litellm import completion, completion_cost

        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        usage = response.get("usage", {}) or {}
        try:
            cost = float(completion_cost(response))
        except Exception:
            cost = 0.0
        text = str(response["choices"][0]["message"]["content"])
        return (
            text,
            cost,
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )

    return complete


def wrap_llm_complete(complete: LLMComplete, *, cost_usd: float = 0.0) -> FrontierComplete:
    """Adapt a text-only completer (tests / injected fakes) into a FrontierComplete."""

    def wrapped(prompt: str) -> tuple[str, float, int, int]:
        return complete(prompt), cost_usd, 0, 0

    return wrapped


def frontier_scorer(
    model: str,
    ledger: CostLedger,
    *,
    complete: FrontierComplete | None = None,
) -> Callable[[list[JudgeInputRecord], str], list[JudgeScore]]:
    """Build a JudgeScorer that scores one record at a time under the budget cap."""
    completer = complete or litellm_frontier_complete(model)

    def score(records: list[JudgeInputRecord], judge_model: str) -> list[JudgeScore]:
        del judge_model  # the ledger-bound model is authoritative
        scores: list[JudgeScore] = []
        for index, record in enumerate(records):
            if not str(record.get("answer", "")).strip():
                scores.append({"faithfulness": 0.0, "answer_relevancy": 0.0})
                continue
            scores.append(_score_one(completer, ledger, model, record, index))
        return scores

    return score


def _score_one(
    completer: FrontierComplete,
    ledger: CostLedger,
    model: str,
    record: JudgeInputRecord,
    case_index: int,
) -> JudgeScore:
    ledger.reserve_call()
    try:
        text, cost_usd, prompt_tokens, completion_tokens = completer(
            build_frontier_judge_prompt(record)
        )
    except BudgetExceeded:
        raise
    except Exception as exc:
        ledger.record(
            LedgerEntry(
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                case_index=case_index,
                error=str(exc),
            )
        )
        _LOG.warning("[scorer-policy] frontier judge call failed: %s", exc)
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}
    ledger.record(
        LedgerEntry(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            case_index=case_index,
        )
    )
    return parse_frontier_judge_response(text)
