"""Gated LLM judge (Ragas), with the trust gate as the first-class contract.

Premise 2: the judge is a GATED dependency. It only contributes to the ranking when it
has been calibrated against human ratings and clears the Spearman-rho floor (default
0.6). Below the bar -- or when no judge is configured / calibrated -- it is DEMOTED to a
diagnostic and the objective reference-correctness score carries the ranking alone.

The gate (`judge_is_trusted`, `run_judge` routing) is pure and unit-testable; the actual
Ragas scoring is injected (`scorer=`) and defaults to a lazy `[rag]`-extra implementation,
so CI never imports ragas.
"""

from dataclasses import dataclass
from typing import Any, Callable

from llb.contracts import JudgeInputRecord, JudgeScore

DEFAULT_THRESHOLD = 0.6

# (samples, judge_model) -> per-row {faithfulness, answer_relevancy}. The seam between our
# pure mapping/extraction and the heavy Ragas `evaluate` call, so the scorer is unit-testable.
RagasEvaluate = Callable[[list[dict[str, Any]], str], list[dict[str, float]]]

# UA-localized metric instructions (Premise: the judge reasons in the eval language). Ragas
# ships English metric prompts; we adapt the two we use to Ukrainian so the judge grades UA
# answers natively instead of translating.
UA_FAITHFULNESS_INSTRUCTION = (
    "Оціни ВІРНІСТЬ відповіді наданому контексту: чи кожне твердження відповіді підтверджується "
    "контекстом. Поверни частку підтверджених тверджень від 0 до 1."
)
UA_ANSWER_RELEVANCY_INSTRUCTION = (
    "Оціни РЕЛЕВАНТНІСТЬ відповіді запитанню: наскільки повно й по суті відповідь відповідає на "
    "запитання, без зайвого. Поверни оцінку від 0 до 1."
)


def judge_is_trusted(calibration_rho: float | None, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True only if a calibration rho exists and meets the threshold."""
    return calibration_rho is not None and calibration_rho >= threshold


@dataclass
class JudgeOutcome:
    """Result of attempting to judge a batch of answers."""

    trusted: bool
    reason: str
    scores: list[JudgeScore] | None = None


def run_judge(
    records: list[JudgeInputRecord],
    judge_model: str | None,
    calibration_rho: float | None,
    threshold: float = DEFAULT_THRESHOLD,
    scorer: Callable[[list[JudgeInputRecord], str], list[JudgeScore]] | None = None,
) -> JudgeOutcome:
    """Route to the judge only if gated-in; otherwise return a demoted outcome.

    `records` are dicts with question / answer / contexts / reference. `scorer` defaults
    to the lazy Ragas implementation.
    """
    if judge_model is None:
        return JudgeOutcome(trusted=False, reason="no judge configured")
    if calibration_rho is None:
        return JudgeOutcome(trusted=False, reason="judge not calibrated")
    if not judge_is_trusted(calibration_rho, threshold):
        return JudgeOutcome(
            trusted=False,
            reason=f"calibration rho {calibration_rho:.3f} < threshold {threshold}",
        )
    scorer = scorer or ragas_scorer
    return JudgeOutcome(trusted=True, reason="calibrated", scores=scorer(records, judge_model))


def to_ragas_samples(records: list[JudgeInputRecord]) -> list[dict[str, Any]]:
    """Map our judge records to Ragas SingleTurnSample fields (faithfulness needs the
    retrieved contexts; answer-relevancy needs the question + response)."""
    return [
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": list(r.get("contexts", [])),
        }
        for r in records
    ]


def extract_scores(rows: list[dict[str, float]]) -> list[JudgeScore]:
    """Pull faithfulness + answer-relevancy per row (tolerating either Ragas key spelling)."""
    scores: list[JudgeScore] = []
    for row in rows:
        faith = float(row.get("faithfulness", 0.0) or 0.0)
        rel = float(
            row.get("answer_relevancy", row.get("response_relevancy", 0.0))
            or 0.0  # 0.1 vs 0.2 name
        )
        scores.append({"faithfulness": faith, "answer_relevancy": rel})
    return scores


def ragas_scorer(
    records: list[JudgeInputRecord],
    judge_model: str,
    *,
    evaluate_fn: RagasEvaluate | None = None,
) -> list[JudgeScore]:
    """Score answers with Ragas faithfulness + answer-relevancy (UA-localized prompts).

    The pure mapping (`to_ragas_samples`) and extraction (`extract_scores`) are unit-tested;
    the heavy `evaluate_fn` defaults to the real Ragas run (needs the `[rag]` extra + a judge
    endpoint) and is injectable for tests.
    """
    samples = to_ragas_samples(records)
    rows = (evaluate_fn or _default_ragas_evaluate)(samples, judge_model)
    return extract_scores(rows)


def _default_ragas_evaluate(
    samples: list[dict[str, Any]], judge_model: str
) -> list[dict[str, float]]:
    """Real Ragas evaluation with UA-localized metric prompts. Pending live validation once the
    judge is chosen (OQ2) and calibration ratings exist; the gate keeps it demoted until then."""
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import Faithfulness, ResponseRelevancy
    except ImportError as exc:
        raise SystemExit(
            'ERROR: the Ragas judge needs the [rag] extra. Run: uv pip install -e ".[rag]"'
        ) from exc

    dataset = EvaluationDataset(samples=[SingleTurnSample(**s) for s in samples])
    faithfulness = Faithfulness()
    relevancy = ResponseRelevancy()
    _localize_metric(faithfulness, UA_FAITHFULNESS_INSTRUCTION)
    _localize_metric(relevancy, UA_ANSWER_RELEVANCY_INSTRUCTION)
    result: Any = evaluate(
        dataset=dataset,
        metrics=[faithfulness, relevancy],
        llm=_judge_llm(judge_model),
    )
    return [dict(row) for row in result.to_pandas().to_dict(orient="records")]


def _localize_metric(metric: Any, instruction: str) -> None:
    """Best-effort: prepend a Ukrainian instruction to a Ragas metric's prompt(s)."""
    get_prompts = getattr(metric, "get_prompts", None)
    if not callable(get_prompts):
        return
    for prompt in get_prompts().values():
        if hasattr(prompt, "instruction"):
            prompt.instruction = f"{instruction}\n{prompt.instruction}"


def _judge_llm(judge_model: str) -> Any:
    """Wrap a litellm-served judge model for Ragas."""
    from langchain_community.chat_models import ChatLiteLLM
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatLiteLLM(model=judge_model, temperature=0.0))
