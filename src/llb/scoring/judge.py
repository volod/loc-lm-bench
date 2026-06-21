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

from llb.config import DEFAULT_EMBEDDING_MODEL
from llb.contracts import JudgeInputRecord, JudgeScore

DEFAULT_THRESHOLD = 0.6

# Judge-model bias disclosure (OQ2). The v1 default judge is a LOCAL Gemma-4 model, chosen for
# no data egress + reproducibility -- but it is NOT independent of the candidate pool: Gemma-4
# (E4B/12B) are candidates, and MamayLM v2 + Lapa are Gemma-3 fine-tunes, so the judge shares
# architecture / tokenizer / pretraining lineage with most of the pool and may self-prefer
# Gemma-family answers over the non-Gemma ones (Qwen, Llama). It is accepted only because the
# judge is GATED (enters ranking only at calibration rho >= threshold, else demoted), objective
# correctness keeps weight in the blend, and the disclosure travels with the board. A non-Gemma
# cross-check judge can quantify the family delta. The model id is NOT hardcoded -- it is set per
# GPU class via config / --judge-model / the Makefile JUDGE_MODEL knob (see current.md).
JUDGE_BIAS_NOTE = (
    "judge is a local Gemma-4 model (not pool-independent): shares lineage with the "
    "Gemma-4 / MamayLM / Lapa candidates -> possible self-preference for Gemma-family "
    "answers; gated by calibration and objective score retains weight; cross-check with a "
    "non-Gemma judge to quantify the family delta"
)

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
        embeddings=_judge_embeddings(),
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
    """Wrap a litellm-served judge model for Ragas (any litellm id -- frontier or a LOCAL
    OpenAI-compatible endpoint, e.g. `hosted_vllm/...` + HOSTED_VLLM_API_BASE or `ollama_chat/...`)."""
    from langchain_community.chat_models import ChatLiteLLM
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatLiteLLM(model=judge_model, temperature=0.0))


def _judge_embeddings(embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    """Local embedding for Ragas answer-relevancy so a LOCAL judge stays fully on-box.

    Ragas otherwise defaults answer-relevancy to OpenAI embeddings -- which both leaks the
    corpus and needs an API key, defeating a local judge. Reuses the pinned RAG embedder
    (Premise 4) so the judge embeds UA text with the same validated model as retrieval.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=embedding_model))
