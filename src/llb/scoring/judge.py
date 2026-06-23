"""Gated local LLM judge using maintained DeepEval metrics and Ukrainian prompts.

The judge is a gated dependency. It contributes to ranking only after calibration against
human ratings clears the Spearman-rho floor. DeepEval is imported lazily and talks to any
OpenAI-compatible local endpoint through its maintained LocalModel adapter.
"""

import os
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, cast
from urllib.parse import urlsplit, urlunsplit

from llb.contracts import JudgeInputRecord, JudgeScore
from llb import env
from llb.paths import load_project_env

DEFAULT_THRESHOLD = 0.6

# Judge-model bias disclosure (OQ2). The v1 default judge is a LOCAL Gemma-4 model, chosen for
# no data egress + reproducibility -- but it is NOT independent of the candidate pool: Gemma-4
# (E4B/12B) are candidates, and MamayLM v2 + Lapa are Gemma-3 fine-tunes, so the judge shares
# architecture / tokenizer / pretraining lineage with most of the pool and may self-prefer
# Gemma-family answers over the non-Gemma ones (Qwen, Llama). It is accepted only because the
# judge is gated and objective correctness keeps weight in the blend.
JUDGE_BIAS_NOTE = (
    "judge is a local Gemma-4 model (not pool-independent): shares lineage with the "
    "Gemma-4 / MamayLM / Lapa candidates -> possible self-preference for Gemma-family "
    "answers; gated by calibration and objective score retains weight; cross-check with a "
    "non-Gemma judge to quantify the family delta"
)

UA_FAITHFULNESS_STEPS = [
    "Виділи всі фактичні твердження з фактичної відповіді.",
    "Для кожного твердження перевір, чи воно безпосередньо підтверджене хоча б одним "
    "фрагментом контексту пошуку.",
    "Знизь оцінку за кожне непідтверджене, суперечливе або вигадане твердження; не використовуй "
    "зовнішні знання.",
    "Найвища оцінка дозволена лише тоді, коли всі фактичні твердження повністю підтверджені "
    "контекстом пошуку.",
]
UA_ANSWER_RELEVANCY_STEPS = [
    "Визнач, яку інформацію прямо запитує вхідне запитання.",
    "Перевір, чи фактична відповідь безпосередньо й по суті відповідає на це запитання.",
    "Знизь оцінку за ухильність, неоднозначність, пропущену ключову інформацію або зайві "
    "відомості, що не допомагають відповісти на запитання.",
    "Не оцінюй правдивість за зовнішніми знаннями: тут оцінюється лише релевантність відповіді "
    "вхідному запитанню.",
]

JudgeEvaluate = Callable[[list[JudgeInputRecord], str], list[dict[str, float]]]


class UkrainianGEvalTemplate:
    """DeepEval G-Eval result prompt with Ukrainian-only judge instructions."""

    _PARAMETER_LABELS = {
        "Actual Output": "Фактична відповідь",
        "Retrieval Context": "Контекст пошуку",
        "Input": "Вхідне запитання",
    }

    @classmethod
    def _localize_parameter_labels(cls, text: str) -> str:
        for english, ukrainian in cls._PARAMETER_LABELS.items():
            text = text.replace(english, ukrainian)
        return text

    @classmethod
    def generate_evaluation_results(
        cls,
        evaluation_steps: str,
        test_case_content: str,
        parameters: str,
        rubric: str | None = None,
        score_range: tuple[int, int] = (0, 10),
        _additional_context: str | None = None,
        multimodal: bool = False,
    ) -> str:
        del rubric, _additional_context, multimodal
        test_case_content = cls._localize_parameter_labels(test_case_content)
        parameters = cls._localize_parameter_labels(parameters)
        return textwrap.dedent(
            f"""\
            Ти оцінювач україномовної RAG-системи. Виконай наведені кроки та оціни тестовий
            приклад цілим числом від {score_range[0]} до {score_range[1]}, де
            {score_range[1]} означає повну відповідність крокам, а {score_range[0]} -- повну
            невідповідність.

            Кроки оцінювання:
            {evaluation_steps}

            Тестовий приклад:
            {test_case_content}

            Параметри, які треба зіставити:
            {parameters}

            Поверни лише коректний JSON без Markdown і додаткового тексту:
            {{"score": <ціле число>, "reason": "стисле обгрунтування українською"}}
            """
        )

    @classmethod
    def generate_strict_evaluation_results(
        cls,
        evaluation_steps: str,
        test_case_content: str,
        parameters: str,
        _additional_context: str | None = None,
        multimodal: bool = False,
    ) -> str:
        return cls.generate_evaluation_results(
            evaluation_steps,
            test_case_content,
            parameters,
            score_range=(0, 1),
            _additional_context=_additional_context,
            multimodal=multimodal,
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
    """Route to the judge only if calibrated and trusted; otherwise demote it."""
    if judge_model is None:
        return JudgeOutcome(trusted=False, reason="no judge configured")
    if calibration_rho is None:
        return JudgeOutcome(trusted=False, reason="judge not calibrated")
    if not judge_is_trusted(calibration_rho, threshold):
        return JudgeOutcome(
            trusted=False,
            reason=f"calibration rho {calibration_rho:.3f} < threshold {threshold}",
        )
    score_fn = scorer or deepeval_scorer
    return JudgeOutcome(trusted=True, reason="calibrated", scores=score_fn(records, judge_model))


def extract_scores(rows: list[dict[str, float]]) -> list[JudgeScore]:
    """Normalize the two judge signals into the canonical score contract."""
    return [
        {
            "faithfulness": float(row.get("faithfulness", 0.0) or 0.0),
            "answer_relevancy": float(row.get("answer_relevancy", 0.0) or 0.0),
        }
        for row in rows
    ]


def deepeval_scorer(
    records: list[JudgeInputRecord],
    judge_model: str,
    *,
    evaluate_fn: JudgeEvaluate | None = None,
    base_url: str | None = None,
) -> list[JudgeScore]:
    """Score faithfulness and answer relevancy with Ukrainian DeepEval G-Eval prompts."""
    if evaluate_fn is not None:
        return extract_scores(evaluate_fn(records, judge_model))
    return _default_deepeval_evaluate(records, judge_model, base_url=base_url)


def _default_deepeval_evaluate(
    records: list[JudgeInputRecord], judge_model: str, *, base_url: str | None = None
) -> list[JudgeScore]:
    served_model, resolved_base_url = resolve_judge_endpoint(judge_model, base_url)
    if resolved_base_url is None:
        raise SystemExit(
            "ERROR: a local judge endpoint is required; set --judge-base-url or "
            f"{env.DEEPEVAL_JUDGE_BASE_URL}."
        )

    os.environ.setdefault(env.DEEPEVAL_TELEMETRY_OPT_OUT, "YES")
    try:
        from deepeval.metrics import GEval
        from deepeval.models import LocalModel
        from deepeval.test_case import LLMTestCase, SingleTurnParams
    except ImportError as exc:
        raise SystemExit(
            'ERROR: the local judge needs the [rag] extra. Run: uv pip install -e ".[rag]"'
        ) from exc

    api_key = os.environ.get(env.DEEPEVAL_JUDGE_API_KEY) or "local"
    model = LocalModel(
        model=served_model,
        base_url=resolved_base_url,
        api_key=api_key,
        temperature=0.0,
        format="json",
    )
    # DeepEval types this as a GEvalTemplate subclass. Keeping our small compatible template
    # independent preserves the lazy optional dependency while the metric engine calls the same
    # two static methods at runtime.
    evaluation_template = cast(Any, UkrainianGEvalTemplate)
    faithfulness = GEval(
        name="UA Faithfulness",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        evaluation_steps=UA_FAITHFULNESS_STEPS,
        model=model,
        async_mode=False,
        evaluation_template=evaluation_template,
        _include_g_eval_suffix=False,
    )
    relevancy = GEval(
        name="UA Answer Relevancy",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        evaluation_steps=UA_ANSWER_RELEVANCY_STEPS,
        model=model,
        async_mode=False,
        evaluation_template=evaluation_template,
        _include_g_eval_suffix=False,
    )

    scores: list[JudgeScore] = []
    for record in records:
        test_case = LLMTestCase(
            input=record["question"],
            actual_output=record["answer"],
            retrieval_context=list(record.get("contexts", [])),
        )
        faith_score = faithfulness.measure(test_case, _show_indicator=False)
        relevancy_score = relevancy.measure(test_case, _show_indicator=False)
        scores.append(
            {
                "faithfulness": float(faith_score),
                "answer_relevancy": float(relevancy_score),
            }
        )
    return scores


def _served_model_id(judge_model: str) -> str:
    prefix, separator, model = judge_model.partition("/")
    if separator and prefix in {"hosted_vllm", "ollama_chat"}:
        return model
    return judge_model


def _judge_base_url_from_prefix(prefix: str) -> str | None:
    if prefix == "hosted_vllm":
        return os.environ.get(env.HOSTED_VLLM_API_BASE) or os.environ.get(env.VLLM_HOST)
    if prefix == "ollama_chat":
        return os.environ.get(env.OLLAMA_API_BASE) or os.environ.get(env.OLLAMA_HOST)
    return None


def _normalize_openai_base_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("judge base URL must be an http(s) URL with a host")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(
            "judge base URL must not contain credentials, query parameters, or a fragment; "
            f"use {env.DEEPEVAL_JUDGE_API_KEY} for authentication"
        )
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def resolve_judge_endpoint(
    judge_model: str, explicit_base_url: str | None = None
) -> tuple[str, str | None]:
    """Resolve legacy local-model prefixes and an OpenAI-compatible endpoint."""
    load_project_env()
    prefix, separator, _model = judge_model.partition("/")
    served_model = _served_model_id(judge_model)
    base_url = explicit_base_url or os.environ.get(env.DEEPEVAL_JUDGE_BASE_URL)
    if base_url is None and separator:
        base_url = _judge_base_url_from_prefix(prefix)
    if base_url is not None:
        base_url = _normalize_openai_base_url(base_url)
    return served_model, base_url


def judge_experiment_metadata(judge_model: str, base_url: str | None = None) -> dict[str, Any]:
    """Non-secret configuration recorded by smoke experiments and run manifests."""
    served_model, resolved_base_url = resolve_judge_endpoint(judge_model, base_url)
    return {
        "provider": "deepeval-geval",
        "model": served_model,
        "base_url": resolved_base_url,
        "prompt_language": "uk",
        "metrics": ["faithfulness", "answer_relevancy"],
    }
