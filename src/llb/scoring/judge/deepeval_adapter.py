"""Lazy DeepEval G-Eval adapter for local Ukrainian judges."""

import logging
import os
from typing import Any

from llb.core import env
from llb.core.contracts.judging import JudgeInputRecord, JudgeScore
from llb.core.paths import resolve_data_dir
from llb.scoring.judge.endpoint import resolve_judge_endpoint
from llb.scoring.judge.template import (
    UA_ANSWER_RELEVANCY_STEPS,
    UA_FAITHFULNESS_STEPS,
    UkrainianGEvalTemplate,
)

_LOG = logging.getLogger(__name__)
_UKRAINIAN_GEVAL_CACHE: dict[type[Any], type[Any]] = {}


def _isolate_deepeval_artifacts() -> None:
    cache_root = resolve_data_dir() / "cache" / "deepeval"
    os.environ.setdefault(env.DEEPEVAL_CACHE_FOLDER, str(cache_root))
    os.environ.setdefault(env.DEEPEVAL_RESULTS_FOLDER, str(cache_root / "results"))


def _ukrainian_geval_class(geval_cls: type[Any]) -> type[Any]:
    cached = _UKRAINIAN_GEVAL_CACHE.get(geval_cls)
    if cached is not None:
        return cached

    class _UkrainianGEval(geval_cls):
        def _get_prompt(
            self,
            method: str,
            *,
            template_class: str | None = None,
            multimodal: bool = False,
            strict: bool = True,
            **kwargs: Any,
        ) -> str:
            if method == "generate_evaluation_results":
                return UkrainianGEvalTemplate.generate_evaluation_results(
                    multimodal=multimodal, **kwargs
                )
            if method == "generate_strict_evaluation_results":
                return UkrainianGEvalTemplate.generate_strict_evaluation_results(
                    multimodal=multimodal, **kwargs
                )
            return super()._get_prompt(  # type: ignore[no-any-return]
                method,
                template_class=template_class,
                multimodal=multimodal,
                strict=strict,
                **kwargs,
            )

    _UKRAINIAN_GEVAL_CACHE[geval_cls] = _UkrainianGEval
    return _UkrainianGEval


def _metrics(geval_cls: type[Any], model: Any, params: Any) -> tuple[Any, Any]:
    ua_geval = _ukrainian_geval_class(geval_cls)
    faithfulness = ua_geval(
        name="UA Faithfulness",
        evaluation_params=[params.ACTUAL_OUTPUT, params.RETRIEVAL_CONTEXT],
        evaluation_steps=UA_FAITHFULNESS_STEPS,
        model=model,
        async_mode=False,
        _include_g_eval_suffix=False,
    )
    relevancy = ua_geval(
        name="UA Answer Relevancy",
        evaluation_params=[params.INPUT, params.ACTUAL_OUTPUT],
        evaluation_steps=UA_ANSWER_RELEVANCY_STEPS,
        model=model,
        async_mode=False,
        _include_g_eval_suffix=False,
    )
    return faithfulness, relevancy


def default_deepeval_evaluate(
    records: list[JudgeInputRecord],
    judge_model: str,
    *,
    base_url: str | None = None,
    diagnostics_out: list[str | None] | None = None,
) -> list[JudgeScore]:
    """Run the optional DeepEval dependency against a local OpenAI-compatible endpoint."""
    served_model, resolved_base_url = resolve_judge_endpoint(judge_model, base_url)
    if resolved_base_url is None:
        raise SystemExit(
            "ERROR: a local judge endpoint is required; set --judge-base-url or "
            f"{env.DEEPEVAL_JUDGE_BASE_URL}."
        )
    os.environ.setdefault(env.DEEPEVAL_TELEMETRY_OPT_OUT, "YES")
    _isolate_deepeval_artifacts()
    try:
        from deepeval.metrics import GEval
        from deepeval.models import LocalModel
        from deepeval.test_case import LLMTestCase, SingleTurnParams
    except ImportError as exc:
        raise SystemExit(
            'ERROR: the local judge needs the [rag] extra. Run: uv pip install -e ".[rag]"'
        ) from exc

    model = LocalModel(
        model=served_model,
        base_url=resolved_base_url,
        api_key=os.environ.get(env.DEEPEVAL_JUDGE_API_KEY) or "local",
        temperature=0.0,
        format="json",
    )
    faithfulness, relevancy = _metrics(GEval, model, SingleTurnParams)
    scores: list[JudgeScore] = []
    failures: dict[int, str] = {}
    for index, record in enumerate(records):
        test_case = LLMTestCase(
            input=record["question"],
            actual_output=record["answer"],
            retrieval_context=list(record.get("contexts", [])),
        )
        scores.append(
            {
                "faithfulness": measure_judge_metric(
                    faithfulness, test_case, "faithfulness", index, failures
                ),
                "answer_relevancy": measure_judge_metric(
                    relevancy, test_case, "answer_relevancy", index, failures
                ),
            }
        )
    if diagnostics_out is not None:
        diagnostics_out.extend(failures.get(index) for index in range(len(records)))
    return scores


def measure_judge_metric(
    metric: Any,
    test_case: Any,
    metric_name: str,
    record_index: int,
    failures: dict[int, str] | None = None,
) -> float:
    """Measure one metric and classify malformed local-judge responses as zero."""
    try:
        return float(metric.measure(test_case, _show_indicator=False))
    except Exception as exc:
        _LOG.warning(
            "[judge] %s failed for record %d (%s: %s); assigning 0.0",
            metric_name,
            record_index,
            type(exc).__name__,
            exc,
        )
        if failures is not None:
            from llb.scoring.judge_diag import classify_judge_exception

            failures[record_index] = classify_judge_exception(exc)
        return 0.0
