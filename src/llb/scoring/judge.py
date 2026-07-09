"""Gated local LLM judge using maintained DeepEval metrics and Ukrainian prompts.

The judge is a gated dependency. It contributes to ranking only after calibration against
human ratings clears the Spearman-rho floor. DeepEval is imported lazily and talks to any
OpenAI-compatible local endpoint through its maintained LocalModel adapter.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from llb.core.contracts import JudgeDiagnostics, JudgeInputRecord, JudgeScore
from llb.core import env
from llb.core.paths import load_project_env, resolve_data_dir
from llb.prompts import render_text, render_text_list, render_text_map

_LOG = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.6
_EMPTY_ANSWER_JUDGE_SCORE: JudgeScore = {"faithfulness": 0.0, "answer_relevancy": 0.0}
_EMPTY_ANSWER_REASON = "empty_answer"  # mirrors judge_diag.JUDGE_DIAG_EMPTY_ANSWER

# Judge-model bias disclosure (OQ2). The v1 default judge is a LOCAL Gemma-4 model, chosen for
# no data egress + reproducibility -- but it is NOT independent of the candidate pool: Gemma-4
# (E4B/12B) are candidates, and MamayLM v2 + Lapa are Gemma-3 fine-tunes, so the judge shares
# architecture / tokenizer / pretraining lineage with most of the pool and may self-prefer
# Gemma-family answers over the non-Gemma ones (Qwen, Llama). It is accepted only because the
# judge is gated and objective correctness keeps weight in the blend.
JUDGE_BIAS_NOTE = render_text("scoring.judge.bias_note")

UA_FAITHFULNESS_STEPS = render_text_list("scoring.judge.faithfulness_steps")
UA_ANSWER_RELEVANCY_STEPS = render_text_list("scoring.judge.relevancy_steps")

JudgeEvaluate = Callable[[list[JudgeInputRecord], str], list[dict[str, float]]]


def _isolate_deepeval_artifacts() -> None:
    """Point DeepEval's keystore (`.deepeval`) + results folder under $DATA_DIR/cache, not the root.

    Uses `setdefault` so an explicit env (e.g. exported by the Makefile) wins; the default follows
    the resolved DATA_DIR. Must run BEFORE `import deepeval` -- DeepEval reads DEEPEVAL_CACHE_FOLDER
    at import time (`constants.HIDDEN_DIR`).
    """
    cache_root = resolve_data_dir() / "cache" / "deepeval"
    os.environ.setdefault(env.DEEPEVAL_CACHE_FOLDER, str(cache_root))
    os.environ.setdefault(env.DEEPEVAL_RESULTS_FOLDER, str(cache_root / "results"))


class UkrainianGEvalTemplate:
    """DeepEval G-Eval result prompt with Ukrainian-only judge instructions."""

    _PARAMETER_LABELS = render_text_map("scoring.judge.parameter_labels")

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
        return render_text(
            "scoring.judge.evaluation_results",
            {
                "score_min": score_range[0],
                "score_max": score_range[1],
                "evaluation_steps": evaluation_steps,
                "test_case_content": test_case_content,
                "parameters": parameters,
            },
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
    diagnostics: JudgeDiagnostics | None = None  # judge diagnostics zero-valued-judge observability


@dataclass(frozen=True)
class _NonEmptyJudgeRecords:
    records: list[JudgeInputRecord]
    positions: list[int]


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


def _has_empty_answers(records: list[JudgeInputRecord]) -> bool:
    return any(not str(record.get("answer", "")).strip() for record in records)


def _split_nonempty_records(records: list[JudgeInputRecord]) -> _NonEmptyJudgeRecords:
    nonempty_records: list[JudgeInputRecord] = []
    nonempty_positions: list[int] = []
    for index, record in enumerate(records):
        if str(record.get("answer", "")).strip():
            nonempty_positions.append(index)
            nonempty_records.append(record)
    return _NonEmptyJudgeRecords(records=nonempty_records, positions=nonempty_positions)


def _empty_answer_scores(count: int) -> list[JudgeScore]:
    return [
        {
            "faithfulness": _EMPTY_ANSWER_JUDGE_SCORE["faithfulness"],
            "answer_relevancy": _EMPTY_ANSWER_JUDGE_SCORE["answer_relevancy"],
        }
        for _ in range(count)
    ]


def _score_nonempty_records(
    records: list[JudgeInputRecord],
    judge_model: str,
    evaluate_fn: JudgeEvaluate | None,
    base_url: str | None,
) -> tuple[list[JudgeScore], list[str | None]]:
    reasons: list[str | None] = []
    scores = (
        deepeval_scorer(
            records,
            judge_model,
            evaluate_fn=evaluate_fn,
            base_url=base_url,
            diagnostics_out=reasons,
        )
        if records
        else []
    )
    return scores, reasons


def _merge_empty_answer_scores(
    records: list[JudgeInputRecord],
    nonempty: _NonEmptyJudgeRecords,
    judged: list[JudgeScore],
    reasons: list[str | None],
) -> tuple[list[JudgeScore], list[str | None]]:
    scores = _empty_answer_scores(len(records))
    merged_reasons: list[str | None] = [_EMPTY_ANSWER_REASON for _ in records]
    for index, score, reason in zip(nonempty.positions, judged, reasons):
        scores[index] = score
        merged_reasons[index] = reason
    return scores, merged_reasons


def _score_with_empty_answer_handling(
    records: list[JudgeInputRecord],
    judge_model: str,
    evaluate_fn: JudgeEvaluate | None,
    base_url: str | None,
) -> tuple[list[JudgeScore], list[str | None]]:
    nonempty = _split_nonempty_records(records)
    judged, reasons = _score_nonempty_records(nonempty.records, judge_model, evaluate_fn, base_url)
    return _merge_empty_answer_scores(records, nonempty, judged, reasons)


def _score_with_injected_evaluate(
    records: list[JudgeInputRecord],
    judge_model: str,
    evaluate_fn: JudgeEvaluate,
    diagnostics_out: list[str | None] | None,
) -> list[JudgeScore]:
    result = extract_scores(evaluate_fn(records, judge_model))
    if diagnostics_out is not None:
        diagnostics_out.extend(None for _ in records)
    return result


def deepeval_scorer(
    records: list[JudgeInputRecord],
    judge_model: str,
    *,
    evaluate_fn: JudgeEvaluate | None = None,
    base_url: str | None = None,
    diagnostics_out: list[str | None] | None = None,
) -> list[JudgeScore]:
    """Score faithfulness and answer relevancy with Ukrainian DeepEval G-Eval prompts.

    `diagnostics_out`, when provided, is filled with one precise reason per record (or None) for
    the judge diagnostics zero-valued-judge observability: `empty_answer` for blank candidate answers and the
    classified failure reason for a judge that could not score a non-empty answer."""
    if records and _has_empty_answers(records):
        scores, reasons = _score_with_empty_answer_handling(
            records, judge_model, evaluate_fn, base_url
        )
        if diagnostics_out is not None:
            diagnostics_out.extend(reasons)
        return scores
    if evaluate_fn is not None:
        return _score_with_injected_evaluate(records, judge_model, evaluate_fn, diagnostics_out)
    return _default_deepeval_evaluate(
        records, judge_model, base_url=base_url, diagnostics_out=diagnostics_out
    )


_UKRAINIAN_GEVAL_CACHE: dict[type[Any], type[Any]] = {}


def _ukrainian_geval_class(geval_cls: type[Any]) -> type[Any]:
    """Subclass GEval so its result prompts come from `UkrainianGEvalTemplate`.

    DeepEval 4.x dropped the `evaluation_template` constructor argument and now resolves every
    prompt through `_get_prompt(method, ...)`. We override that single hook and route the two
    result methods to our Ukrainian template; the evaluation-steps prompt is never reached
    because the metrics are constructed with `evaluation_steps` already supplied. The class is
    cached per GEval type so the lazy optional dependency stays import-light."""
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


def _default_deepeval_evaluate(
    records: list[JudgeInputRecord],
    judge_model: str,
    *,
    base_url: str | None = None,
    diagnostics_out: list[str | None] | None = None,
) -> list[JudgeScore]:
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

    api_key = os.environ.get(env.DEEPEVAL_JUDGE_API_KEY) or "local"
    model = LocalModel(
        model=served_model,
        base_url=resolved_base_url,
        api_key=api_key,
        temperature=0.0,
        format="json",
    )
    # DeepEval 4.x renders prompts through `_get_prompt`, so a thin subclass borrows our
    # Ukrainian template while keeping the metric engine otherwise untouched.
    ua_geval = _ukrainian_geval_class(GEval)
    faithfulness = ua_geval(
        name="UA Faithfulness",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        evaluation_steps=UA_FAITHFULNESS_STEPS,
        model=model,
        async_mode=False,
        _include_g_eval_suffix=False,
    )
    relevancy = ua_geval(
        name="UA Answer Relevancy",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        evaluation_steps=UA_ANSWER_RELEVANCY_STEPS,
        model=model,
        async_mode=False,
        _include_g_eval_suffix=False,
    )

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
                "faithfulness": _measure_judge_metric(
                    faithfulness,
                    test_case,
                    metric_name="faithfulness",
                    record_index=index,
                    failures=failures,
                ),
                "answer_relevancy": _measure_judge_metric(
                    relevancy,
                    test_case,
                    metric_name="answer_relevancy",
                    record_index=index,
                    failures=failures,
                ),
            }
        )
    if diagnostics_out is not None:
        diagnostics_out.extend(failures.get(index) for index in range(len(records)))
    return scores


def _measure_judge_metric(
    metric: Any,
    test_case: Any,
    *,
    metric_name: str,
    record_index: int,
    failures: dict[int, str] | None = None,
) -> float:
    """Measure one DeepEval metric, converting malformed local-judge responses to zero.

    Local judges can fail to return the strict JSON DeepEval expects. That is a judge-quality
    failure, not a reason to abort the benchmark; the objective score remains the headline, and
    the diagnostic metric gets zero for the affected record. When `failures` is provided the
    failure is classified (malformed JSON vs transport) and recorded for the judge diagnostics diagnostics.
    """
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
    """Resolve the served judge model id and OpenAI-compatible endpoint."""
    load_project_env()
    base_url = explicit_base_url or os.environ.get(env.DEEPEVAL_JUDGE_BASE_URL)
    if base_url is not None:
        base_url = _normalize_openai_base_url(base_url)
    return judge_model, base_url


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
