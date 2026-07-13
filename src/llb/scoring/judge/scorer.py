"""Canonical judge-score normalization and empty-answer handling."""

from collections.abc import Callable
from dataclasses import dataclass

from llb.core.contracts import JudgeInputRecord, JudgeScore

_EMPTY_ANSWER_SCORE: JudgeScore = {"faithfulness": 0.0, "answer_relevancy": 0.0}
_EMPTY_ANSWER_REASON = "empty_answer"
JudgeEvaluate = Callable[[list[JudgeInputRecord], str], list[dict[str, float]]]


@dataclass(frozen=True)
class _NonEmptyJudgeRecords:
    records: list[JudgeInputRecord]
    positions: list[int]


def extract_scores(rows: list[dict[str, float]]) -> list[JudgeScore]:
    """Normalize both judge signals into the canonical score contract."""
    return [
        {
            "faithfulness": float(row.get("faithfulness", 0.0) or 0.0),
            "answer_relevancy": float(row.get("answer_relevancy", 0.0) or 0.0),
        }
        for row in rows
    ]


def _split_nonempty_records(records: list[JudgeInputRecord]) -> _NonEmptyJudgeRecords:
    positions = [
        index for index, record in enumerate(records) if str(record.get("answer", "")).strip()
    ]
    return _NonEmptyJudgeRecords([records[index] for index in positions], positions)


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


def _score_with_empty_answers(
    records: list[JudgeInputRecord],
    judge_model: str,
    evaluate_fn: JudgeEvaluate | None,
    base_url: str | None,
) -> tuple[list[JudgeScore], list[str | None]]:
    nonempty = _split_nonempty_records(records)
    judged, judged_reasons = _score_nonempty_records(
        nonempty.records, judge_model, evaluate_fn, base_url
    )
    scores: list[JudgeScore] = [
        {
            "faithfulness": _EMPTY_ANSWER_SCORE["faithfulness"],
            "answer_relevancy": _EMPTY_ANSWER_SCORE["answer_relevancy"],
        }
        for _ in records
    ]
    reasons: list[str | None] = [_EMPTY_ANSWER_REASON for _ in records]
    for index, score, reason in zip(nonempty.positions, judged, judged_reasons):
        scores[index] = score
        reasons[index] = reason
    return scores, reasons


def deepeval_scorer(
    records: list[JudgeInputRecord],
    judge_model: str,
    *,
    evaluate_fn: JudgeEvaluate | None = None,
    base_url: str | None = None,
    diagnostics_out: list[str | None] | None = None,
) -> list[JudgeScore]:
    """Score faithfulness and relevancy while classifying judge-side failures."""
    if any(not str(record.get("answer", "")).strip() for record in records):
        scores, reasons = _score_with_empty_answers(records, judge_model, evaluate_fn, base_url)
        if diagnostics_out is not None:
            diagnostics_out.extend(reasons)
        return scores
    if evaluate_fn is not None:
        result = extract_scores(evaluate_fn(records, judge_model))
        if diagnostics_out is not None:
            diagnostics_out.extend(None for _ in records)
        return result
    from llb.scoring.judge.deepeval_adapter import default_deepeval_evaluate

    return default_deepeval_evaluate(
        records, judge_model, base_url=base_url, diagnostics_out=diagnostics_out
    )
