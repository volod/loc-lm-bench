"""Per-document objective scoring and gated free-form judge diagnostics."""

import json
import logging
from typing import Any

from llb.bench.common import LLMComplete, mean, run_gated_judge
from llb.bench.text_analysis.model import JudgeConfig, JudgeQualityResult, ScoredTextAnalysisDocs
from llb.bench.text_analysis.prompts import (
    JUDGED_EXTRACT_KINDS,
    JUDGE_INTENT,
    analysis_prompt,
    long_doc_question,
    parse_predictions,
)
from llb.core.contracts import JudgeInputRecord, JudgeScore, TextAnalysisCaseRow
from llb.eval.common import EMPTY, MALFORMED, OK
from llb.eval.map_reduce import run_map_reduce_text
from llb.scoring import text_analysis as ta
from llb.scoring.aggregate import bootstrap_mean_ci

_LOG = logging.getLogger(__name__)


def score_doc_batch(
    doc_ids: list[str],
    labels_by_doc: dict[str, list[ta.PlantedLabel]],
    docs: dict[str, str],
    complete: LLMComplete,
    similarity: ta.Similarity,
) -> ScoredTextAnalysisDocs:
    rows: list[TextAnalysisCaseRow] = []
    case_objectives: list[float] = []
    judge_records: list[JudgeInputRecord] = []
    judge_row_index: list[int] = []
    n_ok = 0
    for doc_id in doc_ids:
        labels = labels_by_doc[doc_id]
        doc_text = docs[doc_id]
        status, predictions = _predict_doc_extractions(doc_id, doc_text, labels, complete)
        scored = ta.score_document(predictions, labels, similarity)
        if status == OK:
            n_ok += 1
        case_objectives.append(float(scored["objective_score"]))
        row = _case_row(doc_id, status, scored, len(labels))
        row_index = len(rows)
        long_doc_record = _long_doc_judge_record(labels, doc_text, complete)
        if long_doc_record is not None:
            answer, record = long_doc_record
            row["long_doc_answer"] = answer[:280]
            judge_records.append(record)
            judge_row_index.append(row_index)
        _append_freeform_judge_records(
            predictions, doc_text, judge_records, judge_row_index, row_index
        )
        rows.append(row)
    return ScoredTextAnalysisDocs(
        doc_ids, rows, case_objectives, judge_records, judge_row_index, n_ok
    )


def run_judged_quality(scored: ScoredTextAnalysisDocs, config: JudgeConfig) -> JudgeQualityResult:
    outcome = run_gated_judge(
        scored.judge_records,
        judge_model=config.model,
        judge_rho=config.rho,
        threshold=config.threshold,
        scorer=config.scorer,
        base_url=config.base_url,
    )
    if outcome.trusted and outcome.scores:
        quality, quality_ci = _attach_judged_quality(
            scored.rows, outcome.scores, scored.judge_row_index
        )
        return JudgeQualityResult(outcome, quality, quality_ci)
    if config.model is not None:
        _LOG.info("[text-analysis] judge demoted (%s)", outcome.reason)
    return JudgeQualityResult(outcome, None, None)


def judged_quality(score: JudgeScore) -> float:
    return (float(score["faithfulness"]) + float(score["answer_relevancy"])) / 2.0


def _predict_doc_extractions(
    doc_id: str,
    doc_text: str,
    labels: list[ta.PlantedLabel],
    complete: LLMComplete,
) -> tuple[str, dict[str, list[str]]]:
    kinds = sorted({label.kind for label in labels if label.kind != ta.LONG_DOC})
    raw = complete(analysis_prompt(doc_id, doc_text, kinds)) if kinds else ""
    if kinds and not raw.strip():
        return EMPTY, {}
    if not kinds:
        return OK, {}
    try:
        return OK, parse_predictions(raw, kinds)
    except (ValueError, json.JSONDecodeError):
        return MALFORMED, {}


def _case_row(
    doc_id: str, status: str, scored: dict[str, Any], n_labels: int
) -> TextAnalysisCaseRow:
    f1_by_kind = {kind: subtask["f1"] for kind, subtask in scored["subtasks"].items()}
    return {
        "item_id": doc_id,
        "status": status,
        "objective_score": float(scored["objective_score"]),
        "n_objective_subtasks": int(scored["n_objective_subtasks"]),
        "n_labels": n_labels,
        "subtask_f1_json": json.dumps(f1_by_kind, ensure_ascii=False, sort_keys=True),
    }


def _long_doc_judge_record(
    labels: list[ta.PlantedLabel], doc_text: str, complete: LLMComplete
) -> tuple[str, JudgeInputRecord] | None:
    question = long_doc_question(labels)
    if question is None:
        return None
    answer = run_map_reduce_text(complete, question, doc_text)
    return answer, {"question": question, "answer": answer, "contexts": [doc_text]}


def _append_freeform_judge_records(
    predictions: dict[str, list[str]],
    doc_text: str,
    judge_records: list[JudgeInputRecord],
    judge_row_index: list[int],
    row_index: int,
) -> None:
    for kind in JUDGED_EXTRACT_KINDS:
        answer = " ".join(predictions.get(kind, [])).strip()
        if answer:
            judge_records.append(
                {"question": JUDGE_INTENT[kind], "answer": answer, "contexts": [doc_text]}
            )
            judge_row_index.append(row_index)


def _attach_judged_quality(
    rows: list[TextAnalysisCaseRow], scores: list[JudgeScore], judge_row_index: list[int]
) -> tuple[float, tuple[float, float] | None]:
    per_record = [judged_quality(score) for score in scores]
    per_row: dict[int, list[float]] = {}
    for row_index, value in zip(judge_row_index, per_record):
        per_row.setdefault(row_index, []).append(value)
    for row_index, values in per_row.items():
        rows[row_index]["judged_quality"] = round(mean(values), 6)
    return round(mean(per_record), 6), bootstrap_mean_ci(per_record)
