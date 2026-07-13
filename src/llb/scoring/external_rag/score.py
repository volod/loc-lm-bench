"""Objective scoring, answer classification, and provider-source normalization."""

import re
from typing import Any

from llb.eval import common as eval_common
from llb.scoring.correctness import answer_correctness
from llb.scoring.external_rag_common import (
    ABSTENTION_MARKERS,
    ANSWER_FIELD_CANDIDATES,
    DEFAULT_SOURCE_LIMIT,
    ERROR_FIELD_CANDIDATES,
    SOURCES_FIELD_CANDIDATES,
    STATUS_ABSTAINED,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_REFUSAL,
    _SOURCE_FOOTER_RE,
    _as_float,
    _as_int,
    _string,
)
from llb.scoring.external_rag_sources import SourceMap, audit_row_sources


def score_records(
    records: list[dict[str, Any]],
    *,
    answer_field: str | None = None,
    sources_field: str | None = None,
    error_field: str | None = None,
    source_limit: int = DEFAULT_SOURCE_LIMIT,
    strip_source_footer: bool = True,
    source_map: SourceMap | None = None,
) -> list[dict[str, object]]:
    """Score records and return worksheet rows sorted by review priority."""
    from llb.scoring.external_rag.worksheet import base_row

    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, 1):
        raw_answer, answer_field_used = field_value(record, answer_field, ANSWER_FIELD_CANDIDATES)
        raw_error, error_field_used = field_value(record, error_field, ERROR_FIELD_CANDIDATES)
        raw_sources, sources_field_used = field_value(
            record, sources_field, SOURCES_FIELD_CANDIDATES
        )
        answer = _string(raw_answer)
        scored_answer = clean_answer_for_scoring(answer, strip_source_footer=strip_source_footer)
        error = _string(raw_error)
        status = classify_external_answer(scored_answer, error)
        scores = answer_correctness(scored_answer, _string(record.get("reference_answer")))
        row = base_row(
            record,
            input_index=index,
            status=status,
            scores=scores,
            answer=answer,
            scored_answer=scored_answer,
            error=error,
            answer_field=answer_field_used,
            error_field=error_field_used,
            sources_field=sources_field_used,
            source_limit=source_limit,
        )
        row.update(source_columns(raw_sources, source_limit))
        if source_map is not None:
            gold_spans = record.get("source_spans")
            row.update(
                audit_row_sources(
                    source_list(raw_sources),
                    gold_spans if isinstance(gold_spans, list) else [],
                    source_map,
                )
            )
        rows.append(row)
    _attach_ranks(rows)
    rows.sort(
        key=lambda row: (
            _as_int(row.get("review_priority_rank")),
            -_as_float(row.get("objective_score")),
            _as_int(row.get("input_index")),
        )
    )
    return rows


def clean_answer_for_scoring(answer: str, *, strip_source_footer: bool = True) -> str:
    """Remove transport-only decorations before objective answer scoring."""
    text = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    if strip_source_footer:
        text = _SOURCE_FOOTER_RE.sub("", text).strip()
    return text


def classify_external_answer(answer: str, error: str) -> str:
    """Classify the answer for reliability and human-review grouping."""
    if error.strip():
        return STATUS_ERROR
    if not answer.strip():
        return STATUS_EMPTY
    if eval_common.is_refusal(answer):
        return STATUS_REFUSAL
    normalized = eval_common.normalize_refusal_text(answer)
    if any(marker in normalized for marker in ABSTENTION_MARKERS):
        return STATUS_ABSTAINED
    return STATUS_OK


def source_columns(raw_sources: object, source_limit: int) -> dict[str, object]:
    sources = source_list(raw_sources)
    out: dict[str, object] = {}
    for index, source in enumerate(sources[:source_limit], 1):
        out[f"source_{index}_article_id"] = _string(source.get("article_id") or source.get("id"))
        out[f"source_{index}_doc_id"] = _string(source.get("doc_id") or source.get("document_id"))
        out[f"source_{index}_title"] = _string(
            source.get("article_title") or source.get("title") or source.get("name")
        )
        out[f"source_{index}_score"] = _string(source.get("score"))
        out[f"source_{index}_url"] = _string(source.get("url") or source.get("uri"))
    return out


def source_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def field_value(
    record: dict[str, Any], requested: str | None, candidates: tuple[str, ...]
) -> tuple[object, str]:
    if requested is not None:
        return record.get(requested), requested
    for field in candidates:
        if field in record:
            return record.get(field), field
    return "", ""


def _attach_ranks(rows: list[dict[str, object]]) -> None:
    score_order = sorted(
        range(len(rows)),
        key=lambda index: (
            -_as_float(rows[index].get("objective_score")),
            _as_int(rows[index].get("input_index")),
        ),
    )
    for rank, index in enumerate(score_order, 1):
        rows[index]["score_rank"] = rank
    priorities = {
        STATUS_ERROR: 0,
        STATUS_EMPTY: 1,
        STATUS_ABSTAINED: 2,
        STATUS_REFUSAL: 3,
        STATUS_OK: 4,
    }
    review_order = sorted(
        range(len(rows)),
        key=lambda index: (
            priorities.get(_string(rows[index].get("status")), 5),
            _as_float(rows[index].get("objective_score")),
            _as_int(rows[index].get("source_count")),
            _as_int(rows[index].get("input_index")),
        ),
    )
    for rank, index in enumerate(review_order, 1):
        rows[index]["review_priority_rank"] = rank
