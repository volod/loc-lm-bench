"""Stable CSV worksheet schema and row rendering."""

import csv
import io
from pathlib import Path
from typing import Any

from llb.core.contracts.rag import CorrectnessScores
from llb.core.fsutil import atomic_write_text
from llb.scoring.external_rag.score import field_value, source_list
from llb.scoring.external_rag_common import (
    HUMAN_CORRECTED_ANSWER_FIELD,
    HUMAN_DECISION_FIELD,
    HUMAN_NOTES_FIELD,
    HUMAN_SCORE_FIELD,
    HUMAN_STATUS_FIELD,
    MODEL_FIELD_CANDIDATES,
    PROVIDER_FIELD_CANDIDATES,
    ROUTE_FIELD_CANDIDATES,
    _round,
    _string,
)
from llb.scoring.external_rag_source_map import SOURCE_AUDIT_COLUMNS


def write_csv(
    rows: list[dict[str, object]], path: Path, *, source_limit: int, source_audit: bool = False
) -> None:
    """Write the detailed per-row review worksheet."""
    fieldnames = csv_columns(source_limit, source_audit=source_audit)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(_csv_row(row, fieldnames) for row in rows)
    atomic_write_text(path, out.getvalue())


def csv_columns(source_limit: int, *, source_audit: bool = False) -> list[str]:
    """Return the stable column order for review and downstream analysis."""
    columns = [
        "review_priority_rank",
        "score_rank",
        "input_index",
        "id",
        "split",
        "verified",
        "status",
        "objective_score",
        "token_f1",
        "exact",
        "contains",
        "question",
        "reference_answer",
        "scored_answer",
        "llm_answer",
        "llm_model",
        "llm_provider",
        "llm_route",
        "llm_error",
        "answer_field",
        "error_field",
        "sources_field",
        "source_doc_id",
        "source_span_1_doc_id",
        "source_span_1_char_start",
        "source_span_1_char_end",
        "source_span_1_text",
        "source_count",
    ]
    for index in range(1, source_limit + 1):
        columns.extend(
            [
                f"source_{index}_article_id",
                f"source_{index}_doc_id",
                f"source_{index}_title",
                f"source_{index}_score",
                f"source_{index}_url",
            ]
        )
    if source_audit:
        columns.extend(SOURCE_AUDIT_COLUMNS)
    columns.extend(
        [
            HUMAN_SCORE_FIELD,
            HUMAN_DECISION_FIELD,
            HUMAN_NOTES_FIELD,
            HUMAN_CORRECTED_ANSWER_FIELD,
            HUMAN_STATUS_FIELD,
        ]
    )
    return columns


def base_row(
    record: dict[str, Any],
    *,
    input_index: int,
    status: str,
    scores: CorrectnessScores,
    answer: str,
    scored_answer: str,
    error: str,
    answer_field: str,
    error_field: str,
    sources_field: str,
    source_limit: int,
) -> dict[str, object]:
    span = _first_span(record)
    return {
        "review_priority_rank": 0,
        "score_rank": 0,
        "input_index": input_index,
        "id": _string(record.get("id")),
        "split": _string(record.get("split")),
        "verified": str(bool(record.get("verified", False))).lower(),
        "status": status,
        "objective_score": _round(scores["score"]),
        "token_f1": _round(scores["token_f1"]),
        "exact": _round(scores["exact"]),
        "contains": _round(scores["contains"]),
        "question": _string(record.get("question")),
        "reference_answer": _string(record.get("reference_answer")),
        "scored_answer": scored_answer,
        "llm_answer": answer,
        "llm_model": _field_string(record, MODEL_FIELD_CANDIDATES),
        "llm_provider": _field_string(record, PROVIDER_FIELD_CANDIDATES),
        "llm_route": _field_string(record, ROUTE_FIELD_CANDIDATES),
        "llm_error": error,
        "answer_field": answer_field,
        "error_field": error_field,
        "sources_field": sources_field,
        "source_doc_id": _string(record.get("source_doc_id")),
        "source_span_1_doc_id": _string(span.get("doc_id")),
        "source_span_1_char_start": _string(span.get("char_start")),
        "source_span_1_char_end": _string(span.get("char_end")),
        "source_span_1_text": _string(span.get("text")),
        "source_count": len(source_list(record.get(sources_field) if sources_field else None)),
        HUMAN_SCORE_FIELD: _string(record.get(HUMAN_SCORE_FIELD)),
        HUMAN_DECISION_FIELD: _string(record.get(HUMAN_DECISION_FIELD)),
        HUMAN_NOTES_FIELD: _string(record.get(HUMAN_NOTES_FIELD)),
        HUMAN_CORRECTED_ANSWER_FIELD: _string(record.get(HUMAN_CORRECTED_ANSWER_FIELD)),
        HUMAN_STATUS_FIELD: _string(record.get(HUMAN_STATUS_FIELD)),
        **{key: "" for key in _empty_source_column_names(source_limit)},
    }


def _field_string(record: dict[str, Any], candidates: tuple[str, ...]) -> str:
    value, _field = field_value(record, None, candidates)
    return _string(value)


def _first_span(record: dict[str, Any]) -> dict[str, Any]:
    spans = record.get("source_spans")
    if isinstance(spans, list) and spans and isinstance(spans[0], dict):
        return spans[0]
    return {}


def _empty_source_column_names(source_limit: int) -> list[str]:
    return [
        f"source_{index}_{field}"
        for index in range(1, source_limit + 1)
        for field in ("article_id", "doc_id", "title", "score", "url")
    ]


def _csv_row(row: dict[str, object], fieldnames: list[str]) -> dict[str, str]:
    return {field: " ".join(_string(row.get(field)).splitlines()) for field in fieldnames}
