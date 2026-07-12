"""Score answered JSONL exports from an external or closed RAG system.

The normal `run-eval` path owns retrieval and generation locally. This module covers the other
operator workflow: a RAG system outside the benchmark has already answered each gold question, and
the benchmark should produce objective estimates plus final human-reviewed CSV/report artifacts.

This module keeps the scoring core (record scoring, status classification, the CSV worksheet, and
the `score_external_rag_file` orchestration). The shared schema/helpers live in
`external_rag_common.py`, headline aggregation in `external_rag_summary.py`, and the Markdown report
in `external_rag_report.py`; the public names from all three are re-exported so
`llb.scoring.external_rag.<name>` keeps working.
"""

import csv
import io
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llb.core.contracts import CorrectnessScores
from llb.core.fsutil import atomic_write_text
from llb.eval import common as eval_common
from llb.scoring.correctness import answer_correctness
from llb.scoring.external_rag_common import (
    ABSTENTION_MARKERS,
    ANSWER_FIELD_CANDIDATES,
    DEFAULT_SOURCE_LIMIT,
    ERROR_FIELD_CANDIDATES,
    HUMAN_CORRECTED_ANSWER_FIELD,
    HUMAN_DECISION_ACCEPT,
    HUMAN_DECISION_FIELD,
    HUMAN_DECISION_PARTIAL,
    HUMAN_DECISION_REJECT,
    HUMAN_DECISIONS,
    HUMAN_FIELDS,
    HUMAN_NOTES_FIELD,
    HUMAN_SCORE_FIELD,
    HUMAN_STATUS_FIELD,
    HUMAN_STATUS_SCORED,
    MODEL_FIELD_CANDIDATES,
    PROVIDER_FIELD_CANDIDATES,
    ROUTE_FIELD_CANDIDATES,
    SOURCES_FIELD_CANDIDATES,
    STATUS_ABSTAINED,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_REFUSAL,
    _SOURCE_FOOTER_RE,
    _as_float,
    _as_int,
    _round,
    _string,
    ExternalRagPaths,
    ExternalRagResult,
)
from llb.scoring.external_rag_report import write_report
from llb.scoring.external_rag_sources import (
    SOURCE_AUDIT_COLUMNS,
    SourceMap,
    audit_row_sources,
    load_source_map,
    summarize_source_audit,
)
from llb.scoring.external_rag_summary import summarize

__all__ = [
    # shared schema (external_rag_common)
    "ABSTENTION_MARKERS",
    "ANSWER_FIELD_CANDIDATES",
    "DEFAULT_SOURCE_LIMIT",
    "ERROR_FIELD_CANDIDATES",
    "HUMAN_CORRECTED_ANSWER_FIELD",
    "HUMAN_DECISION_ACCEPT",
    "HUMAN_DECISION_FIELD",
    "HUMAN_DECISION_PARTIAL",
    "HUMAN_DECISION_REJECT",
    "HUMAN_DECISIONS",
    "HUMAN_FIELDS",
    "HUMAN_NOTES_FIELD",
    "HUMAN_SCORE_FIELD",
    "HUMAN_STATUS_FIELD",
    "HUMAN_STATUS_SCORED",
    "MODEL_FIELD_CANDIDATES",
    "PROVIDER_FIELD_CANDIDATES",
    "ROUTE_FIELD_CANDIDATES",
    "SOURCES_FIELD_CANDIDATES",
    "STATUS_ABSTAINED",
    "STATUS_EMPTY",
    "STATUS_ERROR",
    "STATUS_OK",
    "STATUS_REFUSAL",
    "ExternalRagPaths",
    "ExternalRagResult",
    # aggregation + report (re-exported)
    "summarize",
    "write_report",
    # scoring core
    "classify_external_answer",
    "clean_answer_for_scoring",
    "clear_human_fields",
    "csv_columns",
    "ensure_human_fields",
    "human_reviewed_count",
    "is_human_scored",
    "load_jsonl",
    "score_external_rag_file",
    "score_records",
    "write_csv",
    "write_jsonl",
]


def score_external_rag_file(
    answers_path: Path,
    *,
    csv_out: Path | None = None,
    report_out: Path | None = None,
    answer_field: str | None = None,
    sources_field: str | None = None,
    error_field: str | None = None,
    source_limit: int = DEFAULT_SOURCE_LIMIT,
    strip_source_footer: bool = True,
    label: str | None = None,
    source_map_path: Path | None = None,
) -> ExternalRagResult:
    """Read an answered JSONL file and write the detailed CSV plus Markdown report.

    `source_map_path` (external-rag-source-mapping) joins the provider's returned source
    records onto corpus spans, adding the source-hit / first-hit-rank / missing-mapping
    columns and the source-span audit summary.
    """
    if source_limit < 0:
        raise ValueError("source_limit must be >= 0")
    records = load_jsonl(answers_path)
    if not records:
        raise ValueError(f"{answers_path}: no JSONL records found")

    source_map = load_source_map(source_map_path) if source_map_path is not None else None
    scored = score_records(
        records,
        answer_field=answer_field,
        sources_field=sources_field,
        error_field=error_field,
        source_limit=source_limit,
        strip_source_footer=strip_source_footer,
        source_map=source_map,
    )
    csv_path = csv_out or answers_path.with_suffix(".csv")
    report_path = report_out or answers_path.with_name(f"{answers_path.stem}.report.md")
    summary = summarize(scored, answers_path=answers_path, label=label)
    if source_map is not None:
        summary["source_audit"] = summarize_source_audit(scored)
    write_csv(scored, csv_path, source_limit=source_limit, source_audit=source_map is not None)
    write_report(
        scored,
        summary,
        report_path,
        answers_path=answers_path,
        csv_path=csv_path,
        source_limit=source_limit,
    )
    return ExternalRagResult(
        rows=scored, summary=summary, paths=ExternalRagPaths(csv_path, report_path)
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows with file:line context on parse failures."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(item)
    return rows


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Atomically write JSONL records, preserving Unicode text for human review."""
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    atomic_write_text(path, text)


def ensure_human_fields(records: Sequence[dict[str, Any]]) -> bool:
    """Ensure every record has the JSONL-backed human review fields.

    Returns true when at least one record changed.
    """
    changed = False
    for record in records:
        for field in HUMAN_FIELDS:
            if field not in record:
                record[field] = ""
                changed = True
    return changed


def clear_human_fields(records: Sequence[dict[str, Any]]) -> None:
    """Clear JSONL-backed human review state in place."""
    for record in records:
        for field in HUMAN_FIELDS:
            record[field] = ""


def is_human_scored(record: dict[str, Any]) -> bool:
    """Whether a record carries the required human scoring fields."""
    decision = _string(record.get(HUMAN_DECISION_FIELD)).strip().lower()
    score_text = _string(record.get(HUMAN_SCORE_FIELD)).strip()
    if decision not in HUMAN_DECISIONS or not score_text:
        return False
    try:
        score = float(score_text)
    except ValueError:
        return False
    return 0.0 <= score <= 1.0


def human_reviewed_count(records: Sequence[dict[str, Any]]) -> int:
    """Number of records with complete human review state."""
    return sum(1 for record in records if is_human_scored(record))


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
    """Score records and return CSV-ready rows sorted by human review priority."""
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, 1):
        raw_answer, answer_field_used = _field_value(record, answer_field, ANSWER_FIELD_CANDIDATES)
        raw_error, error_field_used = _field_value(record, error_field, ERROR_FIELD_CANDIDATES)
        raw_sources, sources_field_used = _field_value(
            record, sources_field, SOURCES_FIELD_CANDIDATES
        )
        answer = _string(raw_answer)
        scored_answer = clean_answer_for_scoring(answer, strip_source_footer=strip_source_footer)
        error = _string(raw_error)
        status = classify_external_answer(scored_answer, error)
        scores = answer_correctness(scored_answer, _string(record.get("reference_answer")))
        row = _base_row(
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
        row.update(_source_columns(raw_sources, source_limit))
        if source_map is not None:
            gold_spans = record.get("source_spans")
            row.update(
                audit_row_sources(
                    _source_list(raw_sources),
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
    text = _strip_think(answer).strip()
    if strip_source_footer:
        text = _SOURCE_FOOTER_RE.sub("", text).strip()
    return text


def classify_external_answer(answer: str, error: str) -> str:
    """Classify the answer for reliability and human review grouping."""
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


def write_csv(
    rows: list[dict[str, object]], path: Path, *, source_limit: int, source_audit: bool = False
) -> None:
    """Write the detailed per-row worksheet CSV."""
    fieldnames = csv_columns(source_limit, source_audit=source_audit)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(_csv_row(row, fieldnames) for row in rows)
    atomic_write_text(path, out.getvalue())


def csv_columns(source_limit: int, *, source_audit: bool = False) -> list[str]:
    """Stable CSV column order for human review and downstream analysis."""
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


def _base_row(
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
        "llm_model": _field_string(record, None, MODEL_FIELD_CANDIDATES),
        "llm_provider": _field_string(record, None, PROVIDER_FIELD_CANDIDATES),
        "llm_route": _field_string(record, None, ROUTE_FIELD_CANDIDATES),
        "llm_error": error,
        "answer_field": answer_field,
        "error_field": error_field,
        "sources_field": sources_field,
        "source_doc_id": _string(record.get("source_doc_id")),
        "source_span_1_doc_id": _string(span.get("doc_id")),
        "source_span_1_char_start": _string(span.get("char_start")),
        "source_span_1_char_end": _string(span.get("char_end")),
        "source_span_1_text": _string(span.get("text")),
        "source_count": _source_count(record, sources_field),
        HUMAN_SCORE_FIELD: _string(record.get(HUMAN_SCORE_FIELD)),
        HUMAN_DECISION_FIELD: _string(record.get(HUMAN_DECISION_FIELD)),
        HUMAN_NOTES_FIELD: _string(record.get(HUMAN_NOTES_FIELD)),
        HUMAN_CORRECTED_ANSWER_FIELD: _string(record.get(HUMAN_CORRECTED_ANSWER_FIELD)),
        HUMAN_STATUS_FIELD: _string(record.get(HUMAN_STATUS_FIELD)),
        **{key: "" for key in _empty_source_column_names(source_limit)},
    }


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

    review_order = sorted(
        range(len(rows)),
        key=lambda index: (
            _status_priority(_string(rows[index].get("status"))),
            _as_float(rows[index].get("objective_score")),
            _as_int(rows[index].get("source_count")),
            _as_int(rows[index].get("input_index")),
        ),
    )
    for rank, index in enumerate(review_order, 1):
        rows[index]["review_priority_rank"] = rank


def _status_priority(status: str) -> int:
    priorities = {
        STATUS_ERROR: 0,
        STATUS_EMPTY: 1,
        STATUS_ABSTAINED: 2,
        STATUS_REFUSAL: 3,
        STATUS_OK: 4,
    }
    return priorities.get(status, 5)


def _source_columns(raw_sources: object, source_limit: int) -> dict[str, object]:
    sources = _source_list(raw_sources)
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


def _source_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _source_count(record: dict[str, Any], sources_field: str) -> int:
    sources = record.get(sources_field) if sources_field else None
    return len(_source_list(sources))


def _empty_source_column_names(source_limit: int) -> list[str]:
    names: list[str] = []
    for index in range(1, source_limit + 1):
        names.extend(
            [
                f"source_{index}_article_id",
                f"source_{index}_doc_id",
                f"source_{index}_title",
                f"source_{index}_score",
                f"source_{index}_url",
            ]
        )
    return names


def _first_span(record: dict[str, Any]) -> dict[str, Any]:
    spans = record.get("source_spans")
    if isinstance(spans, list) and spans and isinstance(spans[0], dict):
        return spans[0]
    return {}


def _field_value(
    record: dict[str, Any], requested: str | None, candidates: tuple[str, ...]
) -> tuple[object, str]:
    if requested is not None:
        return record.get(requested), requested
    for field in candidates:
        if field in record:
            return record.get(field), field
    return "", ""


def _field_string(
    record: dict[str, Any], requested: str | None, candidates: tuple[str, ...]
) -> str:
    value, _field = _field_value(record, requested, candidates)
    return _string(value)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _csv_row(row: dict[str, object], fieldnames: list[str]) -> dict[str, str]:
    return {field: _one_line(_string(row.get(field))) for field in fieldnames}


def _one_line(value: str) -> str:
    return " ".join(value.splitlines())
