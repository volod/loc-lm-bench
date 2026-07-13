"""Terminal card rendering for the external-RAG review session (pure presentation)."""

from collections.abc import Sequence
from typing import Any

from llb.scoring.external_rag.score import (
    clean_answer_for_scoring,
    field_value,
    score_records,
    source_list,
)
from llb.scoring.external_rag_common import (
    HUMAN_CORRECTED_ANSWER_FIELD,
    HUMAN_DECISION_ACCEPT,
    HUMAN_DECISION_FIELD,
    HUMAN_DECISION_PARTIAL,
    HUMAN_DECISION_REJECT,
    HUMAN_NOTES_FIELD,
    HUMAN_SCORE_FIELD,
    _string,
)

TEXT_PREVIEW_CHARS = 1200
SOURCE_PREVIEW_CHARS = 700
SPAN_LIMIT = 3


def format_card(
    record: dict[str, Any],
    position: int,
    total: int,
    reviewed: int,
    *,
    answer_field: str | None,
    sources_field: str | None,
    error_field: str | None,
    source_limit: int,
    strip_source_footer: bool,
) -> str:
    """Render the current row for human scoring."""
    scored = score_records(
        [record],
        answer_field=answer_field,
        sources_field=sources_field,
        error_field=error_field,
        source_limit=source_limit,
        strip_source_footer=strip_source_footer,
    )[0]
    raw_answer, _answer_field_used = field_value(
        record,
        answer_field,
        ("llm_answer", "predicted_answer", "model_answer", "answer"),
    )
    raw_error, _error_field_used = field_value(record, error_field, ("llm_error", "error"))
    raw_sources, _sources_field_used = field_value(
        record, sources_field, ("llm_sources", "sources", "retrieved_sources")
    )
    remaining = total - reviewed
    answer = _string(raw_answer)
    scored_answer = clean_answer_for_scoring(answer, strip_source_footer=strip_source_footer)
    lines = [
        "===== external RAG human review =====",
        f"item {position}/{total} (reviewed {reviewed}, remaining {remaining})",
        f"== id: {_string(record.get('id'))}",
        f"== meta: split={_string(record.get('split'))} "
        f"source_doc_id={_string(record.get('source_doc_id'))} "
        f"verified={_string(record.get('verified'))}",
        f"== auto_score: status={_string(scored.get('status'))} "
        f"objective={_float_text(scored.get('objective_score'))} "
        f"exact/token/contains={_float_text(scored.get('exact'))}/"
        f"{_float_text(scored.get('token_f1'))}/{_float_text(scored.get('contains'))}",
        "",
        f"== question: {_preview_one_line(_string(record.get('question')), TEXT_PREVIEW_CHARS)}",
        f"== reference_answer: "
        f"{_preview_one_line(_string(record.get('reference_answer')), TEXT_PREVIEW_CHARS)}",
        "== gold_source_text",
        *_source_span_lines(record),
        f"== llm_answer: {_preview_one_line(answer, TEXT_PREVIEW_CHARS)}",
        f"== scored_answer: {_preview_one_line(scored_answer, TEXT_PREVIEW_CHARS)}",
        "== llm_sources",
        *_returned_source_lines(raw_sources, source_limit),
        f"== llm_error: {_preview_one_line(_string(raw_error), TEXT_PREVIEW_CHARS) or '(none)'}",
        f"== human: {HUMAN_SCORE_FIELD}={_field(record, HUMAN_SCORE_FIELD, '(unscored)')} "
        f"{HUMAN_DECISION_FIELD}={_field(record, HUMAN_DECISION_FIELD, '(unset)')}",
        f"== human_notes: {_field(record, HUMAN_NOTES_FIELD, '')}",
        f"== human_corrected_answer: {_field(record, HUMAN_CORRECTED_ANSWER_FIELD, '')}",
    ]
    return "\n".join(lines)


def completion_panel(records: Sequence[dict[str, Any]]) -> str:
    """All-reviewed screen."""
    counts = _decision_counts(records)
    decision_text = ", ".join(f"{key}={value}" for key, value in counts.items()) or "none"
    return "\n".join(
        [
            f"===== all {len(records)} rows scored ({decision_text}) =====",
            "  review/change: b = last row, j <N> = jump to row N",
            "  finish: press Enter or q to save JSONL and write CSV/report",
        ]
    )


def _source_span_lines(record: dict[str, Any]) -> list[str]:
    spans = record.get("source_spans")
    if not isinstance(spans, list) or not spans:
        return ["  (none)"]
    lines: list[str] = []
    for index, span in enumerate(spans[:SPAN_LIMIT], 1):
        if not isinstance(span, dict):
            continue
        doc_id = _string(span.get("doc_id"))
        start = _string(span.get("char_start"))
        end = _string(span.get("char_end"))
        text = _preview(_string(span.get("text")), SOURCE_PREVIEW_CHARS)
        lines.append(f"  span {index}: doc={doc_id} chars={start}-{end}")
        lines.append(_indent(text or "(empty)", prefix="    "))
    return lines or ["  (none)"]


def _returned_source_lines(raw_sources: object, limit: int) -> list[str]:
    if limit <= 0:
        return ["  (source display disabled)"]
    sources = source_list(raw_sources)
    if not sources:
        return ["  (none)"]
    lines: list[str] = []
    for index, source in enumerate(sources[:limit], 1):
        title = _string(source.get("article_title") or source.get("title") or source.get("name"))
        article_id = _string(source.get("article_id") or source.get("id"))
        doc_id = _string(source.get("doc_id") or source.get("document_id"))
        score = _string(source.get("score"))
        url = _string(source.get("url") or source.get("uri"))
        text = _string(source.get("text") or source.get("snippet") or source.get("content"))
        lines.append(
            f"  source {index}: title={title or '(none)'} id={article_id or '(none)'} "
            f"doc={doc_id or '(none)'} score={score or '(none)'} url={url or '(none)'}"
        )
        if text:
            lines.append(_indent(_preview(text, SOURCE_PREVIEW_CHARS), prefix="    "))
    return lines


def _field(record: dict[str, Any], name: str, blank: str) -> str:
    value = _string(record.get(name)).strip()
    return value if value else blank


def _decision_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {
        HUMAN_DECISION_ACCEPT: 0,
        HUMAN_DECISION_PARTIAL: 0,
        HUMAN_DECISION_REJECT: 0,
    }
    for record in records:
        decision = _string(record.get(HUMAN_DECISION_FIELD)).strip().lower()
        if decision in counts:
            counts[decision] += 1
    return {key: value for key, value in counts.items() if value}


def _indent(text: str, prefix: str = "  ") -> str:
    if not text:
        return prefix.rstrip()
    return "\n".join(prefix + line for line in text.splitlines())


def _preview(text: str, limit: int) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)] + "..."


def _preview_one_line(text: str, limit: int) -> str:
    return _preview(" ".join(text.split()), limit)


def _float_text(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    if not isinstance(value, str):
        return "0.0000"
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return "0.0000"
