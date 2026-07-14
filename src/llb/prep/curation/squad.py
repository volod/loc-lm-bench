"""Curate merged SQuAD-format goldset drafts (external-draft contract Artifact A).

Merges any number of exported SQuAD files (whole `{"data": ...}` documents, flattened record
lists, or fenced batches), repairs near-verbatim answers/contexts back to exact corpus text,
filters invalid and flabby items, deduplicates across services, and emits ONE SQuAD JSON ready
for `make ingest-squad`.
"""

import logging
from pathlib import Path
from typing import Any

from llb.prep.curation.common import (
    CurationReport,
    QuestionEmbedder,
    drop_exact_duplicates,
    drop_near_duplicates,
    unique_ids,
)
from llb.prep.curation.input import (
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_MIN_CONTEXT_CHARS,
    normalize_text,
)
from llb.prep.curation.squad_rows import (
    _Row,
    _is_flabby,
    _load_rows,
    _repair_answer,
    _repair_context,
)

_LOG = logging.getLogger(__name__)


def _emit(rows: list[_Row]) -> dict[str, Any]:
    """Group kept rows back into nested SQuAD: one article per title, one paragraph per context."""
    articles: dict[str, dict[str, Any]] = {}
    for row in rows:
        article = articles.setdefault(row.title, {"title": row.title, "paragraphs": []})
        paragraph = next((p for p in article["paragraphs"] if p["context"] == row.context), None)
        if paragraph is None:
            paragraph = {"context": row.context, "qas": []}
            article["paragraphs"].append(paragraph)
        start = row.context.find(row.answer)
        paragraph["qas"].append(
            {
                "id": row.item_id,
                "question": row.question,
                "answers": [{"text": row.answer, "answer_start": max(start, 0)}],
            }
        )
    return {"version": "1.0", "data": list(articles.values())}


def _row_is_valid(
    row: _Row,
    corpus_texts: dict[str, str] | None,
    min_context_chars: int,
    report: CurationReport,
) -> bool:
    """Reject empty / short rows, then repair grounding and drop flabby questions."""
    if not row.question:
        report.reject_invalid(row.item_id, row.source, "empty question")
        return False
    if not row.answer:
        report.reject_invalid(row.item_id, row.source, "empty answer")
        return False
    if len(row.context.strip()) < min_context_chars:
        report.reject_invalid(row.item_id, row.source, "context too short")
        return False
    return (
        _repair_context(row, corpus_texts, report)
        and _repair_answer(row, report)
        and not _is_flabby(row, report)
    )


def _row_ids(valid: list[_Row]) -> list[str]:
    return [r.item_id for r in valid]


def _row_sources(valid: list[_Row]) -> list[str]:
    return [r.source for r in valid]


def _span_signature(row: _Row) -> str:
    return f"{row.title}|{normalize_text(row.context)}|{normalize_text(row.answer)}"


def _dedup_rows(
    valid: list[_Row],
    *,
    embedder: QuestionEmbedder | None,
    dedup_threshold: float,
    dedup_spans: bool,
    prior_questions: list[str] | None,
    report: CurationReport,
) -> list[_Row]:
    """Exact question dedup, optional exact span dedup, then embedding near-dup filtering."""
    keep = drop_exact_duplicates(
        [normalize_text(r.question) for r in valid], report, _row_ids(valid), _row_sources(valid)
    )
    valid = [valid[i] for i in keep]
    if dedup_spans:
        keep = drop_exact_duplicates(
            [_span_signature(r) for r in valid], report, _row_ids(valid), _row_sources(valid)
        )
        valid = [valid[i] for i in keep]
    keep = drop_near_duplicates(
        [r.question for r in valid],
        embedder,
        dedup_threshold,
        report,
        _row_ids(valid),
        _row_sources(valid),
        prior_texts=prior_questions,
    )
    return [valid[i] for i in keep]


def curate_squad(
    inputs: list[Path],
    *,
    corpus_texts: dict[str, str] | None = None,
    embedder: QuestionEmbedder | None = None,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    min_context_chars: int = DEFAULT_MIN_CONTEXT_CHARS,
    dedup_spans: bool = False,
    prior_questions: list[str] | None = None,
) -> tuple[dict[str, Any], CurationReport]:
    """Merge + repair + filter + dedup SQuAD drafts; returns (merged SQuAD JSON, report)."""
    report = CurationReport(kind="squad")
    rows = _load_rows(inputs, report)
    report.loaded = len(rows)

    valid = [row for row in rows if _row_is_valid(row, corpus_texts, min_context_chars, report)]
    valid = _dedup_rows(
        valid,
        embedder=embedder,
        dedup_threshold=dedup_threshold,
        dedup_spans=dedup_spans,
        prior_questions=prior_questions,
        report=report,
    )

    final_ids = unique_ids([r.item_id for r in valid], report, [r.source for r in valid])
    for row, item_id in zip(valid, final_ids):
        row.item_id = item_id
    report.kept = len(valid)
    _LOG.info(
        "[curate] squad: kept %d/%d (%d invalid, %d flabby, %d exact-dup, %d near-dup)",
        report.kept,
        report.loaded,
        len(report.invalid),
        len(report.flabby),
        len(report.exact_duplicates),
        len(report.near_duplicates),
    )
    return _emit(valid), report
