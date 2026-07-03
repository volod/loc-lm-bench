"""Curate merged SQuAD-format goldset drafts (external-draft contract Artifact A).

Merges any number of exported SQuAD files (whole `{"data": ...}` documents, flattened record
lists, or fenced batches), repairs near-verbatim answers/contexts back to exact corpus text,
filters invalid and flabby items, deduplicates across services, and emits ONE SQuAD JSON ready
for `make ingest-squad`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.prep.frontier import ground_span
from llb.prep.curation.common import (
    CurationReport,
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_MIN_CONTEXT_CHARS,
    MAX_ANSWER_CHARS,
    MAX_ANSWER_CONTEXT_FRACTION,
    QuestionEmbedder,
    drop_exact_duplicates,
    drop_near_duplicates,
    load_json_documents,
    normalize_text,
    question_too_vague,
    references_document_structure,
    unique_ids,
)
from llb.prep.ontology.refine import is_circular

_LOG = logging.getLogger(__name__)


@dataclass
class _Row:
    """One flattened QA row with its curation lineage."""

    item_id: str
    source: str
    title: str
    context: str
    question: str
    answer: str


def _flatten(value: Any) -> list[dict[str, Any]]:
    """Flatten nested SQuAD (`{"data": [...]}` / a bare article) or pass flattened records
    through. Unlike `ingest_squad.normalize`, the article `title` (the staged corpus doc id)
    is preserved -- curation grounds contexts against it."""
    if isinstance(value, dict) and "data" in value:
        articles = value["data"] if isinstance(value["data"], list) else [value["data"]]
    elif isinstance(value, dict) and "paragraphs" in value:
        articles = [value]
    elif isinstance(value, list):
        if value and all(isinstance(v, dict) and "paragraphs" in v for v in value):
            articles = value
        else:
            return [v for v in value if isinstance(v, dict)]  # already-flattened records
    else:
        return []
    records: list[dict[str, Any]] = []
    for article in articles:
        title = str(article.get("title") or "")
        for para in article.get("paragraphs", []):
            for qa in para.get("qas", []):
                answers = qa.get("answers") or []
                records.append(
                    {
                        "id": qa.get("id"),
                        "title": title,
                        "context": para.get("context", ""),
                        "question": qa.get("question", ""),
                        "answers": {"text": [a.get("text", "") for a in answers]},
                    }
                )
    return records


def _load_rows(inputs: list[Path], report: CurationReport) -> list[_Row]:
    rows: list[_Row] = []
    for path in inputs:
        source = str(path)
        n_before = len(rows)
        for value in load_json_documents(path):
            for rec in _flatten(value):
                answers = rec.get("answers") or {}
                if isinstance(answers, list):  # flattened records may carry a raw answers list
                    texts = [a.get("text", "") for a in answers if isinstance(a, dict)]
                else:
                    texts = answers.get("text") or []
                rows.append(
                    _Row(
                        item_id=str(rec.get("id") or f"{path.stem}-{len(rows):04d}"),
                        source=source,
                        title=str(rec.get("title") or ""),
                        context=str(rec.get("context") or ""),
                        question=str(rec.get("question") or "").strip(),
                        answer=str(texts[0]) if texts else "",
                    )
                )
        report.sources[source] = len(rows) - n_before
    return rows


def _repair_context(row: _Row, corpus_texts: dict[str, str] | None, report: CurationReport) -> bool:
    """Re-ground the context in the staged corpus (named doc first, then all docs).

    Returns False (invalid) when the context cannot be located; repairs `row.context` to the
    exact corpus substring and fixes a wrong/missing `title` when found elsewhere.
    """
    if corpus_texts is None:
        return True
    search_order = [row.title] if row.title in corpus_texts else []
    search_order += [doc for doc in corpus_texts if doc not in search_order]
    for doc in search_order:
        grounded = ground_span(corpus_texts[doc], row.context)
        if grounded is None:
            continue
        _start, exact = grounded
        if exact != row.context:
            report.note_repair(row.item_id, row.source, "context re-snapped to exact corpus text")
            row.context = exact
        if doc != row.title:
            report.note_repair(row.item_id, row.source, f"title corrected to {doc}")
            row.title = doc
        return True
    report.reject_invalid(row.item_id, row.source, "context not found in corpus")
    return False


def _repair_answer(row: _Row, report: CurationReport) -> bool:
    """Ensure the answer is an exact substring of the (possibly repaired) context."""
    if row.answer and row.answer in row.context:
        return True
    grounded = ground_span(row.context, row.answer)
    if grounded is None:
        report.reject_invalid(row.item_id, row.source, "answer is not a substring of its context")
        return False
    _start, exact = grounded
    report.note_repair(row.item_id, row.source, "answer re-snapped to exact context text")
    row.answer = exact
    return True


def _is_flabby(row: _Row, report: CurationReport) -> bool:
    if question_too_vague(row.question):
        report.reject_flabby(row.item_id, row.source, "question too short or vague")
        return True
    if references_document_structure(row.question):
        report.reject_flabby(row.item_id, row.source, "question references document structure")
        return True
    if is_circular(row.question, row.answer, row.answer):
        report.reject_flabby(row.item_id, row.source, "question leaks its answer (circular)")
        return True
    if len(row.answer) > MAX_ANSWER_CHARS or (
        len(row.context) > 0 and len(row.answer) / len(row.context) > MAX_ANSWER_CONTEXT_FRACTION
    ):
        report.reject_flabby(row.item_id, row.source, "answer span too long for span scoring")
        return True
    return False


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

    valid: list[_Row] = []
    for row in rows:
        if not row.question:
            report.reject_invalid(row.item_id, row.source, "empty question")
            continue
        if not row.answer:
            report.reject_invalid(row.item_id, row.source, "empty answer")
            continue
        if len(row.context.strip()) < min_context_chars:
            report.reject_invalid(row.item_id, row.source, "context too short")
            continue
        if not _repair_context(row, corpus_texts, report):
            continue
        if not _repair_answer(row, report):
            continue
        if _is_flabby(row, report):
            continue
        valid.append(row)

    ids = [r.item_id for r in valid]
    sources = [r.source for r in valid]
    keep = drop_exact_duplicates([normalize_text(r.question) for r in valid], report, ids, sources)
    valid = [valid[i] for i in keep]
    if dedup_spans:
        keep = drop_exact_duplicates(
            [f"{r.title}|{normalize_text(r.context)}|{normalize_text(r.answer)}" for r in valid],
            report,
            [r.item_id for r in valid],
            [r.source for r in valid],
        )
        valid = [valid[i] for i in keep]
    keep = drop_near_duplicates(
        [r.question for r in valid],
        embedder,
        dedup_threshold,
        report,
        [r.item_id for r in valid],
        [r.source for r in valid],
        prior_texts=prior_questions,
    )
    valid = [valid[i] for i in keep]

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
