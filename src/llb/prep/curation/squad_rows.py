"""Focused squad rows implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from llb.prep.frontier import ground_span
from llb.prep.curation.common import CurationReport
from llb.prep.curation.input import (
    MAX_ANSWER_CHARS,
    MAX_ANSWER_CONTEXT_FRACTION,
    load_json_documents,
    question_too_vague,
    references_document_structure,
)
from llb.prep.ontology.refine import is_circular


@dataclass
class _Row:
    """One flattened QA row with its curation lineage."""

    item_id: str
    source: str
    title: str
    context: str
    question: str
    answer: str


def _articles_from(value: Any) -> list[dict[str, Any]] | None:
    """The nested-SQuAD article list, or None when `value` is not nested SQuAD."""
    if isinstance(value, dict) and "data" in value:
        return value["data"] if isinstance(value["data"], list) else [value["data"]]
    if isinstance(value, dict) and "paragraphs" in value:
        return [value]
    if (
        isinstance(value, list)
        and value
        and all(isinstance(v, dict) and "paragraphs" in v for v in value)
    ):
        return value
    return None


def _article_records(article: dict[str, Any]) -> list[dict[str, Any]]:
    """Flattened QA records of one nested-SQuAD article, `title` preserved."""
    title = str(article.get("title") or "")
    records: list[dict[str, Any]] = []
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


def _flatten(value: Any) -> list[dict[str, Any]]:
    """Flatten nested SQuAD (`{"data": [...]}` / a bare article) or pass flattened records
    through. Unlike `ingest_squad.normalize`, the article `title` (the staged corpus doc id)
    is preserved -- curation grounds contexts against it."""
    articles = _articles_from(value)
    if articles is not None:
        return [record for article in articles for record in _article_records(article)]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]  # already-flattened records
    return []


def _answer_texts(rec: dict[str, Any]) -> list[str]:
    answers = rec.get("answers") or {}
    if isinstance(answers, list):  # flattened records may carry a raw answers list
        return [a.get("text", "") for a in answers if isinstance(a, dict)]
    return answers.get("text") or []


def _row_from_record(rec: dict[str, Any], path: Path, source: str, index: int) -> _Row:
    texts = _answer_texts(rec)
    return _Row(
        item_id=str(rec.get("id") or f"{path.stem}-{index:04d}"),
        source=source,
        title=str(rec.get("title") or ""),
        context=str(rec.get("context") or ""),
        question=str(rec.get("question") or "").strip(),
        answer=str(texts[0]) if texts else "",
    )


def _load_rows(inputs: list[Path], report: CurationReport) -> list[_Row]:
    rows: list[_Row] = []
    for path in inputs:
        source = str(path)
        n_before = len(rows)
        for value in load_json_documents(path):
            for rec in _flatten(value):
                rows.append(_row_from_record(rec, path, source, len(rows)))
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
