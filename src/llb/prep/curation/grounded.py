"""Curate merged corpus-grounded goldset drafts (external-draft contract Artifact B).

Multi-service / multi-batch exports of the grounded-JSONL shape (`quote` + `source_doc_id` per
row) are merged into ONE importable JSONL: each `quote` is re-grounded to the exact corpus text
(near-verbatim quotes re-snapped, non-verbatim rows dropped and reported), flabby rows are filtered,
and duplicate questions are removed across services -- exactly the discipline the SQuAD and chain
curators already apply. `llb import-external-draft` then reads the single curated file.
"""

import logging
from pathlib import Path
from typing import Any

from llb.prep.curation.common import (
    DEFAULT_DEDUP_THRESHOLD,
    MAX_ANSWER_CHARS,
    CurationReport,
    QuestionEmbedder,
    drop_exact_duplicates,
    drop_near_duplicates,
    load_json_documents,
    load_jsonl_rows,
    normalize_text,
    question_too_vague,
    references_document_structure,
    unique_ids,
)
from llb.prep.frontier import ground_span
from llb.prep.ontology.refine import is_circular

_LOG = logging.getLogger(__name__)


def _load_rows(inputs: list[Path], report: CurationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in inputs:
        source = str(path)
        n_before = len(rows)
        for row in load_jsonl_rows(load_json_documents(path)):
            if isinstance(row, dict):
                row = dict(row)
                row["_source"] = source
                rows.append(row)
        report.sources[source] = len(rows) - n_before
    return rows


def _row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("id") or f"grounded-{index:04d}")


def _ground_quote(
    item_id: str,
    source: str,
    row: dict[str, Any],
    corpus_texts: dict[str, str] | None,
    report: CurationReport,
) -> bool:
    """Re-ground `quote` in its named corpus doc; re-snap near-verbatim, reject non-verbatim."""
    if corpus_texts is None:
        return True
    doc_id = str(row.get("source_doc_id") or "")
    text = corpus_texts.get(doc_id)
    if text is None:
        report.reject_invalid(item_id, source, f"unknown source_doc_id {doc_id}")
        return False
    grounded = ground_span(text, str(row.get("quote", "")))
    if grounded is None:
        report.reject_invalid(item_id, source, f"quote not a verbatim substring of {doc_id}")
        return False
    _start, exact = grounded
    if exact != row.get("quote"):
        report.note_repair(item_id, source, "quote re-snapped to exact corpus text")
        row["quote"] = exact
    if not str(row.get("reference_answer") or "").strip():
        report.note_repair(item_id, source, "reference_answer set from quote")
        row["reference_answer"] = exact
    return True


def _is_flabby(item_id: str, source: str, row: dict[str, Any], report: CurationReport) -> bool:
    question = str(row.get("question", ""))
    answer = str(row.get("reference_answer") or row.get("quote") or "")
    if question_too_vague(question):
        report.reject_flabby(item_id, source, "question too short or vague")
        return True
    if references_document_structure(question):
        report.reject_flabby(item_id, source, "question references document structure")
        return True
    if is_circular(question, answer, answer):
        report.reject_flabby(item_id, source, "question leaks its answer (circular)")
        return True
    if len(str(row.get("quote", ""))) > MAX_ANSWER_CHARS:
        report.reject_flabby(item_id, source, "quote span too long for span scoring")
        return True
    return False


def curate_grounded(
    inputs: list[Path],
    *,
    corpus_texts: dict[str, str] | None = None,
    embedder: QuestionEmbedder | None = None,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    prior_questions: list[str] | None = None,
) -> tuple[list[dict[str, Any]], CurationReport]:
    """Merge + re-ground + filter + dedup grounded-JSONL drafts; returns (rows, report)."""
    report = CurationReport(kind="grounded")
    rows = _load_rows(inputs, report)
    report.loaded = len(rows)

    valid: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        item_id = _row_id(row, i)
        source = row["_source"]
        if not str(row.get("question") or "").strip():
            report.reject_invalid(item_id, source, "empty question")
            continue
        if not str(row.get("quote") or "").strip():
            report.reject_invalid(item_id, source, "empty quote")
            continue
        if not str(row.get("source_doc_id") or "").strip():
            report.reject_invalid(item_id, source, "no source_doc_id")
            continue
        if not _ground_quote(item_id, source, row, corpus_texts, report):
            continue
        if _is_flabby(item_id, source, row, report):
            continue
        valid.append(row)

    ids = [_row_id(r, i) for i, r in enumerate(valid)]
    sources = [r["_source"] for r in valid]
    keep = drop_exact_duplicates(
        [normalize_text(str(r.get("question", ""))) for r in valid], report, ids, sources
    )
    valid = [valid[i] for i in keep]
    keep = drop_near_duplicates(
        [str(r.get("question", "")) for r in valid],
        embedder,
        dedup_threshold,
        report,
        [_row_id(r, i) for i, r in enumerate(valid)],
        [r["_source"] for r in valid],
        prior_texts=prior_questions,
    )
    valid = [valid[i] for i in keep]

    final_ids = unique_ids(
        [_row_id(r, i) for i, r in enumerate(valid)], report, [r["_source"] for r in valid]
    )
    out: list[dict[str, Any]] = []
    for row, item_id in zip(valid, final_ids):
        cleaned = {k: v for k, v in row.items() if k != "_source"}
        cleaned["id"] = item_id
        out.append(cleaned)
    report.kept = len(out)
    _LOG.info(
        "[curate] grounded: kept %d/%d (%d invalid, %d flabby, %d exact-dup, %d near-dup)",
        report.kept,
        report.loaded,
        len(report.invalid),
        len(report.flabby),
        len(report.exact_duplicates),
        len(report.near_duplicates),
    )
    return out, report
