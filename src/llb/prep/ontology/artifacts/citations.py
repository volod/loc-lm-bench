"""PDF sidecar copying, indexing, page joins, and coverage metrics."""

import json
import shutil
from pathlib import Path
from typing import Any

from llb.goldset.schema import SourceSpan
from llb.prep.ontology.models import DocExtraction
from llb.prep.pdf.model import PDF_CITATION_SUFFIX, PDF_CORPUS_MANIFEST, PDF_CORPUS_QUALITY

CITATION_META_FILES = (PDF_CORPUS_MANIFEST, PDF_CORPUS_QUALITY)


def pdf_citation_path(corpus_root: Path | str, doc_id: str) -> Path:
    return Path(corpus_root) / Path(doc_id).with_suffix(PDF_CITATION_SUFFIX).name


def copy_pdf_citation_sidecars(
    source_root: Path | str, target_root: Path | str, doc_ids: list[str]
) -> list[str]:
    src = Path(source_root)
    dst = Path(target_root)
    copied: list[str] = []
    for doc_id in sorted(set(doc_ids)):
        source = pdf_citation_path(src, doc_id)
        if not source.is_file():
            continue
        target = dst / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(target.name)
    for name in CITATION_META_FILES:
        source = src / name
        if source.is_file():
            target = dst / name
            shutil.copyfile(source, target)
            copied.append(target.name)
    return copied


def load_pdf_citation_index(corpus_root: Path | str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(corpus_root).glob(f"*{PDF_CITATION_SUFFIX}")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        doc_id = payload.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            index[doc_id] = payload
    return index


def span_page_refs(span: SourceSpan, index: dict[str, dict[str, Any]]) -> list[dict[str, object]]:
    payload = index.get(span.doc_id)
    if not payload:
        return []
    pages = payload.get("pages")
    refs: list[dict[str, object]] = []
    for page in pages if isinstance(pages, list) else []:
        page_no = _overlapping_page_no(page, span)
        if page_no is not None:
            refs.append(
                {
                    "source": str(payload.get("source") or ""),
                    "page": page_no,
                    "parser": str(payload.get("parser") or ""),
                }
            )
    return refs


def evidence_spans(extractions: list[DocExtraction]) -> list[SourceSpan]:
    spans: list[SourceSpan] = []
    for extraction in extractions:
        for entity in extraction.entities:
            spans.extend(entity.mentions)
        spans.extend(event.evidence for event in extraction.events)
        spans.extend(claim.evidence for claim in extraction.claims)
        spans.extend(fact.evidence for fact in extraction.facts)
    return spans


def span_coverage(spans: list[SourceSpan], index: dict[str, dict[str, Any]]) -> dict[str, object]:
    covered = sum(1 for span in spans if span_page_refs(span, index))
    return {
        "n_spans": len(spans),
        "n_with_pdf_page": covered,
        "coverage": ratio(covered, len(spans)),
    }


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _overlapping_page_no(page: Any, span: SourceSpan) -> int | None:
    if not isinstance(page, dict):
        return None
    text_start = _int_or_none(page.get("text_start"))
    text_end = _int_or_none(page.get("text_end"))
    start = text_start if text_start is not None else _int_or_none(page.get("char_start"))
    end = text_end if text_end is not None else _int_or_none(page.get("char_end"))
    page_no = _int_or_none(page.get("page"))
    if start is None or end is None or page_no is None:
        return None
    return page_no if span.char_start < end and span.char_end > start else None


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
