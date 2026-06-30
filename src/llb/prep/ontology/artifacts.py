"""PDF-aware artifacts for ontology-assisted draft bundles.

The drafting pipeline already produces canonical `goldset.jsonl`, `ontology.json`, and
`extraction.jsonl`. This module adds the PDF-corpus calibration layer: citation-sidecar coverage,
source-backed prompt dictionary candidates, and citation-valid "needle" items for human review.
Everything is deterministic and local; no model calls happen here.
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset
from llb.prep.ontology.constants import (
    CORPUS_DIRNAME,
    NEEDLE_GOLDSET_FILENAME,
    PDF_ONTOLOGY_REPORT_FILENAME,
    PROMPT_DICTIONARY_FILENAME,
    PROMPT_DICTIONARY_MAX_EXAMPLES,
)
from llb.prep.ontology.models import DocExtraction, DocRecord, OntologyCandidate
from llb.prep.pdf_corpus import PDF_CITATION_SUFFIX, PDF_CORPUS_MANIFEST, PDF_CORPUS_QUALITY

_CITATION_META_FILES = (PDF_CORPUS_MANIFEST, PDF_CORPUS_QUALITY)


def pdf_citation_path(corpus_root: Path | str, doc_id: str) -> Path:
    """Return the sidecar path for a PDF-derived corpus document id."""
    return Path(corpus_root) / Path(doc_id).with_suffix(PDF_CITATION_SUFFIX).name


def copy_pdf_citation_sidecars(
    source_root: Path | str, target_root: Path | str, doc_ids: list[str]
) -> list[str]:
    """Copy PDF citation sidecars and corpus PDF metadata into a draft bundle corpus directory."""
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
    for name in _CITATION_META_FILES:
        source = src / name
        if source.is_file():
            target = dst / name
            shutil.copyfile(source, target)
            copied.append(target.name)
    return copied


def load_pdf_citation_index(corpus_root: Path | str) -> dict[str, dict[str, Any]]:
    """Load PDF citation sidecars as `doc_id -> sidecar payload`."""
    root = Path(corpus_root)
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob(f"*{PDF_CITATION_SUFFIX}")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        doc_id = payload.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            index[doc_id] = payload
    return index


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def span_page_refs(span: SourceSpan, index: dict[str, dict[str, Any]]) -> list[dict[str, object]]:
    """Return PDF page references whose generated-corpus range overlaps `span`."""
    payload = index.get(span.doc_id)
    if not payload:
        return []
    pages = payload.get("pages")
    refs: list[dict[str, object]] = []
    for page in pages if isinstance(pages, list) else []:
        if not isinstance(page, dict):
            continue
        text_start = _int_or_none(page.get("text_start"))
        text_end = _int_or_none(page.get("text_end"))
        start = text_start if text_start is not None else _int_or_none(page.get("char_start"))
        end = text_end if text_end is not None else _int_or_none(page.get("char_end"))
        page_no = _int_or_none(page.get("page"))
        if start is None or end is None or page_no is None:
            continue
        if span.char_start < end and span.char_end > start:
            refs.append(
                {
                    "source": str(payload.get("source") or ""),
                    "page": page_no,
                    "parser": str(payload.get("parser") or ""),
                }
            )
    return refs


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _evidence_spans(extractions: list[DocExtraction]) -> list[SourceSpan]:
    spans: list[SourceSpan] = []
    for extraction in extractions:
        for entity in extraction.entities:
            spans.extend(entity.mentions)
        spans.extend(event.evidence for event in extraction.events)
        spans.extend(claim.evidence for claim in extraction.claims)
        spans.extend(fact.evidence for fact in extraction.facts)
    return spans


def _span_coverage(spans: list[SourceSpan], index: dict[str, dict[str, Any]]) -> dict[str, object]:
    covered = sum(1 for span in spans if span_page_refs(span, index))
    return {
        "n_spans": len(spans),
        "n_with_pdf_page": covered,
        "coverage": _ratio(covered, len(spans)),
    }


def _example(span: SourceSpan, index: dict[str, dict[str, Any]]) -> dict[str, object]:
    return {
        "doc_id": span.doc_id,
        "char_start": span.char_start,
        "char_end": span.char_end,
        "text": span.text,
        "pdf_pages": span_page_refs(span, index),
    }


def _add_candidate(
    rows: dict[tuple[str, str, str], dict[str, Any]],
    *,
    kind: str,
    term: str,
    ontology_type: str,
    span: SourceSpan,
    index: dict[str, dict[str, Any]],
    aliases: list[str] | None = None,
    subject: str | None = None,
    object_: str | None = None,
) -> None:
    clean_term = " ".join(term.split())
    if not clean_term:
        return
    key = (kind, ontology_type, clean_term.casefold())
    row = rows.setdefault(
        key,
        {
            "term": clean_term,
            "kind": kind,
            "ontology_type": ontology_type,
            "support_count": 0,
            "doc_count": 0,
            "aliases": [],
            "examples": [],
        },
    )
    row["support_count"] += 1
    docs = row.setdefault("_docs", set())
    docs.add(span.doc_id)
    if subject is not None:
        row.setdefault("subjects", set()).add(subject)
    if object_ is not None:
        row.setdefault("objects", set()).add(object_)
    for alias in aliases or []:
        if alias and alias not in row["aliases"]:
            row["aliases"].append(alias)
    if len(row["examples"]) < PROMPT_DICTIONARY_MAX_EXAMPLES:
        row["examples"].append(_example(span, index))


def _row_int(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    return value if isinstance(value, int) else 0


def build_prompt_dictionary_candidates(
    extractions: list[DocExtraction], index: dict[str, dict[str, Any]]
) -> list[dict[str, object]]:
    """Build source-backed prompt dictionary candidates from entities and relation evidence."""
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for extraction in extractions:
        for entity in extraction.entities:
            for mention in entity.mentions:
                _add_candidate(
                    rows,
                    kind="entity",
                    term=entity.name,
                    ontology_type=entity.type,
                    span=mention,
                    index=index,
                    aliases=entity.aliases,
                )
        for fact in extraction.facts:
            _add_candidate(
                rows,
                kind="relation",
                term=fact.relation,
                ontology_type="relation",
                span=fact.evidence,
                index=index,
                subject=fact.subject,
                object_=fact.object,
            )

    out: list[dict[str, object]] = []
    for row in rows.values():
        docs = row.pop("_docs")
        row["doc_count"] = len(docs)
        if "subjects" in row:
            row["subjects"] = sorted(row["subjects"])
        if "objects" in row:
            row["objects"] = sorted(row["objects"])
        out.append(row)
    out.sort(
        key=lambda row: (
            -_row_int(row, "support_count"),
            -_row_int(row, "doc_count"),
            str(row["kind"]),
            str(row["term"]).casefold(),
        )
    )
    return out


def citation_valid_items(items: list[GoldItem], index: dict[str, dict[str, Any]]) -> list[GoldItem]:
    """Return drafted gold items whose every source span maps to at least one PDF page sidecar."""
    return [
        item for item in items if all(span_page_refs(span, index) for span in item.source_spans)
    ]


def _write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_calibration_artifacts(
    out_dir: Path | str,
    docs: list[DocRecord],
    extractions: list[DocExtraction],
    ontology: OntologyCandidate,
    items: list[GoldItem],
    *,
    elapsed_s: float,
    settings: dict[str, object],
) -> dict[str, object]:
    """Write PDF calibration/report artifacts into an ontology draft bundle."""
    root = Path(out_dir)
    corpus_root = root / CORPUS_DIRNAME
    citation_index = load_pdf_citation_index(corpus_root)
    dictionary = build_prompt_dictionary_candidates(extractions, citation_index)
    needles = citation_valid_items(items, citation_index)

    _write_jsonl(dictionary, root / PROMPT_DICTIONARY_FILENAME)
    dump_goldset(needles, root / NEEDLE_GOLDSET_FILENAME)

    nonempty_docs = sum(
        1
        for extraction in extractions
        if extraction.entities or extraction.facts or extraction.claims
    )
    evidence_spans = _evidence_spans(extractions)
    item_spans = [span for item in items for span in item.source_spans]
    facts_by_doc: dict[str, int] = defaultdict(int)
    for extraction in extractions:
        facts_by_doc[extraction.doc_id] += len(extraction.facts)

    report: dict[str, object] = {
        "kind": "pdf-ontology-calibration",
        "settings": settings,
        "elapsed_s": round(elapsed_s, 3),
        "documents": len(docs),
        "documents_with_nonempty_extraction": nonempty_docs,
        "parse_rate": _ratio(nonempty_docs, len(docs)),
        "grounded_entities": sum(len(extraction.entities) for extraction in extractions),
        "grounded_facts": sum(len(extraction.facts) for extraction in extractions),
        "grounded_claims": sum(len(extraction.claims) for extraction in extractions),
        "ontology_entity_types": len(ontology.entity_types),
        "ontology_relation_types": len(ontology.relation_types),
        "draft_items": len(items),
        "pdf_sidecar_docs": len(citation_index),
        "page_span_citation_coverage": _span_coverage(evidence_spans, citation_index),
        "item_page_span_citation_coverage": _span_coverage(item_spans, citation_index),
        "citation_valid_needle_items": len(needles),
        "dictionary_term_yield": len(dictionary),
        "facts_by_doc": dict(sorted(facts_by_doc.items())),
        "artifacts": {
            "prompt_dictionary_candidates": PROMPT_DICTIONARY_FILENAME,
            "needle_items": NEEDLE_GOLDSET_FILENAME,
        },
        "gates": {
            "nonzero_grounded_facts": bool(sum(len(e.facts) for e in extractions) > 0),
            "has_prompt_dictionary_candidates": bool(dictionary),
            "has_citation_valid_needles": bool(needles),
        },
    }
    (root / PDF_ONTOLOGY_REPORT_FILENAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
