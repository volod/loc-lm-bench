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
from typing import Any, cast

from llb.goldset.schema import GoldItem, SourceSpan
from llb.prep.ontology.constants import (
    CORPUS_DIRNAME,
    DEFAULT_QUESTION_TYPE,
    NEEDLE_GOLDSET_FILENAME,
    PDF_ONTOLOGY_REPORT_FILENAME,
    PROMPT_DICTIONARY_FILENAME,
    PROMPT_DICTIONARY_MAX_EXAMPLES,
)
from llb.prep.ontology.models import DocExtraction, DocRecord, ItemLabels, OntologyCandidate
from llb.prep.ontology.needles import NeedleRetriever, annotate_needle_retrieval
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
        page_no = _overlapping_page_no(page, span)
        if page_no is None:
            continue
        refs.append(
            {
                "source": str(payload.get("source") or ""),
                "page": page_no,
                "parser": str(payload.get("parser") or ""),
            }
        )
    return refs


def _overlapping_page_no(page: Any, span: SourceSpan) -> int | None:
    """The page number when the page's corpus char range overlaps `span`, else None."""
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


def _attach_labels(
    rows: list[dict[str, object]], item_labels: dict[str, ItemLabels] | None
) -> None:
    """Add `question_type` / `difficulty` to each needle row from its item label (yield-max)."""
    if not item_labels:
        return
    for row in rows:
        label = item_labels.get(str(row.get("id")))
        if label is not None:
            row["question_type"] = label.question_type
            row["difficulty"] = label.difficulty


def _needle_rows_and_report(
    needles: list[GoldItem],
    *,
    retrieval_store: NeedleRetriever | None,
    retrieval_k: int,
    drop_nonretrievable_needles: bool,
    item_labels: dict[str, ItemLabels] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if retrieval_store is None:
        rows = [cast(dict[str, object], item.model_dump()) for item in needles]
        report: dict[str, object] = {"enabled": False}
    else:
        rows, report = annotate_needle_retrieval(
            needles,
            retrieval_store,
            k=retrieval_k,
            drop_nonretrievable=drop_nonretrievable_needles,
        )
    _attach_labels(rows, item_labels)
    return rows, report


def _label_distributions(
    items: list[GoldItem], item_labels: dict[str, ItemLabels] | None
) -> tuple[dict[str, int], dict[str, int]]:
    """Count drafted items per question type and per difficulty from the item labels."""
    by_type: dict[str, int] = defaultdict(int)
    by_difficulty: dict[str, int] = defaultdict(int)
    labels = item_labels or {}
    for item in items:
        label = labels.get(item.id)
        by_type[label.question_type if label else DEFAULT_QUESTION_TYPE] += 1
        by_difficulty[label.difficulty if label else "medium"] += 1
    return dict(sorted(by_type.items())), dict(sorted(by_difficulty.items()))


def _retrieval_fraction_by_type(
    needle_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Per-question-type retrieval-unique needle fraction (rank found within top-k)."""
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"needles": 0, "retrievable": 0})
    for row in needle_rows:
        qtype = str(row.get("question_type") or DEFAULT_QUESTION_TYPE)
        groups[qtype]["needles"] += 1
        if row.get("retrieval_rank") is not None:
            groups[qtype]["retrievable"] += 1
    return {
        qtype: {
            "needles": g["needles"],
            "retrievable": g["retrievable"],
            "retrievable_fraction": _ratio(g["retrievable"], g["needles"]),
        }
        for qtype, g in sorted(groups.items())
    }


# The gates whose AND is the `passed` roll-up. Every corpus needs grounded evidence of ANY kind
# and a non-empty gold set; PDF-derived corpora additionally need a citation-valid needle item.
_CORPUS_REQUIRED_GATES = ("nonzero_grounded_extractions", "nonzero_draft_items")
_PDF_REQUIRED_GATE = "has_citation_valid_needles"


def required_gate_names(has_pdf_sidecars: bool) -> list[str]:
    """The single source of truth for which gates block the `passed` roll-up (see `_gates`)."""
    names = list(_CORPUS_REQUIRED_GATES)
    if has_pdf_sidecars:
        names.append(_PDF_REQUIRED_GATE)
    return names


def _gates(
    *,
    grounded_total: int,
    grounded_facts: int,
    n_items: int,
    has_dictionary: bool,
    n_needles: int,
    has_pdf_sidecars: bool,
) -> dict[str, bool]:
    """Quality gates for a drafted bundle, plus a single PDF-aware `passed` roll-up.

    `passed` is the operator/orchestration signal for "this draft is worth reviewing". It requires
    grounded evidence of ANY kind (entity/event/claim/fact -- a corpus rich in entities/claims but
    sparse in SRO relations is still valid) AND a non-empty gold set. For PDF-derived corpora it
    additionally requires at least one citation-valid needle item, since the needle-in-haystack set
    is the point of a PDF run; a plain-text corpus has no page sidecars to validate against, so that
    gate is not applicable and does not block. `nonzero_grounded_facts` stays as an informational
    signal (SRO relations power the GraphRAG store) but is no longer the sole blocker.
    """
    gates = {
        "nonzero_grounded_extractions": grounded_total > 0,
        "nonzero_grounded_facts": grounded_facts > 0,
        "nonzero_draft_items": n_items > 0,
        "has_prompt_dictionary_candidates": has_dictionary,
        "has_citation_valid_needles": n_needles > 0,
        "pdf_citation_gate_applicable": has_pdf_sidecars,
    }
    gates["passed"] = all(gates[name] for name in required_gate_names(has_pdf_sidecars))
    return gates


def write_calibration_artifacts(
    out_dir: Path | str,
    docs: list[DocRecord],
    extractions: list[DocExtraction],
    ontology: OntologyCandidate,
    items: list[GoldItem],
    *,
    elapsed_s: float,
    settings: dict[str, object],
    retrieval_store: NeedleRetriever | None = None,
    retrieval_k: int = 10,
    drop_nonretrievable_needles: bool = False,
    item_labels: dict[str, ItemLabels] | None = None,
    coverage_matrix: dict[str, object] | None = None,
    dedup_report: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write PDF calibration/report artifacts into an ontology draft bundle."""
    root = Path(out_dir)
    corpus_root = root / CORPUS_DIRNAME
    citation_index = load_pdf_citation_index(corpus_root)
    dictionary = build_prompt_dictionary_candidates(extractions, citation_index)
    needles = citation_valid_items(items, citation_index)
    needle_rows, needle_retrieval = _needle_rows_and_report(
        needles,
        retrieval_store=retrieval_store,
        retrieval_k=retrieval_k,
        drop_nonretrievable_needles=drop_nonretrievable_needles,
        item_labels=item_labels,
    )
    question_type_distribution, difficulty_distribution = _label_distributions(items, item_labels)

    _write_jsonl(dictionary, root / PROMPT_DICTIONARY_FILENAME)
    _write_jsonl(needle_rows, root / NEEDLE_GOLDSET_FILENAME)

    stats = _extraction_stats(extractions)
    evidence_spans = _evidence_spans(extractions)
    item_spans = [span for item in items for span in item.source_spans]
    grounded_facts = int(stats["grounded_facts"])

    gates = _gates(
        grounded_total=int(stats["grounded_total"]),
        grounded_facts=grounded_facts,
        n_items=len(items),
        has_dictionary=bool(dictionary),
        n_needles=len(needles),
        has_pdf_sidecars=bool(citation_index),
    )
    if needle_retrieval.get("enabled"):
        retrievable_items = needle_retrieval.get("retrievable_items")
        gates["has_retrieval_unique_needles"] = (
            retrievable_items > 0 if isinstance(retrievable_items, int) else False
        )

    nonempty_docs = int(stats["nonempty_docs"])
    report: dict[str, object] = {
        "kind": "pdf-ontology-calibration",
        "settings": settings,
        "elapsed_s": round(elapsed_s, 3),
        "documents": len(docs),
        "documents_with_nonempty_extraction": nonempty_docs,
        "parse_rate": _ratio(nonempty_docs, len(docs)),
        "grounded_entities": stats["grounded_entities"],
        "grounded_events": stats["grounded_events"],
        "grounded_facts": grounded_facts,
        "grounded_claims": stats["grounded_claims"],
        "ontology_entity_types": len(ontology.entity_types),
        "ontology_relation_types": len(ontology.relation_types),
        "draft_items": len(items),
        "pdf_sidecar_docs": len(citation_index),
        "page_span_citation_coverage": _span_coverage(evidence_spans, citation_index),
        "item_page_span_citation_coverage": _span_coverage(item_spans, citation_index),
        "citation_valid_needle_items": len(needles),
        "needle_items_written": len(needle_rows),
        "needle_retrieval": needle_retrieval,
        "retrieval_unique_needle_items": needle_retrieval.get("retrievable_items"),
        "retrieval_unique_needle_fraction": needle_retrieval.get("retrievable_fraction"),
        "dictionary_term_yield": len(dictionary),
        "question_type_distribution": question_type_distribution,
        "difficulty_distribution": difficulty_distribution,
        "facts_by_doc": stats["facts_by_doc"],
        "artifacts": {
            "prompt_dictionary_candidates": PROMPT_DICTIONARY_FILENAME,
            "needle_items": NEEDLE_GOLDSET_FILENAME,
        },
        "gates": gates,
    }
    if coverage_matrix is not None:
        report["coverage_matrix"] = coverage_matrix
    if dedup_report is not None:
        report["dedup"] = dedup_report
    if needle_retrieval.get("enabled"):
        report["retrieval_unique_needle_fraction_by_question_type"] = _retrieval_fraction_by_type(
            needle_rows
        )
    (root / PDF_ONTOLOGY_REPORT_FILENAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _extraction_stats(extractions: list[DocExtraction]) -> dict[str, Any]:
    """Grounded-object counts and per-doc fact tallies over all extractions."""
    nonempty_docs = sum(
        1
        for extraction in extractions
        if extraction.entities or extraction.facts or extraction.claims or extraction.events
    )
    facts_by_doc: dict[str, int] = defaultdict(int)
    for extraction in extractions:
        facts_by_doc[extraction.doc_id] += len(extraction.facts)
    grounded_entities = sum(len(extraction.entities) for extraction in extractions)
    grounded_events = sum(len(extraction.events) for extraction in extractions)
    grounded_facts = sum(len(extraction.facts) for extraction in extractions)
    grounded_claims = sum(len(extraction.claims) for extraction in extractions)
    return {
        "nonempty_docs": nonempty_docs,
        "grounded_entities": grounded_entities,
        "grounded_events": grounded_events,
        "grounded_facts": grounded_facts,
        "grounded_claims": grounded_claims,
        "grounded_total": grounded_entities + grounded_events + grounded_facts + grounded_claims,
        "facts_by_doc": dict(sorted(facts_by_doc.items())),
    }
