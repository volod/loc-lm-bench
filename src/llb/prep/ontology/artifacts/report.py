"""Calibration gates, aggregate report construction, and artifact persistence."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from llb.goldset.schema import GoldItem
from llb.prep.ontology.artifacts.citations import (
    evidence_spans,
    load_pdf_citation_index,
    ratio,
    span_coverage,
)
from llb.prep.ontology.artifacts.dictionary import build_prompt_dictionary_candidates
from llb.prep.ontology.artifacts.needles import (
    citation_valid_items,
    label_distributions,
    needle_rows_and_report,
    retrieval_fraction_by_type,
    write_jsonl,
)
from llb.prep.ontology.constants import (
    CORPUS_DIRNAME,
    NEEDLE_GOLDSET_FILENAME,
    PDF_ONTOLOGY_REPORT_FILENAME,
    PROMPT_DICTIONARY_FILENAME,
)
from llb.prep.ontology.models import DocExtraction, DocRecord, ItemLabels, OntologyCandidate
from llb.prep.ontology.needles import NeedleRetriever

CORPUS_REQUIRED_GATES = ("nonzero_grounded_extractions", "nonzero_draft_items")
PDF_REQUIRED_GATE = "has_citation_valid_needles"


def required_gate_names(has_pdf_sidecars: bool) -> list[str]:
    names = list(CORPUS_REQUIRED_GATES)
    if has_pdf_sidecars:
        names.append(PDF_REQUIRED_GATE)
    return names


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
    root = Path(out_dir)
    citation_index = load_pdf_citation_index(root / CORPUS_DIRNAME)
    dictionary = build_prompt_dictionary_candidates(extractions, citation_index)
    needles = citation_valid_items(items, citation_index)
    needle_rows, needle_retrieval = needle_rows_and_report(
        needles,
        retrieval_store=retrieval_store,
        retrieval_k=retrieval_k,
        drop_nonretrievable_needles=drop_nonretrievable_needles,
        item_labels=item_labels,
    )
    question_types, difficulties = label_distributions(items, item_labels)
    write_jsonl(dictionary, root / PROMPT_DICTIONARY_FILENAME)
    write_jsonl(needle_rows, root / NEEDLE_GOLDSET_FILENAME)
    stats = extraction_stats(extractions)
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
        retrievable = needle_retrieval.get("retrievable_items")
        gates["has_retrieval_unique_needles"] = (
            retrievable > 0 if isinstance(retrievable, int) else False
        )
    report: dict[str, object] = {
        "kind": "pdf-ontology-calibration",
        "settings": settings,
        "elapsed_s": round(elapsed_s, 3),
        "documents": len(docs),
        "documents_with_nonempty_extraction": stats["nonempty_docs"],
        "parse_rate": ratio(int(stats["nonempty_docs"]), len(docs)),
        "grounded_entities": stats["grounded_entities"],
        "grounded_events": stats["grounded_events"],
        "grounded_facts": grounded_facts,
        "grounded_claims": stats["grounded_claims"],
        "ontology_entity_types": len(ontology.entity_types),
        "ontology_relation_types": len(ontology.relation_types),
        "draft_items": len(items),
        "pdf_sidecar_docs": len(citation_index),
        "page_span_citation_coverage": span_coverage(evidence_spans(extractions), citation_index),
        "item_page_span_citation_coverage": span_coverage(item_spans, citation_index),
        "citation_valid_needle_items": len(needles),
        "needle_items_written": len(needle_rows),
        "needle_retrieval": needle_retrieval,
        "retrieval_unique_needle_items": needle_retrieval.get("retrievable_items"),
        "retrieval_unique_needle_fraction": needle_retrieval.get("retrievable_fraction"),
        "dictionary_term_yield": len(dictionary),
        "question_type_distribution": question_types,
        "difficulty_distribution": difficulties,
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
        report["retrieval_unique_needle_fraction_by_question_type"] = retrieval_fraction_by_type(
            needle_rows
        )
    (root / PDF_ONTOLOGY_REPORT_FILENAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _gates(
    *,
    grounded_total: int,
    grounded_facts: int,
    n_items: int,
    has_dictionary: bool,
    n_needles: int,
    has_pdf_sidecars: bool,
) -> dict[str, bool]:
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


def extraction_stats(extractions: list[DocExtraction]) -> dict[str, Any]:
    nonempty_docs = sum(
        1
        for extraction in extractions
        if extraction.entities or extraction.facts or extraction.claims or extraction.events
    )
    facts_by_doc: dict[str, int] = defaultdict(int)
    for extraction in extractions:
        facts_by_doc[extraction.doc_id] += len(extraction.facts)
    counts = {
        "grounded_entities": sum(len(item.entities) for item in extractions),
        "grounded_events": sum(len(item.events) for item in extractions),
        "grounded_facts": sum(len(item.facts) for item in extractions),
        "grounded_claims": sum(len(item.claims) for item in extractions),
    }
    return {
        "nonempty_docs": nonempty_docs,
        **counts,
        "grounded_total": sum(counts.values()),
        "facts_by_doc": dict(sorted(facts_by_doc.items())),
    }
