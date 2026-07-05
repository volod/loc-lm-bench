"""Merge ontology/topic inventories from multiple services into one coverage plan.

Prompt `01-ontology-inventory.md` is typically run once per service; different services surface
different entities, relations, and sensitive topics from the same corpus. Merging their
inventories widens the coverage plan that steers the drafting prompts (`02`-`04`), which directly
raises the number of distinct needles the services can draft.

Merge keys are lexical: entities by (doc, casefolded name) with the type normalized into the
project's closed vocabulary (out-of-vocabulary types collapse to MISC exactly as the local graph
lane does), relations by the normalized (subject, relation, object) triple, topics and sensitive
topics by casefolded label. When a corpus is given, grounding quotes are re-snapped to exact
corpus text and entries whose quotes cannot be located are dropped.
"""

import logging
from pathlib import Path
from typing import Any

from llb.prep.frontier import ground_span
from llb.prep.curation.common import (
    CurationReport,
    load_json_documents,
    normalize_text,
)
from llb.prep.ontology.entity_types import normalize_entity_type

_LOG = logging.getLogger(__name__)


def _merge_labels(existing: list[str], new: list[Any]) -> list[str]:
    seen = {normalize_text(t) for t in existing}
    for label in new:
        text = str(label).strip()
        if text and normalize_text(text) not in seen:
            existing.append(text)
            seen.add(normalize_text(text))
    return existing


def _ground_quote(
    entry_desc: str,
    source: str,
    quote: str,
    doc_text: str | None,
    report: CurationReport,
) -> str | None:
    """Exact-or-repaired quote, or None when the quote cannot be located in the doc."""
    if doc_text is None:
        return quote
    grounded = ground_span(doc_text, quote)
    if grounded is None:
        report.reject_invalid(entry_desc, source, "quote not found in document")
        return None
    _start, exact = grounded
    if exact != quote:
        report.note_repair(entry_desc, source, "quote re-snapped to exact document text")
    return exact


def _iter_inventory_objects(
    value: Any, source: str, report: CurationReport
) -> list[dict[str, Any]]:
    """Return inventory response objects from one parsed export value.

    NotebookLM continuation sessions are often saved as one JSON array of complete response
    objects: [{batch 1 inventory}, {batch 2 inventory}, ...]. The inventory curator treats that as
    equivalent to separate files while keeping other artifact kinds' array handling unchanged.
    """
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        report.reject_invalid("inventory", source, "inventory export is not an object")
        return []

    objects: list[dict[str, Any]] = []
    for idx, item in enumerate(value, 1):
        if isinstance(item, dict):
            objects.append(item)
        else:
            report.reject_invalid(
                f"inventory batch {idx}", source, "inventory array entry is not an object"
            )
    return objects


def _merge_entities(
    doc: dict[str, Any],
    entities: list[Any],
    doc_text: str | None,
    source: str,
    report: CurationReport,
) -> None:
    merged: dict[tuple[str, str], dict[str, Any]] = {
        (normalize_text(str(e.get("name", ""))), str(e.get("type", ""))): e for e in doc["entities"]
    }
    for raw in entities:
        if not isinstance(raw, dict) or not str(raw.get("name", "")).strip():
            report.reject_invalid(f"{doc['doc']}: entity", source, "missing entity name")
            continue
        name = str(raw["name"]).strip()
        raw_type = str(raw.get("type", ""))
        canonical = normalize_entity_type(raw_type)
        if canonical != raw_type:
            report.note_repair(f"{doc['doc']}: {name}", source, f"type {raw_type} -> {canonical}")
        quote = _ground_quote(
            f"{doc['doc']}: entity {name}", source, str(raw.get("quote", "")), doc_text, report
        )
        if quote is None:
            continue
        key = (normalize_text(name), canonical)
        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                "name": name,
                "type": canonical,
                "mentions": int(raw.get("mentions") or 1),
                "quote": quote,
            }
        else:
            existing["mentions"] = max(
                int(existing.get("mentions") or 1), int(raw.get("mentions") or 1)
            )
    doc["entities"] = list(merged.values())


def _merge_keyed_quotes(
    doc: dict[str, Any],
    field: str,
    entries: list[Any],
    key_fields: tuple[str, ...],
    doc_text: str | None,
    source: str,
    report: CurationReport,
) -> None:
    """Merge relation / numeric-fact entries keyed by their normalized identifying fields."""
    merged: dict[tuple[str, ...], dict[str, Any]] = {
        tuple(normalize_text(str(e.get(k, ""))) for k in key_fields): e for e in doc[field]
    }
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        key = tuple(normalize_text(str(raw.get(k, ""))) for k in key_fields)
        if not all(key):
            report.reject_invalid(f"{doc['doc']}: {field}", source, f"missing {key_fields} field")
            continue
        if key in merged:
            continue
        quote = _ground_quote(
            f"{doc['doc']}: {field} {key[0][:40]}",
            source,
            str(raw.get("quote", "")),
            doc_text,
            report,
        )
        if quote is None:
            continue
        entry = dict(raw)
        entry["quote"] = quote
        merged[key] = entry
    doc[field] = list(merged.values())


def curate_inventory(
    inputs: list[Path],
    *,
    corpus_texts: dict[str, str] | None = None,
) -> tuple[dict[str, Any], CurationReport]:
    """Merge inventory.json exports into one coverage plan; returns (inventory, report)."""
    report = CurationReport(kind="inventory")
    docs: dict[str, dict[str, Any]] = {}
    cross: dict[str, dict[str, Any]] = {}

    for path in inputs:
        source = str(path)
        loaded = 0
        for value in load_json_documents(path):
            for value in _iter_inventory_objects(value, source, report):
                for raw_doc in value.get("documents", []):
                    if not isinstance(raw_doc, dict) or not str(raw_doc.get("doc", "")).strip():
                        report.reject_invalid("document", source, "missing doc name")
                        continue
                    loaded += 1
                    name = str(raw_doc["doc"]).strip()
                    doc_text = corpus_texts.get(name) if corpus_texts is not None else None
                    if corpus_texts is not None and doc_text is None:
                        report.reject_invalid(name, source, "document not in corpus")
                        continue
                    doc = docs.setdefault(
                        name,
                        {
                            "doc": name,
                            "topics": [],
                            "entities": [],
                            "relations": [],
                            "numeric_facts": [],
                            "sensitive_topics": [],
                        },
                    )
                    _merge_labels(doc["topics"], raw_doc.get("topics", []))
                    _merge_labels(doc["sensitive_topics"], raw_doc.get("sensitive_topics", []))
                    _merge_entities(doc, raw_doc.get("entities", []), doc_text, source, report)
                    _merge_keyed_quotes(
                        doc,
                        "relations",
                        raw_doc.get("relations", []),
                        ("subject", "relation", "object"),
                        doc_text,
                        source,
                        report,
                    )
                    _merge_keyed_quotes(
                        doc,
                        "numeric_facts",
                        raw_doc.get("numeric_facts", []),
                        ("fact",),
                        doc_text,
                        source,
                        report,
                    )
                for raw_link in value.get("cross_document", []):
                    if not isinstance(raw_link, dict):
                        continue
                    key = normalize_text(str(raw_link.get("entity_or_topic", "")))
                    if not key:
                        continue
                    link = cross.setdefault(
                        key,
                        {
                            "entity_or_topic": str(raw_link["entity_or_topic"]).strip(),
                            "docs": [],
                            "note": str(raw_link.get("note", "")),
                        },
                    )
                    link["docs"] = _merge_labels(link["docs"], raw_link.get("docs", []))
        report.sources[source] = loaded

    report.loaded = sum(report.sources.values())
    report.kept = len(docs)
    merged = {"documents": list(docs.values()), "cross_document": list(cross.values())}
    _LOG.info(
        "[curate] inventory: %d documents merged from %d inputs (%d invalid entries)",
        len(docs),
        len(inputs),
        len(report.invalid),
    )
    return merged, report
