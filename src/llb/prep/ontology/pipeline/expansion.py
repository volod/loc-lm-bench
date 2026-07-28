"""Reuse and carry-forward helpers for widening a drafted multi-hop slice."""

import json
from itertools import combinations
from pathlib import Path

from llb.goldset.schema import GoldItem, SourceSpan, load_goldset
from llb.graph.ingest import load_extractions
from llb.prep.ontology.constants import (
    EXTRACTION_FILENAME,
    NEEDLE_GOLDSET_FILENAME,
    PROVENANCE_FILENAME,
)
from llb.prep.ontology.models import DocExtraction, DocRecord, ItemLabels

_MULTI_HOP = "multi-hop"
_HARD = "hard"
SpanKey = tuple[str, int, int]
SpanPair = tuple[SpanKey, SpanKey]


def _source_document_fingerprints(bundle: Path) -> dict[str, tuple[str, int]]:
    provenance_path = bundle / PROVENANCE_FILENAME
    if not provenance_path.is_file():
        raise ValueError(f"reuse extraction bundle has no {PROVENANCE_FILENAME}: {bundle}")
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    rows = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"reuse extraction bundle has no document fingerprints: {bundle}")
    fingerprints: dict[str, tuple[str, int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"reuse extraction bundle has invalid document fingerprints: {bundle}")
        doc_id = row.get("doc_id")
        sha256 = row.get("sha256")
        n_chars = row.get("n_chars")
        if (
            not isinstance(doc_id, str)
            or not isinstance(sha256, str)
            or not isinstance(n_chars, int)
        ):
            raise ValueError(f"reuse extraction bundle has invalid document fingerprints: {bundle}")
        if doc_id in fingerprints:
            raise ValueError(f"reuse extraction bundle has duplicate document ids: {bundle}")
        fingerprints[doc_id] = (sha256, n_chars)
    return fingerprints


def _fingerprint_mismatch(
    current: dict[str, tuple[str, int]], expected: dict[str, tuple[str, int]]
) -> tuple[list[str], list[str], list[str]]:
    changed = sorted(
        doc_id for doc_id in current.keys() & expected.keys() if current[doc_id] != expected[doc_id]
    )
    missing = sorted(expected.keys() - current.keys())
    extra = sorted(current.keys() - expected.keys())
    return changed, missing, extra


def _validate_extraction_ids(
    extractions: list[DocExtraction], doc_ids: set[str], source: Path
) -> None:
    extraction_ids = [extraction.doc_id for extraction in extractions]
    if len(extraction_ids) != len(set(extraction_ids)):
        raise ValueError(f"reuse extraction bundle has duplicate document ids: {source}")
    if set(extraction_ids) == doc_ids:
        return
    missing = sorted(doc_ids - set(extraction_ids))
    extra = sorted(set(extraction_ids) - doc_ids)
    raise ValueError(
        "reuse extraction bundle does not match the current corpus "
        f"(missing={missing}, extra={extra}): {source}"
    )


def _extraction_spans(extraction: DocExtraction) -> list[SourceSpan]:
    return [
        *(span for entity in extraction.entities for span in entity.mentions),
        *(event.evidence for event in extraction.events),
        *(claim.evidence for claim in extraction.claims),
        *(fact.evidence for fact in extraction.facts),
    ]


def _validate_extraction_spans(extractions: list[DocExtraction], texts: dict[str, str]) -> None:
    for extraction in extractions:
        for span in _extraction_spans(extraction):
            text = texts[span.doc_id]
            if text[span.char_start : span.char_end] != span.text:
                raise ValueError(
                    "reuse extraction span does not ground in the current corpus: "
                    f"{span.doc_id}:{span.char_start}-{span.char_end}"
                )


def reused_extractions(bundle: Path | str, docs: list[DocRecord]) -> list[DocExtraction]:
    """Load a prior bundle's extraction only when it exactly covers the current corpus."""
    source = Path(bundle)
    expected = _source_document_fingerprints(source)
    current = {doc.doc_id: (doc.sha256, doc.n_chars) for doc in docs}
    if current != expected:
        changed, missing, extra = _fingerprint_mismatch(current, expected)
        raise ValueError(
            "reuse extraction bundle does not match the current corpus fingerprints "
            f"(changed={changed}, missing={missing}, extra={extra}): {source}"
        )
    extractions = load_extractions(source / EXTRACTION_FILENAME)
    doc_ids = {doc.doc_id for doc in docs}
    _validate_extraction_ids(extractions, doc_ids, source)
    texts = {doc.doc_id: doc.text for doc in docs}
    _validate_extraction_spans(extractions, texts)
    return extractions


def labeled_multi_hop_ids(bundle: Path) -> set[str]:
    sidecar = bundle / NEEDLE_GOLDSET_FILENAME
    if not sidecar.is_file():
        raise ValueError(f"carry-forward bundle has no {NEEDLE_GOLDSET_FILENAME}: {bundle}")
    ids: set[str] = set()
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("question_type") == _MULTI_HOP:
            ids.add(str(row["id"]))
    return ids


def prior_multihop_span_pairs(bundles: list[Path | str]) -> set[SpanPair]:
    """Evidence pairs already represented in prior labeled multi-hop bundles."""
    pairs: set[SpanPair] = set()
    for raw_bundle in bundles:
        bundle = Path(raw_bundle)
        if not (bundle / NEEDLE_GOLDSET_FILENAME).is_file():
            continue
        selected_ids = labeled_multi_hop_ids(bundle)
        for item in load_goldset(bundle / "goldset.jsonl"):
            if item.id not in selected_ids:
                continue
            keys = sorted(
                {(span.doc_id, span.char_start, span.char_end) for span in item.source_spans}
            )
            pairs.update(combinations(keys, 2))
    return pairs


def _validate_carried_item(item: GoldItem, texts: dict[str, str], bundle: Path) -> None:
    if len(item.source_spans) < 2:
        raise ValueError(f"carry-forward multi-hop item has fewer than two spans: {item.id}")
    for span in item.source_spans:
        text = texts.get(span.doc_id)
        if text is None or text[span.char_start : span.char_end] != span.text:
            raise ValueError(
                f"carry-forward item {item.id} does not ground in the current corpus: {bundle}"
            )


def _selected_multi_hop_items(bundle: Path) -> list[GoldItem]:
    selected_ids = labeled_multi_hop_ids(bundle)
    selected = [item for item in load_goldset(bundle / "goldset.jsonl") if item.id in selected_ids]
    if len(selected) != len(selected_ids):
        raise ValueError(f"carry-forward labels do not match goldset rows: {bundle}")
    return selected


def _collect_carried_items(
    bundles: list[Path | str], texts: dict[str, str]
) -> tuple[list[GoldItem], set[str], list[dict[str, object]], int]:
    carried: list[GoldItem] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    sources: list[dict[str, object]] = []
    dropped = 0
    for raw_bundle in bundles:
        bundle = Path(raw_bundle)
        selected = _selected_multi_hop_items(bundle)
        for item in selected:
            _validate_carried_item(item, texts, bundle)
            normalized = _normalized_question(item.question)
            if normalized in seen_questions:
                dropped += 1
                continue
            if item.id in seen_ids:
                raise ValueError(f"duplicate carried multi-hop item id: {item.id}")
            seen_ids.add(item.id)
            seen_questions.add(normalized)
            carried.append(item)
        sources.append({"bundle": str(bundle), "labeled_items": len(selected)})
    return carried, seen_ids, sources, dropped


def _rewrite_colliding_ids(
    items: list[GoldItem], labels: dict[str, ItemLabels], seen_ids: set[str]
) -> int:
    rewritten = 0
    for index, item in enumerate(items):
        if item.id not in seen_ids:
            seen_ids.add(item.id)
            continue
        old_id = item.id
        item.id = f"{old_id}-expansion-{index}"
        labels[item.id] = labels.pop(old_id)
        seen_ids.add(item.id)
        rewritten += 1
    return rewritten


def carry_forward_multi_hop(
    items: list[GoldItem],
    labels: dict[str, ItemLabels],
    bundles: list[Path | str],
    docs: list[DocRecord],
) -> tuple[list[GoldItem], dict[str, ItemLabels], dict[str, object]]:
    """Prepend prior labeled multi-hop rows, keeping one collision-free review ledger."""
    texts = {doc.doc_id: doc.text for doc in docs}
    carried, seen_ids, sources, dropped_exact_duplicates = _collect_carried_items(bundles, texts)
    rewritten = _rewrite_colliding_ids(items, labels, seen_ids)

    carried_labels = {
        item.id: ItemLabels(question_type=_MULTI_HOP, difficulty=_HARD) for item in carried
    }
    report: dict[str, object] = {
        "enabled": True,
        "sources": sources,
        "carried_items": len(carried),
        "dropped_carried_exact_duplicates": dropped_exact_duplicates,
        "new_items": len(items),
        "combined_items": len(carried) + len(items),
        "rewritten_new_ids": rewritten,
    }
    return carried + items, {**carried_labels, **labels}, report


def _normalized_question(text: str) -> str:
    return " ".join(text.casefold().split())
