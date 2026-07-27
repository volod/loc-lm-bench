"""Readiness audit for a widened multi-hop draft ledger."""

import json
from pathlib import Path

from llb.goldset.schema import load_goldset
from llb.goldset.validate import validate_items
from llb.prep.ontology.constants import (
    MULTI_HOP_NEAR_DUP_ANSWER_COSINE_THRESHOLD,
    NEAR_DUP_COSINE_THRESHOLD,
)
from llb.prep.ontology.inventory import inventory_corpus
from llb.prep.ontology.language import is_ukrainian_dominant
from llb.prep.ontology.pipeline.expansion import labeled_multi_hop_ids


def audit_multihop_expansion(
    bundle: Path | str, *, decision_floor: int, minimum_items: int
) -> dict[str, object]:
    """Audit the widened review ledger and return a machine-readable readiness report."""
    root = Path(bundle)
    items = load_goldset(root / "goldset.jsonl")
    labeled_ids = labeled_multi_hop_ids(root)
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    carry = provenance.get("multi_hop_carry_forward", {})
    dedup = provenance.get("dedup", {})
    validation_errors = list(validate_items(items, root / "corpus")["errors"])
    errors = list(validation_errors)
    texts = {doc.doc_id: doc.text for doc in inventory_corpus(root / "corpus")}
    all_span_exact = all(
        texts.get(span.doc_id, "")[span.char_start : span.char_end] == span.text
        for item in items
        for span in item.source_spans
    )

    item_ids = {item.id for item in items}
    if item_ids != labeled_ids:
        errors.append("goldset and labeled multi-hop sidecar ids differ")
    if any(
        len({(span.doc_id, span.char_start, span.char_end) for span in item.source_spans}) < 2
        for item in items
    ):
        errors.append("one or more multi-hop rows have fewer than two distinct exact spans")
    all_ukrainian = not any(
        not is_ukrainian_dominant(text)
        for item in items
        for text in (item.question, item.reference_answer)
    )
    if not all_ukrainian:
        errors.append("one or more multi-hop rows fail the Ukrainian output gate")
    normalized = [" ".join(item.question.casefold().split()) for item in items]
    if len(normalized) != len(set(normalized)):
        errors.append("the combined ledger has exact duplicate questions")
    if not isinstance(dedup, dict) or int(dedup.get("prior_questions", 0)) == 0:
        errors.append("the expansion has no recorded prior-question dedup")
    elif isinstance(carry, dict):
        checked = int(dedup.get("checked", 0))
        dropped = int(dedup.get("dropped", 0))
        new_items = int(carry.get("new_items", 0))
        if checked != dropped + new_items:
            errors.append("dedup accounting does not cover every newly drafted item")

    drafted = len(items)
    headroom = drafted - decision_floor
    if drafted < minimum_items:
        errors.append(f"drafted multi-hop size {drafted} is below required minimum {minimum_items}")
    return {
        "kind": "multihop-draft-expansion-audit",
        "bundle": str(root),
        "drafted_multi_hop_items": drafted,
        "decision_floor": decision_floor,
        "review_headroom": headroom,
        "minimum_items": minimum_items,
        "carried_items": int(carry.get("carried_items", 0)) if isinstance(carry, dict) else 0,
        "new_items": int(carry.get("new_items", 0)) if isinstance(carry, dict) else drafted,
        "dedup_prior_questions": (
            int(dedup.get("prior_questions", 0)) if isinstance(dedup, dict) else 0
        ),
        "dedup_dropped": int(dedup.get("dropped", 0)) if isinstance(dedup, dict) else 0,
        "dedup_question_threshold": NEAR_DUP_COSINE_THRESHOLD,
        "dedup_answer_threshold": MULTI_HOP_NEAR_DUP_ANSWER_COSINE_THRESHOLD,
        "all_rows_span_exact": all_span_exact and not validation_errors,
        "all_rows_ukrainian": all_ukrainian,
        "ready_for_human_review": not errors,
        "errors": errors,
    }
