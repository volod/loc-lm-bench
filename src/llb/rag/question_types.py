"""Question-type labels for retrieval slicing, read from an ontology bundle's needle sidecar.

Retrieval reports slice by question type (`factoid`, `comparative`, `multi-hop`, ...), but a gold
set JSONL carries no type: the label lives in the draft bundle's `needle_items.jsonl` sidecar. This
module is the one place that knows where that sidecar sits relative to a gold set -- beside it, or
one level up when the gold set is an accepted ledger under `accepted/`.
"""

import json
from pathlib import Path

SIDECAR_NAME = "needle_items.jsonl"
ACCEPTED_DIRNAME = "accepted"


def sidecar_path(goldset: Path) -> Path | None:
    """The needle sidecar for this gold set, or None when the bundle has none."""
    candidates = [goldset.parent / SIDECAR_NAME]
    if goldset.parent.name == ACCEPTED_DIRNAME:
        candidates.append(goldset.parent.parent / SIDECAR_NAME)
    return next((path for path in candidates if path.is_file()), None)


def load_question_types(goldset: Path) -> dict[str, str]:
    """Map item id -> question type from the gold set's needle sidecar ({} when absent)."""
    source = sidecar_path(goldset)
    if source is None:
        return {}
    labels: dict[str, str] = {}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = row.get("id")
            question_type = row.get("question_type")
            if isinstance(item_id, str) and isinstance(question_type, str):
                labels[item_id] = question_type
    return labels


def aligned_question_types(goldset: Path, item_ids: list[str]) -> list[str | None] | None:
    """Question types aligned one-to-one with `item_ids`, or None when no sidecar exists."""
    if sidecar_path(goldset) is None:
        return None
    labels = load_question_types(goldset)
    return [labels.get(item_id) for item_id in item_ids]
