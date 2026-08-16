"""Question-type labels for retrieval slicing, read from a draft bundle's item sidecars.

Retrieval reports slice by question type (`factoid`, `numeric`, `comparative`, `multi-hop`, ...),
but a gold set JSONL carries no type: the label lives beside it, in the draft bundle's
`needle_items.jsonl` (ontology-assisted drafting) or `item_provenance.jsonl` (the external-draft
import lane). This module is the one place that knows where those sidecars sit relative to a gold
set -- beside it, or one level up when the gold set is an accepted ledger under `accepted/` -- and
it JOINS them, so a bundle carrying either one (or both) slices the same way. The first sidecar
that labels an item wins; the rest only fill gaps.
"""

import json
from collections import defaultdict
from pathlib import Path

SIDECAR_NAME = "needle_items.jsonl"
PROVENANCE_SIDECAR_NAME = "item_provenance.jsonl"
SIDECAR_NAMES = (SIDECAR_NAME, PROVENANCE_SIDECAR_NAME)
ACCEPTED_DIRNAME = "accepted"


def sidecar_paths(goldset: Path) -> list[Path]:
    """Every question-type sidecar for this gold set, nearest directory first."""
    roots = [goldset.parent]
    if goldset.parent.name == ACCEPTED_DIRNAME:
        roots.append(goldset.parent.parent)
    return [root / name for root in roots for name in SIDECAR_NAMES if (root / name).is_file()]


def _labels_from(source: Path) -> dict[str, str]:
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


def load_question_types(goldset: Path) -> dict[str, str]:
    """Map item id -> question type, joined over the gold set's sidecars ({} when it has none)."""
    joined: dict[str, str] = {}
    for source in sidecar_paths(goldset):
        for item_id, question_type in _labels_from(source).items():
            joined.setdefault(item_id, question_type)
    return joined


def aligned_question_types(goldset: Path, item_ids: list[str]) -> list[str | None] | None:
    """Question types aligned one-to-one with `item_ids`, or None when nothing labels them.

    None means "do not report slices at all" -- an empty or absent sidecar labels no item, and a
    report of empty slices would read as a measured result rather than a missing input.
    """
    labels = load_question_types(goldset)
    if not labels:
        return None
    return [labels.get(item_id) for item_id in item_ids]


def load_question_types_by_question(goldset: Path) -> dict[str, str]:
    """Map exact question text to a sidecar type, omitting ambiguous duplicate questions.

    Retrieval receives question text rather than the gold item id.  A duplicate question carrying
    conflicting labels therefore has no safe sidecar route and intentionally falls back to the
    deterministic text heuristic.
    """
    from llb.goldset.schema import load_goldset

    by_id = load_question_types(goldset)
    grouped: dict[str, set[str]] = defaultdict(set)
    for item in load_goldset(goldset):
        question_type = by_id.get(item.id)
        if question_type is not None:
            grouped[item.question].add(question_type)
    return {
        question: next(iter(question_types))
        for question, question_types in grouped.items()
        if len(question_types) == 1
    }
