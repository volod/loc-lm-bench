"""Shared multi-reviewer paths, worksheet loading, and row accessors."""

import json
from collections.abc import Sequence
from pathlib import Path

from llb.goldset.verify_base import ACCEPT, REJECT, REVIEWER_COL, SAMPLE_MANIFEST, load_worksheet

AGREEMENT_FILENAME = "agreement.json"
ADJUDICATION_FILENAME = "adjudication.csv"
PRIOR_DECISIONS_COL = "prior_decisions"
ADJUDICATOR_ID = "adjudicator"


def reviewer_id(index: int) -> str:
    return f"r{index}"


def reviewer_worksheet_path(base: Path, index: int) -> Path:
    base = Path(base)
    return base.with_name(f"{base.stem}.{reviewer_id(index)}{base.suffix}")


def reviewer_worksheets_from_manifest(base_ws: Path) -> list[Path]:
    path = Path(base_ws).with_name(SAMPLE_MANIFEST)
    if not path.is_file():
        return []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    worksheets = manifest.get("worksheets") if isinstance(manifest, dict) else None
    if not isinstance(worksheets, list):
        return []
    return [Path(str(worksheet)) for worksheet in worksheets if worksheet]


def load_reviewer_worksheets(paths: Sequence[Path]) -> dict[str, list[dict[str, str]]]:
    by_reviewer: dict[str, list[dict[str, str]]] = {}
    for path in paths:
        rows, _ = load_worksheet(Path(path))
        reviewer = next(
            ((row.get(REVIEWER_COL) or "").strip() for row in rows if row.get(REVIEWER_COL)),
            Path(path).stem,
        )
        by_reviewer[reviewer] = rows
    return by_reviewer


def decision(row: dict[str, str]) -> str:
    return (row.get("decision") or "").strip()


def edit(row: dict[str, str]) -> str:
    return (row.get("edited_answer") or "").strip()


def rows_by_item(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        (row.get("item_id") or "").strip(): row
        for row in rows
        if (row.get("item_id") or "").strip()
    }


def joint_item_ids(by_reviewer: dict[str, list[dict[str, str]]]) -> list[str]:
    reviewers = list(by_reviewer)
    if not reviewers:
        return []
    first = [
        (row.get("item_id") or "").strip()
        for row in by_reviewer[reviewers[0]]
        if (row.get("item_id") or "").strip()
    ]
    others = [set(rows_by_item(by_reviewer[reviewer])) for reviewer in reviewers[1:]]
    return [item_id for item_id in first if all(item_id in item_ids for item_ids in others)]


def is_disagreement(decisions: Sequence[str], edits: Sequence[str]) -> bool:
    if any(value not in (ACCEPT, REJECT) for value in decisions):
        return False
    if len(set(decisions)) > 1:
        return True
    return decisions[0] == ACCEPT and len(set(edits)) > 1
