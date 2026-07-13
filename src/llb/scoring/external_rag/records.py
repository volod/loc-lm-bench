"""Answered-JSONL I/O and human-review state."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llb.core.fsutil import atomic_write_text
from llb.scoring.external_rag_common import (
    HUMAN_DECISION_FIELD,
    HUMAN_DECISIONS,
    HUMAN_FIELDS,
    HUMAN_SCORE_FIELD,
    _string,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows with file:line context on parse failures."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(item)
    return rows


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Atomically write JSONL records, preserving Unicode text for human review."""
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    atomic_write_text(path, text)


def ensure_human_fields(records: Sequence[dict[str, Any]]) -> bool:
    """Add missing JSONL-backed human-review fields and report whether rows changed."""
    changed = False
    for record in records:
        for field in HUMAN_FIELDS:
            if field not in record:
                record[field] = ""
                changed = True
    return changed


def clear_human_fields(records: Sequence[dict[str, Any]]) -> None:
    """Clear JSONL-backed human review state in place."""
    for record in records:
        for field in HUMAN_FIELDS:
            record[field] = ""


def is_human_scored(record: dict[str, Any]) -> bool:
    """Return whether a record carries a valid decision and score."""
    decision = _string(record.get(HUMAN_DECISION_FIELD)).strip().lower()
    score_text = _string(record.get(HUMAN_SCORE_FIELD)).strip()
    if decision not in HUMAN_DECISIONS or not score_text:
        return False
    try:
        score = float(score_text)
    except ValueError:
        return False
    return 0.0 <= score <= 1.0


def human_reviewed_count(records: Sequence[dict[str, Any]]) -> int:
    """Count records with complete human-review state."""
    return sum(1 for record in records if is_human_scored(record))
