"""Corpus-conflict resolution JSONL adapter."""

import json
from pathlib import Path
from typing import Any

from llb.core.fsutil import atomic_write_text
from llb.review.core import ReviewAction, ReviewAdapter, ReviewRecord
from llb.review.presentation import fields_section, json_section

DECISION_FIELD = "resolution_decision"

_ACTIONS = (
    ReviewAction("k", "Keep both", "keep_both", "positive"),
    ReviewAction("a", "Drop A", "drop_a", "negative"),
    ReviewAction("b", "Drop B", "drop_b", "negative"),
    ReviewAction("c", "Clear", "clear", "neutral"),
)


class ConflictResolutionAdapter(ReviewAdapter):
    """Review unresolved conflict pairs without changing the source corpus."""

    kind = "corpus-conflict-resolution"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.records = [
            row
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
            if isinstance(row, dict)
        ]

    @property
    def actions(self) -> tuple[ReviewAction, ...]:
        return _ACTIONS

    def __len__(self) -> int:
        return len(self.records)

    def record(self, index: int) -> ReviewRecord:
        row = self.records[index]
        key = _value(row.get("finding_id")) or str(index + 1)
        return ReviewRecord(
            key=key,
            title=f"corpus conflict: {key}",
            sections=(
                fields_section("Finding", row, ("relation", "rationale"), "data"),
                json_section("Side A", row.get("a"), "evidence"),
                json_section("Side B", row.get("b"), "evidence"),
                json_section("Governance", row.get("staleness"), "metadata"),
            ),
            stratum=_value(row.get("relation")) or "all",
            verdict=_value(row.get(DECISION_FIELD)),
        )

    def apply(self, index: int, action: str) -> None:
        if action == "clear":
            self.records[index][DECISION_FIELD] = ""
        elif action in ("keep_both", "drop_a", "drop_b"):
            self.records[index][DECISION_FIELD] = action
        else:
            raise ValueError(f"unsupported {self.kind} action: {action}")
        content = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in self.records)
        atomic_write_text(self.path, content)


def _value(value: Any) -> str:
    return "" if value is None else str(value).strip()
