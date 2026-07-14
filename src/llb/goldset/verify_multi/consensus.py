"""Consensus resolution consumed by verification acceptance."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from llb.goldset.verify_base import ACCEPT, HUMAN_COLS, REJECT, load_worksheet
from llb.goldset.verify_multi.common import (
    ADJUDICATION_FILENAME,
    decision,
    edit,
    is_disagreement,
    joint_item_ids,
    load_reviewer_worksheets,
    reviewer_worksheets_from_manifest,
    rows_by_item,
)


ReviewerRows = dict[str, list[dict[str, str]]]


@dataclass(slots=True)
class ConsensusBuilder:
    """Merge reviewer rows, preferring decisions from an adjudication worksheet."""

    by_reviewer: ReviewerRows
    adjudication: Sequence[dict[str, str]] = ()
    reviewers: list[str] = field(init=False)
    indexed: dict[str, dict[str, dict[str, str]]] = field(init=False)
    adjudicated: dict[str, dict[str, str]] = field(init=False)

    def __post_init__(self) -> None:
        self.reviewers = sorted(self.by_reviewer)
        self.indexed = {
            reviewer: rows_by_item(self.by_reviewer[reviewer]) for reviewer in self.reviewers
        }
        self.adjudicated = rows_by_item(self.adjudication)

    def build(self) -> list[dict[str, str]]:
        if not self.reviewers:
            return []
        return [self._merge_item(item_id) for item_id in joint_item_ids(self.by_reviewer)]

    def _merge_item(self, item_id: str) -> dict[str, str]:
        adjudicated = self.adjudicated.get(item_id)
        if adjudicated is not None and decision(adjudicated) in (ACCEPT, REJECT):
            return dict(adjudicated)
        row = dict(self.indexed[self.reviewers[0]][item_id])
        if not self._is_unanimous(item_id):
            self._clear_human_decision(row)
        return row

    def _is_unanimous(self, item_id: str) -> bool:
        rows = [self.indexed[reviewer][item_id] for reviewer in self.reviewers]
        decisions = [decision(row) for row in rows]
        edits = [edit(row) for row in rows]
        return all(value in (ACCEPT, REJECT) for value in decisions) and not is_disagreement(
            decisions, edits
        )

    @staticmethod
    def _clear_human_decision(row: dict[str, str]) -> None:
        for column in HUMAN_COLS:
            row[column] = ""


def resolve_multi_reviewer_rows(worksheet: Path) -> list[dict[str, str]] | None:
    worksheets = reviewer_worksheets_from_manifest(Path(worksheet))
    if len(worksheets) < 2:
        return None
    by_reviewer = load_reviewer_worksheets(worksheets)
    adjudication_path = Path(worksheet).with_name(ADJUDICATION_FILENAME)
    adjudication: list[dict[str, str]] = []
    if adjudication_path.is_file():
        adjudication, _ = load_worksheet(adjudication_path)
    return ConsensusBuilder(by_reviewer, adjudication).build()
