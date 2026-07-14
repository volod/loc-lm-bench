"""Object builder and persistence for inter-reviewer agreement reports."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.goldset.verify_base import ACCEPT, REJECT
from llb.goldset.verify_multi.agreement_metrics import cohen_kappa, fleiss_kappa
from llb.goldset.verify_multi.common import (
    AGREEMENT_FILENAME,
    decision,
    edit,
    is_disagreement,
    joint_item_ids,
    rows_by_item,
)

ReviewerRows = dict[str, list[dict[str, str]]]
IndexedRows = dict[str, dict[str, dict[str, str]]]


@dataclass(slots=True)
class AgreementReportBuilder:
    """Index reviewer rows once, then build each report section independently."""

    by_reviewer: ReviewerRows
    reviewers: list[str] = field(init=False)
    indexed: IndexedRows = field(init=False)
    joint_items: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.reviewers = sorted(self.by_reviewer)
        self.indexed = {
            reviewer: rows_by_item(self.by_reviewer[reviewer]) for reviewer in self.reviewers
        }
        self.joint_items = joint_item_ids(self.by_reviewer)

    def build(self) -> dict[str, object]:
        jointly_decided, disagreements = self._classify_joint_items()
        return {
            "annotators": self.reviewers,
            "joint_items": len(self.joint_items),
            "jointly_decided": len(jointly_decided),
            "observed_agreement": self._observed_agreement(jointly_decided, disagreements),
            "kappa": self._joint_kappa(jointly_decided),
            "kappa_method": "cohen" if len(self.reviewers) == 2 else "fleiss",
            "disagreements": disagreements,
            "per_reviewer": {
                reviewer: self._reviewer_summary(reviewer) for reviewer in self.reviewers
            },
        }

    def _classify_joint_items(self) -> tuple[list[str], list[str]]:
        jointly_decided: list[str] = []
        disagreements: list[str] = []
        for item_id in self.joint_items:
            decisions = self._item_decisions(item_id)
            if not all(value in (ACCEPT, REJECT) for value in decisions):
                continue
            jointly_decided.append(item_id)
            edits = [edit(self.indexed[reviewer][item_id]) for reviewer in self.reviewers]
            if is_disagreement(decisions, edits):
                disagreements.append(item_id)
        return jointly_decided, disagreements

    def _item_decisions(self, item_id: str) -> list[str]:
        return [decision(self.indexed[reviewer][item_id]) for reviewer in self.reviewers]

    @staticmethod
    def _observed_agreement(jointly_decided: list[str], disagreements: list[str]) -> float:
        if not jointly_decided:
            return 0.0
        return (len(jointly_decided) - len(disagreements)) / len(jointly_decided)

    def _joint_kappa(self, jointly_decided: Sequence[str]) -> float | None:
        if len(self.reviewers) < 2 or len(jointly_decided) < 2:
            return None
        if len(self.reviewers) == 2:
            left, right = self.reviewers
            return cohen_kappa(
                [decision(self.indexed[left][item_id]) for item_id in jointly_decided],
                [decision(self.indexed[right][item_id]) for item_id in jointly_decided],
            )
        counts = [self._decision_counts(item_id) for item_id in jointly_decided]
        return fleiss_kappa(counts)

    def _decision_counts(self, item_id: str) -> list[int]:
        decisions = self._item_decisions(item_id)
        return [decisions.count(label) for label in (ACCEPT, REJECT)]

    def _reviewer_summary(self, reviewer: str) -> dict[str, int]:
        decisions = [decision(row) for row in self.by_reviewer[reviewer]]
        return {
            "decided": sum(value in (ACCEPT, REJECT) for value in decisions),
            "accepted": decisions.count(ACCEPT),
            "rejected": decisions.count(REJECT),
        }


def write_agreement_report(base_ws: Path, report: dict[str, object]) -> Path:
    path = Path(base_ws).with_name(AGREEMENT_FILENAME)
    atomic_write_text(path, json.dumps(report, ensure_ascii=False, indent=2))
    return path
