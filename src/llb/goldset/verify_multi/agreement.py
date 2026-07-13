"""Inter-reviewer agreement arithmetic and report persistence."""

import json
from collections.abc import Sequence
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.goldset.verify_base import ACCEPT, REJECT
from llb.goldset.verify_multi.common import (
    AGREEMENT_FILENAME,
    decision,
    edit,
    is_disagreement,
    joint_item_ids,
    rows_by_item,
)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b):
        raise ValueError(f"label sequences differ in length: {len(a)} != {len(b)}")
    count = len(a)
    if count == 0:
        return 0.0
    observed = sum(1 for left, right in zip(a, b) if left == right) / count
    labels = set(a) | set(b)
    expected = sum(
        (list(a).count(label) / count) * (list(b).count(label) / count) for label in labels
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(counts: Sequence[Sequence[int]]) -> float:
    rows = [list(row) for row in counts if sum(row) > 0]
    if not rows:
        return 0.0
    raters = sum(rows[0])
    if raters < 2:
        raise ValueError("Fleiss' kappa needs at least 2 raters per item")
    if any(sum(row) != raters for row in rows):
        raise ValueError("every item must be rated by the same number of raters")
    item_agreement = [
        (sum(count * count for count in row) - raters) / (raters * (raters - 1)) for row in rows
    ]
    observed = sum(item_agreement) / len(rows)
    totals = [sum(row[index] for row in rows) for index in range(len(rows[0]))]
    category_rates = [total / (len(rows) * raters) for total in totals]
    expected = sum(rate * rate for rate in category_rates)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def agreement_report(by_reviewer: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    reviewers = sorted(by_reviewer)
    indexed = {reviewer: rows_by_item(by_reviewer[reviewer]) for reviewer in reviewers}
    joint = joint_item_ids(by_reviewer)
    jointly_decided: list[str] = []
    disagreements: list[str] = []
    for item_id in joint:
        decisions = [decision(indexed[reviewer][item_id]) for reviewer in reviewers]
        edits = [edit(indexed[reviewer][item_id]) for reviewer in reviewers]
        if all(value in (ACCEPT, REJECT) for value in decisions):
            jointly_decided.append(item_id)
            if is_disagreement(decisions, edits):
                disagreements.append(item_id)
    observed = (
        (len(jointly_decided) - len(disagreements)) / len(jointly_decided)
        if jointly_decided
        else 0.0
    )
    return {
        "annotators": reviewers,
        "joint_items": len(joint),
        "jointly_decided": len(jointly_decided),
        "observed_agreement": observed,
        "kappa": _joint_kappa(indexed, reviewers, jointly_decided),
        "kappa_method": "cohen" if len(reviewers) == 2 else "fleiss",
        "disagreements": disagreements,
        "per_reviewer": {
            reviewer: {
                "decided": sum(
                    1 for row in by_reviewer[reviewer] if decision(row) in (ACCEPT, REJECT)
                ),
                "accepted": sum(1 for row in by_reviewer[reviewer] if decision(row) == ACCEPT),
                "rejected": sum(1 for row in by_reviewer[reviewer] if decision(row) == REJECT),
            }
            for reviewer in reviewers
        },
    }


def _joint_kappa(
    indexed: dict[str, dict[str, dict[str, str]]],
    reviewers: Sequence[str],
    jointly_decided: Sequence[str],
) -> float | None:
    if len(reviewers) < 2 or len(jointly_decided) < 2:
        return None
    if len(reviewers) == 2:
        left, right = reviewers
        return cohen_kappa(
            [decision(indexed[left][item_id]) for item_id in jointly_decided],
            [decision(indexed[right][item_id]) for item_id in jointly_decided],
        )
    counts = [
        [
            sum(1 for reviewer in reviewers if decision(indexed[reviewer][item_id]) == label)
            for label in (ACCEPT, REJECT)
        ]
        for item_id in jointly_decided
    ]
    return fleiss_kappa(counts)


def write_agreement_report(base_ws: Path, report: dict[str, object]) -> Path:
    path = Path(base_ws).with_name(AGREEMENT_FILENAME)
    atomic_write_text(path, json.dumps(report, ensure_ascii=False, indent=2))
    return path
