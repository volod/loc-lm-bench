"""Consensus resolution consumed by verification acceptance."""

from collections.abc import Sequence
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


def consensus_rows(
    by_reviewer: dict[str, list[dict[str, str]]],
    adjudication: Sequence[dict[str, str]] = (),
) -> list[dict[str, str]]:
    reviewers = sorted(by_reviewer)
    if not reviewers:
        return []
    indexed = {reviewer: rows_by_item(by_reviewer[reviewer]) for reviewer in reviewers}
    adjudicated = rows_by_item(adjudication)
    merged: list[dict[str, str]] = []
    for item_id in joint_item_ids(by_reviewer):
        adjudicated_row = adjudicated.get(item_id)
        if adjudicated_row is not None and decision(adjudicated_row) in (ACCEPT, REJECT):
            merged.append(dict(adjudicated_row))
            continue
        decisions = [decision(indexed[reviewer][item_id]) for reviewer in reviewers]
        edits = [edit(indexed[reviewer][item_id]) for reviewer in reviewers]
        row = dict(indexed[reviewers[0]][item_id])
        unanimous = all(value in (ACCEPT, REJECT) for value in decisions) and not is_disagreement(
            decisions, edits
        )
        if not unanimous:
            for column in HUMAN_COLS:
                row[column] = ""
        merged.append(row)
    return merged


def resolve_multi_reviewer_rows(worksheet: Path) -> list[dict[str, str]] | None:
    worksheets = reviewer_worksheets_from_manifest(Path(worksheet))
    if len(worksheets) < 2:
        return None
    by_reviewer = load_reviewer_worksheets(worksheets)
    adjudication_path = Path(worksheet).with_name(ADJUDICATION_FILENAME)
    adjudication: list[dict[str, str]] = []
    if adjudication_path.is_file():
        adjudication, _ = load_worksheet(adjudication_path)
    return consensus_rows(by_reviewer, adjudication)
