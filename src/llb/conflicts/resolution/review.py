"""The reviewer's side of a resolution plan: the ledger it is given, and the decisions it returns.

A reviewer is the scarce resource in this lane, so the ledger is written to be read by decision:
records stay one per open ROW -- a drop applies to one span, so the row is the unit a human can
actually decide -- but each names the decision group it belongs to and how many rows share it, and
the records are ordered so a group reads as one block instead of six unrelated conflicts. The
blocks themselves are ordered by TO REVIEW, the count this reviewer is actually funding, which is
the one count the audit report could not rank on because it has no policy to read it from.

Coming back, a decision may be addressed to a row or to a whole group. A group-wide decision may
only KEEP: members of a group share a unit, not a document pair, so "drop a" means a different act
on each of them, and applying one group-wide would suppress spans nobody looked at.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llb.conflicts.resolution.policy import (
    ACTION_DROP_DUPLICATE,
    ACTION_KEEP_BOTH,
    STATUS_ACCEPTED,
)
from llb.core.contracts.common import JsonObject

REVIEW_TYPE = "corpus_conflict_resolution"
STATUS_REVIEW_REQUIRED = "review_required"
_DROP_DECISIONS = ("drop_a", "drop_b")
_ACCEPTED_RATIONALE = "accepted human review decision"


@dataclass(frozen=True)
class GroupCounts:
    """A decision group's size and its two counts of work, as the ledger needs them."""

    rows: int
    decide_rows: int
    review_rows: int


# A row whose group the plan does not describe is its own decision, and it is open by definition:
# only rows already at `review_required` reach the ledger.
_LONE_ROW = GroupCounts(rows=1, decide_rows=1, review_rows=1)


def read_jsonl(path: Path | str) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def merge_review_decisions(plan: JsonObject, path: Path | str) -> None:
    """Apply a reviewed ledger: per-row decisions, plus whole-group `keep_both` decisions."""
    rows = read_jsonl(path)
    per_row = {
        str(row["finding_id"]): str(row.get("resolution_decision", ""))
        for row in rows
        if row.get("finding_id")
    }
    group_wide = group_decisions(rows)
    items = plan.get("items")
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        decision = per_row.get(str(item.get("finding_id"))) or group_wide.get(
            str(item.get("group_id"))
        )
        _apply_decision(item, decision)


def _apply_decision(item: JsonObject, decision: str | None) -> None:
    if decision == ACTION_KEEP_BOTH:
        item.update(action=ACTION_KEEP_BOTH, status=STATUS_ACCEPTED, target_side=None)
        item["target_doc_id"] = None
        item["rationale"] = _ACCEPTED_RATIONALE
    elif decision in _DROP_DECISIONS:
        side = str(decision)[-1]
        ref = item.get(side)
        item.update(action=ACTION_DROP_DUPLICATE, status=STATUS_ACCEPTED, target_side=side)
        item["target_doc_id"] = ref.get("doc_id") if isinstance(ref, dict) else None
        item["rationale"] = _ACCEPTED_RATIONALE


def group_decisions(rows: list[JsonObject]) -> dict[str, str]:
    """Whole-group decisions from the ledger; a destructive one is refused, not applied."""
    decisions: dict[str, str] = {}
    for row in rows:
        group_id = row.get("group_id")
        decision = str(row.get("resolution_decision", ""))
        if row.get("finding_id") or not isinstance(group_id, str) or not decision:
            continue
        if decision != ACTION_KEEP_BOTH:
            raise ValueError(
                f"group {group_id}: a whole-group review decision may only be "
                f"{ACTION_KEEP_BOTH!r}, not {decision!r}; decide a drop on the row it applies to"
            )
        decisions[group_id] = decision
    return decisions


def review_jsonl(plan: JsonObject) -> str:
    """One record per row still needing review, biggest review first, a group at a time.

    The ledger is the one artifact ranked on TO REVIEW rather than on the audit's TO DECIDE: a
    reviewer's cost is the open rows, not the rows whose relation the vocabulary calls work, and
    here -- unlike in the audit report -- the policy has already run, so the funded count exists.
    Every record still carries both counts of its group, so a reviewer reading one row can see how
    the audit sized the same decision.
    """
    items = plan.get("items")
    open_items = [
        item
        for item in (items if isinstance(items, list) else [])
        if isinstance(item, dict) and item.get("status") == STATUS_REVIEW_REQUIRED
    ]
    counts = group_counts(plan)
    rows = [_record(item, counts) for item in sorted(open_items, key=_stake_order(counts))]
    return "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)


def group_counts(plan: JsonObject) -> dict[str, GroupCounts]:
    """Rows, to-decide, and to-review per decision group, keyed by group id."""
    return {
        str(decision["group_id"]): GroupCounts(
            rows=int(decision.get("rows", 1)),
            decide_rows=int(decision.get("decide_rows", 0)),
            review_rows=int(decision.get("review_rows", 0)),
        )
        for decision in plan.get("decisions") or []
        if isinstance(decision, dict)
    }


def _record(item: JsonObject, counts: dict[str, GroupCounts]) -> JsonObject:
    group = counts.get(str(item.get("group_id")), _LONE_ROW)
    return {
        "review_type": REVIEW_TYPE,
        "finding_id": item.get("finding_id"),
        "group_id": item.get("group_id"),
        "group_rows": group.rows,
        # Both counts, never one: `group_review_rows` is what this record costs a reviewer,
        # `group_decide_rows` is what the audit report ranked the same group on.
        "group_decide_rows": group.decide_rows,
        "group_review_rows": group.review_rows,
        "relation": item.get("relation"),
        "rationale": item.get("rationale"),
        "a": item.get("a"),
        "b": item.get("b"),
        "staleness": item.get("staleness"),
        "resolution_decision": "",
    }


def _stake_order(counts: dict[str, GroupCounts]) -> Callable[[JsonObject], tuple[int, int, str]]:
    """Order by review stake first, then by group id -- a group's rows always stay adjacent."""

    def key(item: JsonObject) -> tuple[int, int, str]:
        group_id = str(item.get("group_id") or "")
        index = int(group_id[1:]) if group_id[1:].isdigit() else 0
        return (-counts.get(group_id, _LONE_ROW).review_rows, index, str(item.get("finding_id")))

    return key
