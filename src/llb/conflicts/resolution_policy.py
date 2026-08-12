"""Deterministic policy for turning conflict findings into reviewable actions."""

from dataclasses import dataclass
from typing import Any

from llb.conflicts.constants import (
    REL_COMPLEMENTARY,
    REL_CONTRADICTS,
    REL_DUPLICATE,
    REL_SUBSUMED_BY,
    REL_SUBSUMES,
    REL_SUPERSEDED_BY,
)
from llb.conflicts.group_artifact import group_decisions, group_summaries
from llb.conflicts.hashing import finding_id
from llb.core.contracts.common import JsonObject

POLICY_CONSERVATIVE = "conservative"
POLICY_PREFER_NEWER = "prefer-newer"
POLICIES = (POLICY_CONSERVATIVE, POLICY_PREFER_NEWER)

ACTION_KEEP_BOTH = "keep_both"
ACTION_DROP_DUPLICATE = "drop_duplicate"
ACTION_PREFER_NEWER = "prefer_newer"
ACTION_ESCALATE = "escalate"

STATUS_ACCEPTED = "accepted"
STATUS_REVIEW_REQUIRED = "review_required"


@dataclass(frozen=True)
class _Resolution:
    action: str = ACTION_KEEP_BOTH
    target_side: str | None = None
    rationale: str = "relation carries distinct or complementary knowledge"
    status: str = STATUS_ACCEPTED


def _side(finding: JsonObject, name: str) -> JsonObject:
    value = finding.get(name)
    if not isinstance(value, dict) or not isinstance(value.get("doc_id"), str):
        raise ValueError(f"finding side {name!r} is missing a string doc_id")
    return value


def _older_side(finding: JsonObject) -> str | None:
    staleness = finding.get("staleness")
    newer = staleness.get("newer_side") if isinstance(staleness, dict) else None
    if newer == "a":
        return "b"
    if newer == "b":
        return "a"
    return None


def _duplicate_target(finding: JsonObject) -> str:
    older = _older_side(finding)
    if older is not None:
        return older
    a = _side(finding, "a")
    b = _side(finding, "b")
    return "a" if str(a["doc_id"]) > str(b["doc_id"]) else "b"


def _duplicate_resolution(finding: JsonObject) -> _Resolution:
    if finding.get("tier") == "semantic":
        return _Resolution(
            action=ACTION_ESCALATE,
            status=STATUS_REVIEW_REQUIRED,
            rationale="semantic candidate is not adjudicated deletion authority",
        )
    return _Resolution(
        action=ACTION_DROP_DUPLICATE,
        target_side=_duplicate_target(finding),
        rationale="remove one redundant copy; preserve the newer or stable canonical side",
    )


def _supersession_resolution(finding: JsonObject, policy: str) -> _Resolution:
    target_side = _older_side(finding)
    if policy == POLICY_PREFER_NEWER and target_side is not None:
        return _Resolution(
            action=ACTION_PREFER_NEWER,
            target_side=target_side,
            rationale="governance orders the editions; suppress the older claim",
        )
    return _Resolution(
        action=ACTION_ESCALATE,
        target_side=target_side,
        status=STATUS_REVIEW_REQUIRED,
        rationale="supersession requires explicit review under this policy",
    )


def _resolution_for_relation(finding: JsonObject, policy: str, relation: str) -> _Resolution:
    if relation == REL_DUPLICATE:
        return _duplicate_resolution(finding)
    if relation == REL_SUPERSEDED_BY:
        return _supersession_resolution(finding, policy)
    if relation == REL_CONTRADICTS:
        return _Resolution(
            action=ACTION_ESCALATE,
            status=STATUS_REVIEW_REQUIRED,
            rationale="undated contradiction cannot be resolved from governance",
        )
    if relation in (REL_SUBSUMES, REL_SUBSUMED_BY):
        return _Resolution(
            rationale="subsumption is not deletion authority; retain and annotate both sides"
        )
    if relation == REL_COMPLEMENTARY:
        return _Resolution()
    return _Resolution(
        action=ACTION_ESCALATE,
        status=STATUS_REVIEW_REQUIRED,
        rationale=f"unknown relation {relation!r} requires review",
    )


def resolve_finding(finding: JsonObject, policy: str) -> JsonObject:
    """Choose one safe action; uncertain contradictions remain human work."""
    if policy not in POLICIES:
        raise ValueError(f"unknown resolution policy {policy!r}; choose one of {POLICIES}")
    relation = str(finding.get("relation", ""))
    resolution = _resolution_for_relation(finding, policy, relation)
    target = _side(finding, resolution.target_side) if resolution.target_side is not None else None
    return {
        "finding_id": finding_id(finding),
        "relation": relation,
        "tier": str(finding.get("tier", "")),
        "action": resolution.action,
        "status": resolution.status,
        "target_side": resolution.target_side,
        "target_doc_id": target.get("doc_id") if target is not None else None,
        "rationale": resolution.rationale,
        "a": dict(_side(finding, "a")),
        "b": dict(_side(finding, "b")),
        "staleness": dict(finding.get("staleness") or {}),
    }


def build_plan(findings: list[JsonObject], policy: str, corpus_root: str) -> JsonObject:
    """One item per finding row, plus one decision per group of rows that share a chunk.

    The items are what the overlay is built from and what a rollback restores, so every row keeps
    its own record; `decisions` is what an operator reviews, because six rows quoting one stale
    chunk are one call to make.
    """
    items = [resolve_finding(finding, policy) for finding in findings]
    summaries = group_summaries(findings)
    group_by_finding = {
        fid: summary["group_id"] for summary in summaries for fid in summary["finding_ids"]
    }
    for item in items:
        item["group_id"] = group_by_finding.get(str(item["finding_id"]))
    counts: dict[str, int] = {}
    for item in items:
        action = str(item["action"])
        counts[action] = counts.get(action, 0) + 1
    return {
        "schema_version": 2,
        "policy": policy,
        "corpus_root": corpus_root,
        "items": items,
        "decisions": group_decisions(summaries, items),
        "action_counts": dict(sorted(counts.items())),
    }


def as_json_object(value: Any, *, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value
