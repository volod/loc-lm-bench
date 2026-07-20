"""Deterministic policy for turning conflict findings into reviewable actions."""

import hashlib
import json
from typing import Any

from llb.conflicts.constants import (
    REL_COMPLEMENTARY,
    REL_CONTRADICTS,
    REL_DUPLICATE,
    REL_SUBSUMED_BY,
    REL_SUBSUMES,
    REL_SUPERSEDED_BY,
)
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


def finding_id(finding: JsonObject) -> str:
    """Stable identity for one finding, independent of JSON object key order."""
    encoded = json.dumps(finding, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


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


def resolve_finding(finding: JsonObject, policy: str) -> JsonObject:
    """Choose one safe action; uncertain contradictions remain human work."""
    if policy not in POLICIES:
        raise ValueError(f"unknown resolution policy {policy!r}; choose one of {POLICIES}")
    relation = str(finding.get("relation", ""))
    action = ACTION_KEEP_BOTH
    target_side: str | None = None
    rationale = "relation carries distinct or complementary knowledge"
    status = STATUS_ACCEPTED

    if relation == REL_DUPLICATE:
        if finding.get("tier") == "semantic":
            action = ACTION_ESCALATE
            status = STATUS_REVIEW_REQUIRED
            rationale = "semantic candidate is not adjudicated deletion authority"
        else:
            action = ACTION_DROP_DUPLICATE
            target_side = _duplicate_target(finding)
            rationale = "remove one redundant copy; preserve the newer or stable canonical side"
    elif relation == REL_SUPERSEDED_BY:
        target_side = _older_side(finding)
        if policy == POLICY_PREFER_NEWER and target_side is not None:
            action = ACTION_PREFER_NEWER
            rationale = "governance orders the editions; suppress the older claim"
        else:
            action = ACTION_ESCALATE
            status = STATUS_REVIEW_REQUIRED
            rationale = "supersession requires explicit review under this policy"
    elif relation == REL_CONTRADICTS:
        action = ACTION_ESCALATE
        status = STATUS_REVIEW_REQUIRED
        rationale = "undated contradiction cannot be resolved from governance"
    elif relation in (REL_SUBSUMES, REL_SUBSUMED_BY):
        rationale = "subsumption is not deletion authority; retain and annotate both sides"
    elif relation != REL_COMPLEMENTARY:
        action = ACTION_ESCALATE
        status = STATUS_REVIEW_REQUIRED
        rationale = f"unknown relation {relation!r} requires review"

    target = _side(finding, target_side) if target_side is not None else None
    return {
        "finding_id": finding_id(finding),
        "relation": relation,
        "tier": str(finding.get("tier", "")),
        "action": action,
        "status": status,
        "target_side": target_side,
        "target_doc_id": target.get("doc_id") if target is not None else None,
        "rationale": rationale,
        "a": dict(_side(finding, "a")),
        "b": dict(_side(finding, "b")),
        "staleness": dict(finding.get("staleness") or {}),
    }


def build_plan(findings: list[JsonObject], policy: str, corpus_root: str) -> JsonObject:
    items = [resolve_finding(finding, policy) for finding in findings]
    counts: dict[str, int] = {}
    for item in items:
        action = str(item["action"])
        counts[action] = counts.get(action, 0) + 1
    return {
        "schema_version": 1,
        "policy": policy,
        "corpus_root": corpus_root,
        "items": items,
        "action_counts": dict(sorted(counts.items())),
    }


def as_json_object(value: Any, *, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value
