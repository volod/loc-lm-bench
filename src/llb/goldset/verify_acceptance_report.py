"""Focused verify acceptance report implementation."""

import re
from collections.abc import Sequence
from typing import cast
from llb.goldset.verify_base import (
    ACCEPT,
    ACCEPT_POLICIES,
    CHECK_COLS,
    CHECK_REJECT_CODES,
    DEFAULT_TOLERANCE,
    FAIL,
    POLICY_GLOBAL,
    POLICY_PER_STRATUM,
    POLICY_WEIGHTED,
    REJECT,
)
from llb.goldset.verify_sampling.confidence import row_confidence


def infer_reject_code(row: dict[str, str]) -> str:
    """The reject code implied by the first failed check, else `other`."""
    for col in CHECK_COLS:
        if (row.get(col) or "").strip() == FAIL:
            return CHECK_REJECT_CODES[col]
    return "other"


def rejection_reasons_summary(rows: Sequence[dict[str, str]]) -> dict[str, object]:
    """Aggregate rejected rows by code for draft feedback (`rejection_reasons.json`).

    A blank code is inferred from the first failed check. This makes the concise `x` review action
    produce actionable feedback without requiring the reviewer to repeat the failed criterion.
    """
    by_code: dict[str, dict[str, object]] = {}
    rejected = 0
    for row in rows:
        if (row.get("decision") or "").strip() != REJECT:
            continue
        rejected += 1
        code = (row.get("reject_code") or "").strip() or infer_reject_code(row)
        cell = by_code.setdefault(code, {"count": 0, "items": []})
        cell["count"] = cast(int, cell["count"]) + 1
        entry: dict[str, str] = {"item_id": (row.get("item_id") or "").strip()}
        note = (row.get("human_note") or "").strip()
        if note:
            entry["note"] = note
        cast(list[dict[str, str]], cell["items"]).append(entry)
    return {"rejected": rejected, "by_code": dict(sorted(by_code.items()))}


def ground_answer(doc_text: str, answer: str, *, hint_start: int = 0) -> tuple[int, int] | None:
    """Locate `answer` verbatim in `doc_text`, preferring the occurrence nearest `hint_start`.

    Returns `(char_start, char_end)` or None when the text does not contain the answer -- the
    caller must then BLOCK the edit until the reviewer re-words it to a verbatim span.
    """
    answer = answer.strip()
    if not answer:
        return None
    starts = [m.start() for m in re.finditer(re.escape(answer), doc_text)]
    if not starts:
        return None
    best = min(starts, key=lambda s: abs(s - hint_start))
    return best, best + len(answer)


def worksheet_edits(rows: Sequence[dict[str, str]]) -> dict[str, str]:
    """Item id -> edited reference answer, for rows carrying a non-empty `edited_answer`."""
    return {
        (row.get("item_id") or "").strip(): (row.get("edited_answer") or "").strip()
        for row in rows
        if (row.get("edited_answer") or "").strip() and (row.get("item_id") or "").strip()
    }


def _is_decided(row: dict[str, str]) -> bool:
    return (row.get("decision") or "").strip() in (ACCEPT, REJECT)


def _failed_any_check(row: dict[str, str]) -> bool:
    return any((row.get(col) or "").strip() == FAIL for col in CHECK_COLS)


def confidence_weighted_reject_rate(rows: Sequence[dict[str, str]]) -> float:
    """Reject rate where each decided row weighs `1 + max(row_confidence, 0)`.

    A reject on a row the automated signals (cross-check verdict + retrieval rank) rated
    CONFIDENT is worse than a reject those signals already flagged: it means the pipeline's own
    quality signals mispredict, so it counts more against the bundle. Deterministic from the
    worksheet columns alone.
    """
    weighted_total = weighted_rejected = 0.0
    for row in rows:
        if not _is_decided(row):
            continue
        weight = 1.0 + max(row_confidence(row), 0.0)
        weighted_total += weight
        if (row.get("decision") or "").strip() == REJECT:
            weighted_rejected += weight
    return (weighted_rejected / weighted_total) if weighted_total else 0.0


def _stratum_tolerance(key: str, tolerance: float, overrides: dict[str, float] | None) -> float:
    if overrides and key in overrides:
        return overrides[key]
    return tolerance


def acceptance_report(
    rows: Sequence[dict[str, str]],
    tolerance: float = DEFAULT_TOLERANCE,
    *,
    policy: str = POLICY_GLOBAL,
    stratum_tolerances: dict[str, float] | None = None,
) -> dict[str, object]:
    """Acceptance-sampling summary: per-stratum + overall decided/reject counts and pass/fail.

    A decided item is a `reject` defect; the reject RATE over decided items is compared to
    `tolerance`. Items with a failed check but no explicit decision are surfaced as
    `undecided_with_failures` so nothing silently slips through. Pure -- the caller decides
    what to emit.

    `policy` selects the acceptance arithmetic (`ACCEPT_POLICIES`): `global` compares the
    overall rate (per-stratum results stay advisory, the original rule); `per-stratum`
    requires EVERY stratum within its own tolerance (`stratum_tolerances` overrides the
    global default per stratum key); `weighted` compares the confidence-weighted rate.
    """
    if policy not in ACCEPT_POLICIES:
        raise ValueError(f"unknown acceptance policy {policy!r}; use one of {ACCEPT_POLICIES}")
    per_stratum, decided, rejected, undecided_with_failures = _tally_decisions(rows)
    _score_strata(per_stratum, tolerance, stratum_tolerances)
    overall_rate = (rejected / decided) if decided else 0.0
    weighted_rate = confidence_weighted_reject_rate(rows)
    if policy == POLICY_PER_STRATUM:
        passed = decided > 0 and all(bool(c["passed"]) for c in per_stratum.values())
    elif policy == POLICY_WEIGHTED:
        passed = decided > 0 and weighted_rate <= tolerance
    else:
        passed = decided > 0 and overall_rate <= tolerance
    return {
        "tolerance": tolerance,
        "policy": policy,
        "n": len(rows),
        "decided": decided,
        "rejected": rejected,
        "accepted": decided - rejected,
        "undecided": len(rows) - decided,
        "undecided_with_failures": undecided_with_failures,
        "reject_rate": overall_rate,
        "weighted_reject_rate": weighted_rate,
        "passed": passed,
        "per_stratum": per_stratum,
    }


def _tally_decisions(
    rows: Sequence[dict[str, str]],
) -> tuple[dict[str, dict[str, float]], int, int, int]:
    """Count decided/rejected per stratum plus overall + undecided-with-failed-check rows."""
    per_stratum: dict[str, dict[str, float]] = {}
    decided = rejected = 0
    undecided_with_failures = 0
    for row in rows:
        key = row.get("stratum", "") or "(none)"
        cell = per_stratum.setdefault(key, {"decided": 0, "rejected": 0})
        if _is_decided(row):
            decided += 1
            cell["decided"] += 1
            if (row.get("decision") or "").strip() == REJECT:
                rejected += 1
                cell["rejected"] += 1
        elif _failed_any_check(row):
            undecided_with_failures += 1
    return per_stratum, decided, rejected, undecided_with_failures


def _score_strata(
    per_stratum: dict[str, dict[str, float]],
    tolerance: float,
    stratum_tolerances: dict[str, float] | None,
) -> None:
    """Fill each stratum cell with its reject rate, effective tolerance, and pass flag."""
    for key, cell in per_stratum.items():
        d = cell["decided"]
        cell["reject_rate"] = (cell["rejected"] / d) if d else 0.0
        cell["tolerance"] = _stratum_tolerance(key, tolerance, stratum_tolerances)
        cell["passed"] = float(cell["reject_rate"] <= cell["tolerance"])
