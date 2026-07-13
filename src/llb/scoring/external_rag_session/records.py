"""Human-field state mutations and index math over the answered JSONL records."""

from collections.abc import Sequence
from typing import Any

from llb.scoring.external_rag import (
    HUMAN_DECISION_ACCEPT,
    HUMAN_DECISION_FIELD,
    HUMAN_DECISION_PARTIAL,
    HUMAN_DECISION_REJECT,
    HUMAN_FIELDS,
    HUMAN_SCORE_FIELD,
    HUMAN_STATUS_FIELD,
    HUMAN_STATUS_SCORED,
    human_reviewed_count,
    is_human_scored,
)

ACCEPT_SCORE = 1.0
PARTIAL_SCORE = 0.5
REJECT_SCORE = 0.0
ACCEPT_THRESHOLD = 0.85


def first_unscored_index(records: Sequence[dict[str, Any]]) -> int:
    """Index of the first record without complete human score + decision; 0 if all scored."""
    for index, record in enumerate(records):
        if not is_human_scored(record):
            return index
    return 0


def reviewed_count(records: Sequence[dict[str, Any]]) -> int:
    """How many records have complete human score + decision."""
    return human_reviewed_count(records)


def _set_decision(record: dict[str, Any], decision: str) -> None:
    score = {
        HUMAN_DECISION_ACCEPT: ACCEPT_SCORE,
        HUMAN_DECISION_PARTIAL: PARTIAL_SCORE,
        HUMAN_DECISION_REJECT: REJECT_SCORE,
    }[decision]
    _set_score_and_decision(record, score, decision)


def _set_explicit_score(record: dict[str, Any], score: float) -> None:
    if score >= ACCEPT_THRESHOLD:
        decision = HUMAN_DECISION_ACCEPT
    elif score > 0.0:
        decision = HUMAN_DECISION_PARTIAL
    else:
        decision = HUMAN_DECISION_REJECT
    _set_score_and_decision(record, score, decision)


def _set_score_and_decision(record: dict[str, Any], score: float, decision: str) -> None:
    record[HUMAN_SCORE_FIELD] = f"{score:g}"
    record[HUMAN_DECISION_FIELD] = decision
    record[HUMAN_STATUS_FIELD] = HUMAN_STATUS_SCORED


def _clear_row(record: dict[str, Any]) -> None:
    for field in HUMAN_FIELDS:
        record[field] = ""


def _advanced_index(index: int, records: Sequence[dict[str, Any]]) -> int:
    total = len(records)
    if index < total - 1:
        return index + 1
    if reviewed_count(records) == total:
        return total
    return first_unscored_index(records)


def _get_index(start: int | None, total: int, records: Sequence[dict[str, Any]]) -> int:
    if start is not None:
        return max(0, min(start - 1, total - 1))
    if reviewed_count(records) == total:
        return total
    return first_unscored_index(records)
