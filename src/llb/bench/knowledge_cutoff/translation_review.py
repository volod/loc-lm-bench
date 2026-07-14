"""Freeze a completely reviewed translation worksheet into aligned language lanes."""

import json
from pathlib import Path
from typing import Any

from llb.bench.knowledge_cutoff.data import CutoffEvent
from llb.bench.knowledge_cutoff.translation import (
    DRAFTS_FILENAME,
    MANIFEST_FILENAME,
    SOURCE_FILENAME,
    TRANSLATION_PROFILE,
    WORKSHEET_FILENAME,
    load_translation_drafts,
    source_hash,
    translation_hash,
    validate_translation,
    write_models_jsonl,
)
from llb.core.fsutil import atomic_write_text
from llb.goldset.verify_base import (
    ACCEPT,
    CHECK_COLS,
    PASS,
    REJECT,
    load_worksheet,
    write_worksheet_rows,
)

REVIEWED_EN_FILENAME = "events.en.reviewed.jsonl"
REVIEWED_UK_FILENAME = "events.uk.reviewed.jsonl"
REVIEW_SUMMARY_FILENAME = "review_summary.json"


def _read_events(path: Path) -> list[CutoffEvent]:
    return [
        CutoffEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def review_bundle_status(bundle_dir: Path) -> dict[str, int | bool]:
    """Validate source/draft/worksheet alignment and report review-gate progress."""
    events = _read_events(bundle_dir / SOURCE_FILENAME)
    drafts = load_translation_drafts(bundle_dir / DRAFTS_FILENAME)
    rows, _fields = load_worksheet(bundle_dir / WORKSHEET_FILENAME)
    event_ids = {event.id for event in events}
    if (
        len(drafts) != len(events)
        or len(rows) != len(events)
        or set(drafts) != event_ids
        or {row["item_id"] for row in rows} != event_ids
    ):
        raise ValueError("source, draft, and worksheet ids must match exactly")
    by_id = {row["item_id"]: row for row in rows}
    for event in events:
        draft = drafts[event.id]
        validate_translation(draft, event)
        if by_id[event.id].get("source_hash") != source_hash(event):
            raise ValueError(f"{event.id}: worksheet source identity is stale; refresh the draft")
        if by_id[event.id].get("translation_hash") != translation_hash(draft):
            raise ValueError(f"{event.id}: worksheet translation is stale; refresh the draft")
    accepted = sum(row["decision"] == ACCEPT for row in rows)
    excluded = sum(row["decision"] == REJECT for row in rows)
    undecided = len(rows) - accepted - excluded
    incomplete_accepted = sum(
        row["decision"] == ACCEPT and any(row[column] != PASS for column in CHECK_COLS)
        for row in rows
    )
    return {
        "source_rows": len(events),
        "draft_rows": len(drafts),
        "accepted_rows": accepted,
        "excluded_rows": excluded,
        "undecided_rows": undecided,
        "incomplete_accepted_rows": incomplete_accepted,
        "ready_to_freeze": undecided == 0 and incomplete_accepted == 0 and accepted > 0,
    }


def confirm_accepted_translation_checks(bundle_dir: Path) -> int:
    """Record four implied passes for prior aggregate translation accept decisions."""
    worksheet = bundle_dir / WORKSHEET_FILENAME
    rows, fields = load_worksheet(worksheet)
    changed = 0
    for row in rows:
        if row.get("review_profile") != TRANSLATION_PROFILE or row.get("decision") != ACCEPT:
            continue
        invalid = [column for column in CHECK_COLS if row.get(column) not in ("", PASS)]
        if invalid:
            raise ValueError(
                f"{row['item_id']}: accepted row has explicit failed or invalid checks: {invalid}"
            )
        if any(not row.get(column) for column in CHECK_COLS):
            row.update({column: PASS for column in CHECK_COLS})
            changed += 1
    if changed:
        write_worksheet_rows(worksheet, rows, fields)
    return changed


def freeze_reviewed_bundle(bundle_dir: Path, *, reviewer: str) -> dict[str, Any]:
    """Mechanically gate and freeze accepted translations into aligned event files."""
    if not reviewer.strip():
        raise ValueError("reviewer sign-off must not be empty")
    events = _read_events(bundle_dir / SOURCE_FILENAME)
    drafts = load_translation_drafts(bundle_dir / DRAFTS_FILENAME)
    rows, fields = load_worksheet(bundle_dir / WORKSHEET_FILENAME)
    if len(rows) != len(events) or {row["item_id"] for row in rows} != {e.id for e in events}:
        raise ValueError("worksheet must contain every source event exactly once")
    by_id = {row["item_id"]: row for row in rows}
    accepted_en: list[CutoffEvent] = []
    accepted_uk: list[CutoffEvent] = []
    rejected = 0
    for event in events:
        row = by_id[event.id]
        if row["decision"] not in (ACCEPT, REJECT):
            raise ValueError(f"{event.id}: translation review is undecided")
        if row["decision"] == REJECT:
            rejected += 1
            continue
        if any(row[column] != PASS for column in CHECK_COLS):
            raise ValueError(f"{event.id}: accepted translation needs all four checks to pass")
        draft = drafts[event.id]
        if (
            row["source_hash"] != source_hash(event)
            or draft.source_hash != source_hash(event)
            or row.get("translation_hash") != translation_hash(draft)
        ):
            raise ValueError(f"{event.id}: source identity mismatch")
        accepted_en.append(event)
        accepted_uk.append(
            event.model_copy(
                update={"mcq_question": draft.question_uk, "mcq_choices": draft.choices_uk}
            )
        )
    if not accepted_en:
        raise ValueError("review excluded every translation")
    write_models_jsonl(bundle_dir / REVIEWED_EN_FILENAME, accepted_en)
    write_models_jsonl(bundle_dir / REVIEWED_UK_FILENAME, accepted_uk)
    write_worksheet_rows(bundle_dir / "translation_review.accepted.csv", rows, fields)
    summary = {
        "schema_version": 1,
        "reviewer": reviewer.strip(),
        "resolved_revision": json.loads(
            (bundle_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )["resolved_revision"],
        "source_rows": len(events),
        "accepted_rows": len(accepted_en),
        "excluded_rows": rejected,
        "complete": True,
    }
    atomic_write_text(
        bundle_dir / REVIEW_SUMMARY_FILENAME,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary
