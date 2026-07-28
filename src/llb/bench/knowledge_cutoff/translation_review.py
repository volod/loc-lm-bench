"""Freeze a completely reviewed translation worksheet into aligned language lanes."""

import json
from pathlib import Path
from typing import Any

from llb.bench.knowledge_cutoff.data import CutoffEvent
from llb.bench.knowledge_cutoff.translation_artifacts import (
    DRAFTS_FILENAME,
    MANIFEST_FILENAME,
    SOURCE_FILENAME,
    WORKSHEET_FILENAME,
    load_translation_drafts,
    write_models_jsonl,
)
from llb.bench.knowledge_cutoff.translation_models import (
    TranslationDraft,
    source_hash,
    translation_hash,
    validate_translation,
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


def _accepted_translation(
    event: CutoffEvent,
    row: dict[str, str],
    draft: TranslationDraft,
) -> tuple[CutoffEvent, CutoffEvent] | None:
    if row["decision"] not in (ACCEPT, REJECT):
        raise ValueError(f"{event.id}: translation review is undecided")
    if row["decision"] == REJECT:
        return None
    if any(row[column] != PASS for column in CHECK_COLS):
        raise ValueError(f"{event.id}: accepted translation needs all four checks to pass")
    expected_source_hash = source_hash(event)
    if (
        row["source_hash"] != expected_source_hash
        or draft.source_hash != expected_source_hash
        or row.get("translation_hash") != translation_hash(draft)
    ):
        raise ValueError(f"{event.id}: source identity mismatch")
    translated = event.model_copy(
        update={"mcq_question": draft.question_uk, "mcq_choices": draft.choices_uk}
    )
    return event, translated


def _review_summary(
    bundle_dir: Path, reviewer: str, source_rows: int, accepted: int
) -> dict[str, Any]:
    manifest = json.loads((bundle_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "reviewer": reviewer.strip(),
        "resolved_revision": manifest["resolved_revision"],
        "source_rows": source_rows,
        "accepted_rows": accepted,
        "excluded_rows": source_rows - accepted,
        "complete": True,
    }


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
    for event in events:
        accepted = _accepted_translation(event, by_id[event.id], drafts[event.id])
        if accepted is None:
            continue
        source_event, translated_event = accepted
        accepted_en.append(source_event)
        accepted_uk.append(translated_event)
    if not accepted_en:
        raise ValueError("review excluded every translation")
    write_models_jsonl(bundle_dir / REVIEWED_EN_FILENAME, accepted_en)
    write_models_jsonl(bundle_dir / REVIEWED_UK_FILENAME, accepted_uk)
    write_worksheet_rows(bundle_dir / "translation_review.accepted.csv", rows, fields)
    summary = _review_summary(bundle_dir, reviewer, len(events), len(accepted_en))
    atomic_write_text(
        bundle_dir / REVIEW_SUMMARY_FILENAME,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary
