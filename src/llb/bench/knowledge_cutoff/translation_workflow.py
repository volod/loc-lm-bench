"""Resumable Ukrainian translation drafting workflow."""

import logging
from collections.abc import Callable
from pathlib import Path

from llb.bench.knowledge_cutoff.data import CutoffEvent, LoadedEvents
from llb.bench.knowledge_cutoff.translation_artifacts import (
    DRAFTS_FILENAME,
    SOURCE_FILENAME,
    WORKSHEET_FILENAME,
    load_translation_drafts,
    worksheet_rows,
    write_manifest,
    write_models_jsonl,
)
from llb.bench.knowledge_cutoff.translation_models import (
    TranslationDraft,
    parse_translation,
    source_hash,
    translation_prompt,
    validate_translation,
)
from llb.goldset.verify_base import HUMAN_COLS, WORKSHEET_COLS, load_worksheet, write_worksheet_rows

TRANSLATION_MAX_TOKENS = 512
DRAFT_ATTEMPTS = 2
_LOG = logging.getLogger(__name__)


def _draft_event(event: CutoffEvent, complete: Callable[[str], str]) -> TranslationDraft:
    prompt = translation_prompt(event)
    error: ValueError | None = None
    for attempt in range(1, DRAFT_ATTEMPTS + 1):
        _LOG.info("[knowledge-cutoff-ua-draft] %s attempt %d/%d", event.id, attempt, DRAFT_ATTEMPTS)
        try:
            return parse_translation(complete(prompt), event)
        except ValueError as exc:
            error = exc
            prompt += (
                f"\n\nYour prior response failed validation: {exc}. Return corrected JSON only."
            )
    assert error is not None
    raise error


def translation_progress(loaded: LoadedEvents, out_dir: Path) -> tuple[int, int]:
    """Return valid existing draft rows and total rows, rejecting source drift."""
    drafts = load_translation_drafts(out_dir / DRAFTS_FILENAME)
    event_ids = {event.id for event in loaded.events}
    if unknown := set(drafts) - event_ids:
        raise ValueError(f"draft bundle contains ids outside this revision: {sorted(unknown)[:3]}")
    for event in loaded.events:
        draft = drafts.get(event.id)
        if draft is not None:
            validate_translation(draft, event)
    return len(drafts), len(loaded.events)


def draft_translation_bundle(
    loaded: LoadedEvents,
    *,
    complete: Callable[[str], str],
    out_dir: Path,
    translator: str,
) -> Path:
    """Draft missing translations and refresh the worksheet without losing decisions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    draft_path = out_dir / DRAFTS_FILENAME
    drafts = load_translation_drafts(draft_path)
    translation_progress(loaded, out_dir)
    write_models_jsonl(out_dir / SOURCE_FILENAME, loaded.events)
    write_manifest(loaded, out_dir, translator)
    _draft_missing_events(loaded.events, drafts, draft_path, complete)
    worksheet = out_dir / WORKSHEET_FILENAME
    fresh = worksheet_rows(loaded.events, drafts)
    _merge_human_decisions(worksheet, fresh)
    extras = ["review_profile", "source_answer", "source_hash", "translation_hash"]
    write_worksheet_rows(worksheet, fresh, [*WORKSHEET_COLS, *extras])
    return worksheet


def _draft_missing_events(
    events: list[CutoffEvent],
    drafts: dict[str, TranslationDraft],
    draft_path: Path,
    complete: Callable[[str], str],
) -> None:
    for event in events:
        prior = drafts.get(event.id)
        if prior is not None:
            if prior.source_hash != source_hash(event):
                raise ValueError(f"{event.id}: source changed under an existing translation")
            continue
        drafts[event.id] = _draft_event(event, complete)
        write_models_jsonl(draft_path, [drafts[item.id] for item in events if item.id in drafts])


def _merge_human_decisions(worksheet: Path, fresh: list[dict[str, str]]) -> None:
    if not worksheet.is_file():
        return
    old_rows, _ = load_worksheet(worksheet)
    old_human = {row["item_id"]: row for row in old_rows}
    for row in fresh:
        prior = old_human.get(row["item_id"])
        if prior and _same_translation(prior, row):
            row.update({name: prior.get(name, "") for name in HUMAN_COLS})


def _same_translation(prior: dict[str, str], fresh: dict[str, str]) -> bool:
    return (
        prior.get("source_hash") == fresh["source_hash"]
        and prior.get("translation_hash") == fresh["translation_hash"]
    )
