"""Resumable Ukrainian translation drafts and the publication review gate."""

import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from llb.bench.knowledge_cutoff.data import DATASET_LICENSE, CutoffEvent, LoadedEvents
from llb.core.fsutil import atomic_write_text
from llb.goldset.verify_base import (
    HUMAN_COLS,
    STATUS_PENDING,
    WORKSHEET_COLS,
    load_worksheet,
    write_worksheet_rows,
)

DRAFTS_FILENAME = "translations.draft.jsonl"
SOURCE_FILENAME = "events.source.jsonl"
WORKSHEET_FILENAME = "translation_review.csv"
MANIFEST_FILENAME = "translation_manifest.json"
TRANSLATION_PROFILE = "knowledge-cutoff-translation"
TRANSLATION_MAX_TOKENS = 512
DRAFT_ATTEMPTS = 2
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_LOG = logging.getLogger(__name__)


class TranslationDraft(BaseModel):
    """One source-identity-bound Ukrainian translation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    item_id: str
    source_hash: str
    question_uk: str
    choices_uk: list[str]

    @field_validator("question_uk")
    @classmethod
    def _question_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("question_uk must not be empty")
        return value

    @field_validator("choices_uk")
    @classmethod
    def _four_unique_choices(cls, value: list[str]) -> list[str]:
        if len(value) != 4 or any(not choice.strip() for choice in value):
            raise ValueError("choices_uk must contain four non-empty choices")
        if len(set(value)) != 4:
            raise ValueError("choices_uk must be unique")
        return value


def source_hash(event: CutoffEvent) -> str:
    payload = [event.id, event.mcq_question, event.mcq_choices, event.mcq_answer]
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def translation_hash(draft: TranslationDraft) -> str:
    payload = [draft.question_uk, draft.choices_uk]
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def translation_prompt(event: CutoffEvent) -> str:
    payload = {"question": event.mcq_question, "choices": event.mcq_choices}
    return (
        "Translate this multiple-choice question and every choice from English into natural "
        "Ukrainian. Preserve meaning, named entities, uncertainty, and the exact four-choice "
        "order. Do not add dates, facts, explanations, or temporal clues. Return only JSON with "
        'keys "question_uk" and "choices_uk" (an array of four strings).\n\n'
        + json.dumps(payload, ensure_ascii=False)
    )


def parse_translation(text: str, event: CutoffEvent) -> TranslationDraft:
    start = text.find("{")
    if start < 0:
        raise ValueError(f"{event.id}: translator did not return a JSON object")
    try:
        raw, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{event.id}: invalid translator JSON: {exc}") from exc
    draft = TranslationDraft.model_validate(
        {"item_id": event.id, "source_hash": source_hash(event), **raw}
    )
    validate_translation(draft, event)
    return draft


def validate_translation(draft: TranslationDraft, event: CutoffEvent) -> None:
    """Recheck language, numeric clues, and exact source identity for one draft."""
    if draft.item_id != event.id or draft.source_hash != source_hash(event):
        raise ValueError(f"{event.id}: source changed under an existing translation")
    from llb.prep.ontology.language import is_ukrainian_dominant

    translated_text = draft.question_uk + " " + " ".join(draft.choices_uk)
    if not is_ukrainian_dominant(translated_text):
        raise ValueError(f"{event.id}: translated question and choices are not Ukrainian-dominant")
    source_numbers = Counter(
        _NUMBER.findall(event.mcq_question + " " + " ".join(event.mcq_choices))
    )
    translated_numbers = Counter(_NUMBER.findall(translated_text))
    if translated_numbers != source_numbers:
        raise ValueError(f"{event.id}: translation added, removed, or changed a numeric clue")


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


def write_models_jsonl(path: Path, models: Sequence[BaseModel]) -> None:
    content = "".join(model.model_dump_json() + "\n" for model in models)
    atomic_write_text(path, content)


def load_translation_drafts(path: Path) -> dict[str, TranslationDraft]:
    if not path.is_file():
        return {}
    drafts: dict[str, TranslationDraft] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            draft = TranslationDraft.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if draft.item_id in drafts:
            raise ValueError(f"{path}: duplicate translation id {draft.item_id!r}")
        drafts[draft.item_id] = draft
    return drafts


def _worksheet_rows(
    events: Sequence[CutoffEvent], drafts: dict[str, TranslationDraft]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for event in events:
        draft = drafts[event.id]
        context = json.dumps(
            {"choices_en": event.mcq_choices, "choices_uk": draft.choices_uk},
            ensure_ascii=False,
            indent=2,
        )
        row = {name: "" for name in WORKSHEET_COLS}
        row.update(
            {
                "item_kind": "goldset",
                "item_id": event.id,
                "provenance": "machine-translated",
                "stratum": event.month,
                "question": event.mcq_question,
                "reference_answer": draft.question_uk,
                "span_text": context,
                "context": context,
                "synthetic": "true",
                "human_status": STATUS_PENDING,
                "review_profile": TRANSLATION_PROFILE,
                "source_answer": event.mcq_answer,
                "source_hash": draft.source_hash,
                "translation_hash": translation_hash(draft),
            }
        )
        rows.append(row)
    return rows


def _write_manifest(loaded: LoadedEvents, out_dir: Path, translator: str) -> None:
    payload = {
        "schema_version": 1,
        "dataset": loaded.source.identity,
        "requested_revision": loaded.source.requested_revision,
        "resolved_revision": loaded.source.resolved_revision,
        "license": DATASET_LICENSE,
        "translator": translator,
        "n_source_events": len(loaded.events),
    }
    atomic_write_text(out_dir / MANIFEST_FILENAME, json.dumps(payload, indent=2) + "\n")


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
    """Draft missing translations and refresh the review worksheet without losing decisions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    draft_path = out_dir / DRAFTS_FILENAME
    drafts = load_translation_drafts(draft_path)
    translation_progress(loaded, out_dir)
    write_models_jsonl(out_dir / SOURCE_FILENAME, loaded.events)
    _write_manifest(loaded, out_dir, translator)
    for event in loaded.events:
        prior_draft = drafts.get(event.id)
        if prior_draft is not None:
            if prior_draft.source_hash != source_hash(event):
                raise ValueError(f"{event.id}: source changed under an existing translation")
            continue
        drafts[event.id] = _draft_event(event, complete)
        write_models_jsonl(
            draft_path, [drafts[item.id] for item in loaded.events if item.id in drafts]
        )
    worksheet = out_dir / WORKSHEET_FILENAME
    fresh = _worksheet_rows(loaded.events, drafts)
    if worksheet.is_file():
        old_rows, _ = load_worksheet(worksheet)
        old_human = {row["item_id"]: row for row in old_rows}
        for row in fresh:
            prior_row = old_human.get(row["item_id"])
            if (
                prior_row
                and prior_row.get("source_hash") == row["source_hash"]
                and prior_row.get("translation_hash") == row["translation_hash"]
            ):
                row.update({name: prior_row.get(name, "") for name in HUMAN_COLS})
    extras = ["review_profile", "source_answer", "source_hash", "translation_hash"]
    write_worksheet_rows(worksheet, fresh, [*WORKSHEET_COLS, *extras])
    return worksheet
