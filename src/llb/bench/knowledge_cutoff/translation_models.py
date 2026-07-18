"""Source-bound translation models, parsing, and validation."""

import hashlib
import json
import re
from collections import Counter

from pydantic import BaseModel, ConfigDict, field_validator

from llb.bench.knowledge_cutoff.data import CutoffEvent

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


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
