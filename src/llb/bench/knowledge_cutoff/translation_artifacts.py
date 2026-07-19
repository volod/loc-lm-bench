"""Translation draft, worksheet, and manifest persistence."""

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from llb.bench.knowledge_cutoff.data import DATASET_LICENSE, CutoffEvent, LoadedEvents
from llb.bench.knowledge_cutoff.translation_models import TranslationDraft, translation_hash
from llb.core.fsutil import atomic_write_text
from llb.goldset.verify_base import STATUS_PENDING, WORKSHEET_COLS

DRAFTS_FILENAME = "translations.draft.jsonl"
SOURCE_FILENAME = "events.source.jsonl"
WORKSHEET_FILENAME = "translation_review.csv"
MANIFEST_FILENAME = "translation_manifest.json"
TRANSLATION_PROFILE = "knowledge-cutoff-translation"


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


def worksheet_rows(
    events: Sequence[CutoffEvent], drafts: dict[str, TranslationDraft]
) -> list[dict[str, str]]:
    """Build fresh machine-owned worksheet columns for all translated events."""
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


def write_manifest(loaded: LoadedEvents, out_dir: Path, translator: str) -> None:
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
