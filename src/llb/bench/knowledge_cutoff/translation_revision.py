"""Apply explicit, mechanically revalidated corrections to translation drafts."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from llb.bench.knowledge_cutoff.data import CutoffEvent
from llb.bench.knowledge_cutoff.translation import (
    DRAFTS_FILENAME,
    SOURCE_FILENAME,
    TranslationDraft,
    load_translation_drafts,
    source_hash,
    validate_translation,
    write_models_jsonl,
)


class TranslationRevision(BaseModel):
    """Human- or agent-authored replacement text; source identity is filled mechanically."""

    model_config = ConfigDict(str_strip_whitespace=True)

    item_id: str
    question_uk: str
    choices_uk: list[str]


def _events(path: Path) -> list[CutoffEvent]:
    return [
        CutoffEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def apply_translation_revisions(bundle_dir: Path, revisions_path: Path) -> int:
    """Replace named drafts, preserving source order and rerunning every automatic gate."""
    events = _events(bundle_dir / SOURCE_FILENAME)
    by_id = {event.id: event for event in events}
    drafts = load_translation_drafts(bundle_dir / DRAFTS_FILENAME)
    seen: set[str] = set()
    for line_number, line in enumerate(
        revisions_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            revision = TranslationRevision.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"{revisions_path}:{line_number}: {exc}") from exc
        if revision.item_id in seen:
            raise ValueError(f"{revisions_path}: duplicate revision id {revision.item_id!r}")
        seen.add(revision.item_id)
        event = by_id.get(revision.item_id)
        if event is None:
            raise ValueError(f"{revision.item_id}: revision id is outside the source dataset")
        draft = TranslationDraft(
            item_id=event.id,
            source_hash=source_hash(event),
            question_uk=revision.question_uk,
            choices_uk=revision.choices_uk,
        )
        validate_translation(draft, event)
        drafts[event.id] = draft
    if not seen:
        raise ValueError(f"{revisions_path}: no revisions found")
    write_models_jsonl(
        bundle_dir / DRAFTS_FILENAME, [drafts[event.id] for event in events if event.id in drafts]
    )
    return len(seen)
