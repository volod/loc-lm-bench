"""Shared fixtures for bilingual knowledge-cutoff tests."""

import json
from pathlib import Path

from llb.bench.knowledge_cutoff.data import CutoffEvent, EventSource, LoadedEvents
from llb.bench.knowledge_cutoff.translation_artifacts import WORKSHEET_FILENAME
from llb.goldset.verify_base import CHECK_COLS, PASS, STATUS_DECIDED, load_worksheet
from llb.goldset.verify_base import write_worksheet_rows


def event(event_id: str, month: str = "2025-01") -> CutoffEvent:
    return CutoffEvent(
        id=event_id,
        date=f"{month}-15",
        month=month,
        category="death",
        region="test",
        predictability="low",
        subject="Example subject",
        fact="A test fact.",
        mcq_question="Which statement is correct for the example subject?",
        mcq_choices=["wrong one", "correct marker", "wrong two", "wrong three"],
        mcq_answer="B",
        source="https://example.test/source",
    )


def loaded_events(n: int = 4) -> LoadedEvents:
    events = [event(f"event-{index}", f"2025-{index:02d}") for index in range(1, n + 1)]
    source = EventSource(
        "huggingface", "owner/events", "release", "a" * 40, "events", "train", "CC BY 4.0"
    )
    return LoadedEvents(events, source)


def translation(_prompt: str) -> str:
    return json.dumps(
        {
            "question_uk": "Яке твердження є правильним?",
            "choices_uk": [
                "неправильний варіант один",
                "правильна позначка",
                "неправильний варіант два",
                "неправильний варіант три",
            ],
        },
        ensure_ascii=False,
    )


def accept_all(bundle: Path) -> None:
    worksheet = bundle / WORKSHEET_FILENAME
    rows, fields = load_worksheet(worksheet)
    for row in rows:
        row.update({column: PASS for column in CHECK_COLS})
        row["decision"] = "accept"
        row["human_status"] = STATUS_DECIDED
    write_worksheet_rows(worksheet, rows, fields)
