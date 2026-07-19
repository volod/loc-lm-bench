import json

import pytest
from bilingual_cutoff_helpers import accept_all, event, loaded_events, translation
from llb.bench.knowledge_cutoff.translation_artifacts import WORKSHEET_FILENAME
from llb.bench.knowledge_cutoff.translation_models import parse_translation
from llb.bench.knowledge_cutoff.translation_workflow import draft_translation_bundle
from llb.bench.knowledge_cutoff.translation_review import (
    freeze_reviewed_bundle,
    review_bundle_status,
)
from llb.bench.knowledge_cutoff.translation_revision import apply_translation_revisions
from llb.goldset.verify_base import CHECK_COLS, PASS, STATUS_DECIDED, load_worksheet
from llb.goldset.verify_base import write_worksheet_rows
from llb.goldset.verify_card import format_card
from llb.goldset.verify_session.loop import run_session


def test_translation_parser_rejects_choice_count():
    with pytest.raises(ValueError, match="four non-empty choices"):
        parse_translation('{"question_uk":"Питання?","choices_uk":["a"]}', event("e"))


def test_translation_parser_uses_first_object_and_preserves_numeric_clues():
    source = event("e").model_copy(
        update={"mcq_choices": ["wrong", "correct", "died in 2020", "alive"]}
    )
    valid = json.dumps(
        {
            "question_uk": "Яке твердження правильне?",
            "choices_uk": ["хибне", "правильне", "помер у 2020 році", "живий"],
        },
        ensure_ascii=False,
    )
    assert parse_translation(valid + "\n" + valid, source).choices_uk[2].endswith("році")
    with pytest.raises(ValueError, match="numeric clue"):
        parse_translation(valid.replace("2020", "2021"), source)


def test_draft_is_resumable_and_review_card_uses_translation_checks(tmp_path):
    calls = []

    def complete(prompt: str) -> str:
        calls.append(prompt)
        return translation(prompt)

    source = loaded_events(2)
    worksheet = draft_translation_bundle(
        source, complete=complete, out_dir=tmp_path, translator="local-translator"
    )
    assert len(calls) == 2
    draft_translation_bundle(
        source,
        complete=lambda _prompt: pytest.fail("completed translations must be resumed"),
        out_dir=tmp_path,
        translator="local-translator",
    )
    rows, _fields = load_worksheet(worksheet)
    card = format_card(rows[0], 1, 2, 0)
    assert "factual meaning matches the English source" in card
    assert "no fact, date, or temporal clue was added" in card
    assert "правильна позначка" in card


def test_translation_accept_records_all_implied_checks(tmp_path):
    worksheet = draft_translation_bundle(
        loaded_events(1), complete=translation, out_dir=tmp_path, translator="local"
    )
    run_session(worksheet, inputs=["y", "q"], output=lambda _line: None)
    row = load_worksheet(worksheet)[0][0]
    assert row["decision"] == "accept"
    assert all(row[column] == PASS for column in CHECK_COLS)


def test_translation_accept_refuses_explicit_failed_check(tmp_path):
    worksheet = draft_translation_bundle(
        loaded_events(1), complete=translation, out_dir=tmp_path, translator="local"
    )
    output: list[str] = []
    run_session(worksheet, inputs=["G", "y", "q"], output=output.append)
    row = load_worksheet(worksheet)[0][0]
    assert row["decision"] == ""
    assert any("conflicts with failed checks" in line for line in output)


def test_explicit_revision_is_source_bound_and_revalidated(tmp_path):
    draft_translation_bundle(
        loaded_events(1), complete=translation, out_dir=tmp_path, translator="local"
    )
    accept_all(tmp_path)
    revisions = tmp_path / "revisions.jsonl"
    revisions.write_text(
        json.dumps(
            {
                "item_id": "event-1",
                "question_uk": "Яке твердження про приклад є правильним?",
                "choices_uk": ["хибне один", "правильне", "хибне два", "хибне три"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    assert apply_translation_revisions(tmp_path, revisions) == 1
    draft_translation_bundle(
        loaded_events(1),
        complete=lambda _prompt: pytest.fail("revision refresh must not call the model"),
        out_dir=tmp_path,
        translator="local",
    )
    assert load_worksheet(tmp_path / WORKSHEET_FILENAME)[0][0]["decision"] == ""
    unknown = revisions.read_text(encoding="utf-8").replace("event-1", "unknown")
    revisions.write_text(unknown, encoding="utf-8")
    with pytest.raises(ValueError, match="outside the source dataset"):
        apply_translation_revisions(tmp_path, revisions)


def test_freeze_blocks_undecided_and_incomplete_accepts(tmp_path):
    draft_translation_bundle(
        loaded_events(1), complete=translation, out_dir=tmp_path, translator="local"
    )
    assert review_bundle_status(tmp_path)["ready_to_freeze"] is False
    with pytest.raises(ValueError, match="undecided"):
        freeze_reviewed_bundle(tmp_path, reviewer="reviewer-1")
    worksheet = tmp_path / WORKSHEET_FILENAME
    rows, fields = load_worksheet(worksheet)
    rows[0]["decision"] = "accept"
    rows[0]["human_status"] = STATUS_DECIDED
    write_worksheet_rows(worksheet, rows, fields)
    with pytest.raises(ValueError, match="all four checks"):
        freeze_reviewed_bundle(tmp_path, reviewer="reviewer-1")


def test_validation_rejects_stale_worksheet_source_identity(tmp_path):
    draft_translation_bundle(
        loaded_events(1), complete=translation, out_dir=tmp_path, translator="local"
    )
    worksheet = tmp_path / WORKSHEET_FILENAME
    rows, fields = load_worksheet(worksheet)
    rows[0]["source_hash"] = "stale"
    write_worksheet_rows(worksheet, rows, fields)
    with pytest.raises(ValueError, match="source identity is stale"):
        review_bundle_status(tmp_path)
