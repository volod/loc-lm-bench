import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.bench.knowledge_cutoff.data import CutoffEvent, EventSource, LoadedEvents
from llb.bench.knowledge_cutoff.paired import load_reviewed_lanes, run_bilingual_cutoff
from llb.bench.knowledge_cutoff.paired_report import paired_statistics
from llb.bench.knowledge_cutoff.translation import (
    WORKSHEET_FILENAME,
    draft_translation_bundle,
    parse_translation,
)
from llb.bench.knowledge_cutoff.translation_review import (
    REVIEW_SUMMARY_FILENAME,
    confirm_accepted_translation_checks,
    freeze_reviewed_bundle,
    review_bundle_status,
)
from llb.bench.knowledge_cutoff.translation_revision import apply_translation_revisions
from llb.cli import app
from llb.goldset.verify_base import CHECK_COLS, PASS, STATUS_DECIDED, load_worksheet
from llb.goldset.verify_base import write_worksheet_rows
from llb.goldset.verify_card import format_card
from llb.goldset.verify_session.loop import run_session


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


def test_confirm_prior_aggregate_translation_acceptance(tmp_path):
    draft_translation_bundle(
        loaded_events(1), complete=translation, out_dir=tmp_path, translator="local"
    )
    worksheet = tmp_path / WORKSHEET_FILENAME
    rows, fields = load_worksheet(worksheet)
    rows[0]["decision"] = "accept"
    rows[0]["human_status"] = STATUS_DECIDED
    write_worksheet_rows(worksheet, rows, fields)
    assert confirm_accepted_translation_checks(tmp_path) == 1
    assert review_bundle_status(tmp_path)["ready_to_freeze"] is True


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


def test_frozen_bundle_runs_aligned_paired_report(tmp_path):
    bundle = tmp_path / "translation"
    draft_translation_bundle(
        loaded_events(), complete=translation, out_dir=bundle, translator="local"
    )
    accept_all(bundle)
    assert review_bundle_status(bundle)["ready_to_freeze"] is True
    summary = freeze_reviewed_bundle(bundle, reviewer="reviewer-1")
    assert summary["accepted_rows"] == 4
    english, ukrainian, review = load_reviewed_lanes(bundle)
    assert [item.id for item in english.events] == [item.id for item in ukrainian.events]
    assert "Яке" in ukrainian.events[0].mcq_question
    assert review["resolved_revision"] == "a" * 40

    def complete(prompt: str) -> str:
        marker = "correct marker" if "correct marker" in prompt else "правильна позначка"
        return next(line[0] for line in prompt.splitlines() if marker in line)

    result = run_bilingual_cutoff(
        bundle,
        model="local-test",
        backend="ollama",
        complete=complete,
        data_dir=tmp_path / "runs",
        optuna_trials=5,
    )
    assert result.paired["accuracy_delta"] == 0.0
    assert result.paths is not None
    report_dir = Path(result.paths["manifest"]).parent
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert report["paired"]["bootstrap"]["samples"] == 2000
    assert "Monthly Language Deltas" in (report_dir / "report.md").read_text(encoding="utf-8")
    assert (bundle / REVIEW_SUMMARY_FILENAME).is_file()


def test_paired_statistics_detects_choice_mapping_drift():
    english = [
        {
            "item_id": "e1",
            "month": "2025-01",
            "counts_for_curve": True,
            "choice_order": ["A", "B", "C", "D"],
            "expected": "B",
            "objective_score": 1.0,
        }
    ]
    ukrainian = [{**english[0], "choice_order": ["B", "A", "C", "D"]}]
    with pytest.raises(ValueError, match="different source-choice mappings"):
        paired_statistics(english, ukrainian, seed=42)


@pytest.mark.parametrize(
    "command",
    [
        "knowledge-cutoff-ua-draft",
        "knowledge-cutoff-ua-review",
        "knowledge-cutoff-ua-revise",
        "knowledge-cutoff-ua-confirm-accepted",
        "knowledge-cutoff-ua-validate",
        "knowledge-cutoff-ua-freeze",
        "bench-knowledge-cutoff-bilingual",
    ],
)
def test_bilingual_cli_commands_are_registered(command):
    result = CliRunner().invoke(app, [command, "--help"])
    assert result.exit_code == 0
