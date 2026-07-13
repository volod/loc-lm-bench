import json
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.bench.knowledge_cutoff.data import (
    CutoffEvent,
    EventSource,
    LoadedEvents,
    load_events,
    select_events,
)
from llb.bench.knowledge_cutoff.fit import fit_decay
from llb.bench.knowledge_cutoff.probe import parse_answer, prepare_probe
from llb.bench.knowledge_cutoff.run import run_knowledge_cutoff
from llb.bench.knowledge_cutoff.score import summarize
from llb.cli import app


def event(
    event_id: str = "e1",
    month: str = "2024-01",
    *,
    category: str = "death",
    predictability: str = "low",
) -> CutoffEvent:
    return CutoffEvent(
        id=event_id,
        date=f"{month}-15",
        month=month,
        category=category,
        region="test",
        predictability=predictability,
        subject="Example subject",
        fact="A fabricated test fact.",
        mcq_question=f"Which statement is correct for {event_id}?",
        mcq_choices=["A) wrong one", "B) correct marker", "C) wrong two", "D) wrong three"],
        mcq_answer="B",
        source="https://example.test/source",
    )


def row(month: str, label: str, *, category: str = "death", eligible: bool = True) -> dict:
    return {
        "item_id": f"{month}-{label}",
        "month": month,
        "category": category,
        "predictability": "low",
        "region": "test",
        "counts_for_curve": eligible,
        "label": label,
        "selected": "A" if label != "abstain" else None,
        "expected": "A",
        "choice_order": ["A", "B", "C", "D"],
        "objective_score": float(label == "correct"),
        "response": "A" if label != "abstain" else "",
    }


def test_event_schema_rejects_inconsistent_month_and_duplicate_choices():
    with pytest.raises(ValueError, match="month must match date"):
        event().model_copy(update={"date": "2024-02-01"}).model_validate(
            {**event().model_dump(), "date": "2024-02-01"}
        )
    with pytest.raises(ValueError, match="must be unique"):
        CutoffEvent.model_validate(
            {**event().model_dump(), "mcq_choices": ["same", "same", "third", "fourth"]}
        )


def test_event_schema_normalizes_hugging_face_datetime():
    parsed = CutoffEvent.model_validate({**event().model_dump(), "date": datetime(2024, 1, 15)})
    assert parsed.date == "2024-01-15"


def test_hugging_face_loader_resolves_and_records_exact_revision():
    calls = {}

    def resolve(dataset_id: str, revision: str) -> str:
        calls["resolve"] = (dataset_id, revision)
        return "a" * 40

    def loader(*args, **kwargs):
        calls["load"] = (args, kwargs)
        return [event().model_dump()]

    loaded = load_events(
        dataset_id="owner/events",
        revision="release",
        cache_dir="cache/datasets",
        dataset_loader=loader,
        revision_resolver=resolve,
    )
    assert calls["resolve"] == ("owner/events", "release")
    assert calls["load"][1]["revision"] == "a" * 40
    assert calls["load"][1]["cache_dir"] == "cache/datasets"
    assert loaded.source.requested_revision == "release"
    assert loaded.source.resolved_revision == "a" * 40


def test_exact_hugging_face_commit_skips_branch_resolution():
    commit = "b" * 40

    def should_not_resolve(_dataset_id: str, _revision: str) -> str:
        raise AssertionError("exact commits must work without a revision lookup")

    loaded = load_events(
        dataset_id="owner/events",
        revision=commit,
        dataset_loader=lambda *args, **kwargs: [event().model_dump()],
        revision_resolver=should_not_resolve,
    )
    assert loaded.source.resolved_revision == commit


def test_local_loader_hashes_input_and_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "events.jsonl"
    record = event().model_dump_json()
    path.write_text(record + "\n", encoding="utf-8")
    loaded = load_events(path=path)
    assert loaded.source.kind == "local"
    assert loaded.source.resolved_revision.startswith("sha256:")
    path.write_text(record + "\n" + record + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate event id"):
        load_events(path=path)


def test_probe_rebalances_positions_without_date_disclosure():
    probe = prepare_probe(event())
    assert "2024" not in probe.prompt
    assert "current date" not in probe.prompt.lower()
    assert "A) A)" not in probe.prompt
    assert probe.expected in {"A", "B", "C", "D"}
    correct_line = next(line for line in probe.prompt.splitlines() if "correct marker" in line)
    assert probe.expected == correct_line[0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("B", "B"), ("(c)", "C"), ("Answer is D.", "D"), ("I think B is best", None), ("", None)],
)
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


def test_summary_excludes_predictable_events_and_scores_controls():
    rows = [
        row("2024-01", "correct"),
        row("2024-01", "incorrect"),
        row("2024-02", "incorrect"),
        row("2024-02", "abstain"),
        row("2024-03", "correct", eligible=False),
        row("2024-03", "correct", category="control_alive", eligible=False),
        row("2024-03", "incorrect", category="fake_event", eligible=False),
    ]
    result = summarize(rows)
    assert [point.month for point in result.curve] == ["2024-01", "2024-02"]
    assert result.last_above == "2024-01"
    assert result.first_sustained_below is None
    assert result.controls["living_person_accuracy"] == 1.0
    assert result.controls["fake_event_confabulation_rate"] == 1.0


def test_optuna_fit_is_seeded_and_finds_decay_midpoint():
    rows = []
    correct_by_month = [8, 7, 6, 3, 2, 2]
    for month_index, correct in enumerate(correct_by_month, start=1):
        month = f"2024-{month_index:02d}"
        rows.extend(row(month, "correct") for _ in range(correct))
        rows.extend(row(month, "incorrect") for _ in range(8 - correct))
    summary = summarize(rows)
    first = fit_decay(summary, trials=40, seed=7)
    second = fit_decay(summary, trials=40, seed=7)
    assert first.status == "ok"
    assert first.effective_cutoff in {"2024-03", "2024-04", "2024-05"}
    assert first.cutoff_ordinal == pytest.approx(second.cutoff_ordinal)
    assert first.negative_log_likelihood == pytest.approx(second.negative_log_likelihood)


def test_smoke_limit_spans_the_full_horizon():
    events = [event(f"e{i}", f"2024-{i:02d}") for i in range(1, 13)]
    selected = select_events(events, 3)
    assert [item.month for item in selected] == ["2024-01", "2024-05", "2024-09"]


def test_end_to_end_fake_run_persists_reports(tmp_path):
    events = [
        event(f"e-{month}-{index}", month)
        for month in ("2024-01", "2024-02", "2024-03", "2024-04")
        for index in range(3)
    ]
    loaded = LoadedEvents(
        events,
        EventSource("local", "fixture", None, "sha256:test", None, None, "test-only"),
    )

    def complete(prompt: str) -> str:
        line = next(value for value in prompt.splitlines() if "correct marker" in value)
        return line[0]

    mirrored = {}

    def mirror(_manifest, out_dir):
        mirrored["report_exists"] = (out_dir / "report.json").is_file()

    result = run_knowledge_cutoff(
        loaded,
        model="local-test",
        backend="ollama",
        complete=complete,
        data_dir=tmp_path,
        optuna_trials=5,
        mirror=mirror,
    )
    assert result.summary.eligible_accuracy == 1.0
    assert result.paths is not None
    report = json.loads(Path(result.report_json).read_text(encoding="utf-8"))
    markdown = Path(result.report_markdown).read_text(encoding="utf-8")
    manifest = json.loads(Path(result.paths["manifest"]).read_text(encoding="utf-8"))
    assert report["model"] == "local-test"
    assert "## Monthly Evidence" in markdown
    assert "Apoorv Saxena" in markdown
    assert manifest["config"]["dataset_revision"] == "sha256:test"
    assert mirrored["report_exists"] is True


def test_cli_command_is_registered():
    result = CliRunner().invoke(app, ["bench-knowledge-cutoff", "--help"])
    assert result.exit_code == 0
    assert "effective knowledge cutoff" in result.stdout
