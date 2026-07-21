"""Tests for durable resume."""

import json
from operator import itemgetter
import pytest
from llb.backends.base import ChatResult
from llb.core.config import RunConfig
from llb.executor import durability_journal as durability
from llb.executor.runner import run_eval
from test_durable_eval import (
    DOC,
    FakeLauncher,
    FakeStore,
    _hit_store,
    _read_scores,
    _runner_fn,
    _uninterrupted,
    gold_item,
)


def test_kill_then_resume_matches_uninterrupted(tmp_path, monkeypatch):
    items = [gold_item(f"uk-{i}", f"q{i}", "Київ", "Київ") for i in range(4)]
    store = FakeStore(
        {
            f"q{i}": [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}]
            for i in range(4)
        }
    )

    def responder(_launcher, _messages):
        return ChatResult(text="Київ", completion_tokens=2, latency_s=0.4)

    clean_scores = _uninterrupted(tmp_path, items, store, responder)

    # Pin the run timestamp so the resume can name the same staging dir.
    monkeypatch.setattr(
        "llb.executor.runner_target._run_timestamp", lambda run_id: "20260101T000000.0Z-abc"
    )
    cfg = RunConfig(data_dir=tmp_path / "killed", run_name="dur", top_k=3, model="fake-uk")

    # First attempt: journal 2 cases, then a hard kill mid-run.
    launcher = FakeLauncher(responder)
    inner = _runner_fn(store, launcher, cfg)
    seen = {"n": 0}

    def killer(item):
        if seen["n"] >= 2:
            raise KeyboardInterrupt
        seen["n"] += 1
        return inner(item)

    with pytest.raises(KeyboardInterrupt):
        run_eval(
            cfg,
            items=items,
            store=store,
            launcher=launcher,
            runner_fn=killer,
            mirror=lambda *a: None,
            emit=False,
            sleep=lambda _s: None,
        )

    staging = cfg.run_staging_dir("20260101T000000.0Z-abc")
    assert staging.exists()  # staging preserved on interrupt
    journal_lines = durability.journal_path(staging).read_text(encoding="utf-8").splitlines()
    assert len(journal_lines) == 2  # exactly the two completed cases journaled
    assert not cfg.run_dir("20260101T000000.0Z-abc").exists()  # not finalized

    # Resume: reuse the 2 journaled cases, run the remaining 2, finalize.
    launcher2 = FakeLauncher(responder)
    result = run_eval(
        cfg,
        items=items,
        store=store,
        launcher=launcher2,
        runner_fn=_runner_fn(store, launcher2, cfg),
        mirror=lambda *a: None,
        emit=False,
        resume=cfg.run_dir("20260101T000000.0Z-abc"),
        sleep=lambda _s: None,
    )

    assert result["manifest"].durability["resumed_cases"] == 2
    assert not staging.exists()  # staging consumed by the atomic finalize
    resumed_scores = _read_scores(cfg.run_dir("20260101T000000.0Z-abc"))
    # Resumed run reproduces the uninterrupted per-case scores exactly (ordering-independent).
    by_id = itemgetter("item_id")
    assert sorted(resumed_scores, key=by_id) == sorted(clean_scores, key=by_id)
    # The finalized bundle never carries the journal.
    assert not (cfg.run_dir("20260101T000000.0Z-abc") / durability.JOURNAL_NAME).exists()


def test_resume_refuses_mismatched_goldset(tmp_path, monkeypatch):
    items = [gold_item("uk-1", "q0", "Київ", "Київ")]
    store = _hit_store("q0")

    def responder(_launcher, _messages):
        raise KeyboardInterrupt  # kill before the only case completes

    monkeypatch.setattr(
        "llb.executor.runner_target._run_timestamp", lambda run_id: "20260101T000000.0Z-xyz"
    )
    cfg = RunConfig(data_dir=tmp_path, run_name="dur", top_k=3, model="fake-uk")
    launcher = FakeLauncher(responder)
    with pytest.raises(KeyboardInterrupt):
        run_eval(
            cfg,
            items=items,
            store=store,
            launcher=launcher,
            runner_fn=lambda item: responder(launcher, None),
            mirror=lambda *a: None,
            emit=False,
            sleep=lambda _s: None,
        )

    # Resume with a DIFFERENT goldset (changed question) must abort cleanly and write no bundle.
    changed = [gold_item("uk-1", "q0-CHANGED", "Київ", "Київ")]
    launcher2 = FakeLauncher(lambda _launcher, _messages: ChatResult(text="Київ"))
    with pytest.raises(SystemExit, match="goldset"):
        run_eval(
            cfg,
            items=changed,
            store=store,
            launcher=launcher2,
            runner_fn=_runner_fn(store, launcher2, cfg),
            mirror=lambda *a: None,
            emit=False,
            resume=cfg.run_dir("20260101T000000.0Z-xyz"),
            sleep=lambda _s: None,
        )
    assert not cfg.run_dir("20260101T000000.0Z-xyz").exists()


def test_resume_without_interrupted_run_aborts(tmp_path):
    items = [gold_item("uk-1", "q0", "Київ", "Київ")]
    cfg = RunConfig(data_dir=tmp_path, run_name="dur", top_k=3, model="fake-uk")
    launcher = FakeLauncher(lambda _launcher, _messages: ChatResult(text="Київ"))
    with pytest.raises(SystemExit, match="no interrupted run"):
        run_eval(
            cfg,
            items=items,
            store=_hit_store("q0"),
            launcher=launcher,
            runner_fn=_runner_fn(_hit_store("q0"), launcher, cfg),
            mirror=lambda *a: None,
            emit=False,
            resume=cfg.run_dir("20260101T000000.0Z-missing"),
            sleep=lambda _s: None,
        )


def test_journal_record_is_idempotent_and_trims_state(tmp_path):
    j = durability.CaseJournal(tmp_path / durability.JOURNAL_NAME)
    state = {
        "retrieved": [{"doc_id": "d", "char_start": 0, "char_end": 3, "text": "abc"}],
        "answer": "Київ",
        "status": "ok",
        "error": None,
        "usage": {"completion_tokens": 2},
        "query_processed": "processed q",
        "query_corrections": 1,
        "query_hypothetical_answer": "hypothesis",
        "query_decomposition": '{"subqueries":["part"]}',
        "query_subqueries": ["part"],
        "context": "SHOULD NOT BE JOURNALED",
        "question": "q",
    }
    j.record("uk-1", state)
    j.record("uk-1", state)  # idempotent: no second line
    lines = (tmp_path / durability.JOURNAL_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["item_id"] == "uk-1"
    assert "context" not in payload["state"] and "question" not in payload["state"]
    assert payload["state"]["query_hypothetical_answer"] == "hypothesis"
    assert payload["state"]["query_subqueries"] == ["part"]

    # A fresh journal reloads the record and skips a trailing malformed (killed-run) line.
    with (tmp_path / durability.JOURNAL_NAME).open("a", encoding="utf-8") as fh:
        fh.write('{"item_id": "uk-2", "sta')  # truncated
    reloaded = durability.CaseJournal(tmp_path / durability.JOURNAL_NAME)
    assert reloaded.load() == 1
    assert reloaded.get("uk-1")["answer"] == "Київ"
