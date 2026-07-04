"""Durability: per-case retry, backend relaunch, and journal-backed resume (durable-eval-runner).

Every fault is driven by fakes -- a flaky/failing responder, a relaunch-counting launcher, and an
injected no-op sleep -- so the whole recovery vertical runs without an endpoint, GPU, or real clock.
"""

import json
from operator import itemgetter

import pytest

from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT, BackendLauncher, ChatResult
from llb.config import RunConfig
from llb.eval import graph
from llb.executor import durability
from llb.executor.runner import run_eval
from llb.goldset.schema import GoldItem

DOC = "Київ є столицею України. Дніпро тече через місто."


def gold_item(item_id, question, reference, answer_text, split="final"):
    start = DOC.find(answer_text)
    return GoldItem(
        id=item_id,
        lang="uk",
        question=question,
        reference_answer=reference,
        source_doc_id="kyiv.txt",
        source_spans=[
            {
                "doc_id": "kyiv.txt",
                "char_start": start,
                "char_end": start + len(answer_text),
                "text": answer_text,
            }
        ],
        provenance="public-reused",
        verified=True,
        split=split,
    )


class FakeStore:
    def __init__(self, by_question):
        self._by_question = by_question

    def retrieve(self, question, k):
        return self._by_question.get(question, [])[:k]


class FakeLauncher(BackendLauncher):
    def __init__(self, responder):
        super().__init__(model="fake-uk", meta={"backend": "fake"})
        self._responder = responder
        self.starts = 0

    def start(self):
        self.starts += 1

    def chat(self, messages, max_tokens, temperature, timeout):
        return self._responder(self, messages)


def _runner_fn(store, launcher, cfg):
    retrieve = graph.make_retrieve_node(store, cfg.top_k)
    generate = graph.make_generate_node(
        launcher, cfg.max_tokens, cfg.temperature, cfg.request_timeout_s
    )

    def run(item):
        state = {"question": item.question, "gold_spans": []}
        state.update(retrieve(state))
        state.update(generate(state))
        return state

    return run


def _hit_store(q):
    return FakeStore(
        {q: [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}]}
    )


def _read_scores(run_dir):
    """Read persisted per-case scores back as a list of dicts from `scores.jsonl`."""
    jsonl = run_dir / "scores.jsonl"
    return [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]


# --- per-case retry -------------------------------------------------------------------------


def test_transient_failures_retry_then_succeed(tmp_path):
    q = "Яка столиця України?"
    items = [gold_item("uk-1", q, "Київ", "Київ")]

    fails_left = {"n": 2}

    def responder(_launcher, _messages):
        if fails_left["n"] > 0:
            fails_left["n"] -= 1
            return ChatResult(text="", error=ERR_TIMEOUT)
        return ChatResult(text="Київ", completion_tokens=2, latency_s=0.4)

    launcher = FakeLauncher(responder)
    cfg = RunConfig(data_dir=tmp_path, run_name="retry", top_k=3, model="fake-uk")
    result = run_eval(
        cfg,
        items=items,
        store=_hit_store(q),
        launcher=launcher,
        runner_fn=_runner_fn(_hit_store(q), launcher, cfg),
        mirror=lambda *a: None,
        emit=False,
        max_case_retries=2,
        sleep=lambda _s: None,
    )

    assert result["manifest"].durability["case_retries"] == 2
    assert result["manifest"].durability["backend_relaunches"] == 0
    # The case was recovered and scored ok (F1 == 1.0 against the reference "Київ").
    assert result["metrics"]["objective_score"] == 1.0
    run_dir = cfg.run_dir(result["run_timestamp"])
    assert _read_scores(run_dir)[0]["status"] == "ok"


def test_non_transport_status_is_never_retried(tmp_path):
    q = "Яка столиця України?"
    items = [gold_item("uk-1", q, "Київ", "Київ")]
    calls = {"n": 0}

    def responder(_launcher, _messages):
        calls["n"] += 1
        return ChatResult(text="")  # -> empty (a real outcome, not a transport fault)

    launcher = FakeLauncher(responder)
    cfg = RunConfig(data_dir=tmp_path, run_name="noretry", top_k=3, model="fake-uk")
    result = run_eval(
        cfg,
        items=items,
        store=_hit_store(q),
        launcher=launcher,
        runner_fn=_runner_fn(_hit_store(q), launcher, cfg),
        mirror=lambda *a: None,
        emit=False,
        max_case_retries=5,
        sleep=lambda _s: None,
    )
    assert calls["n"] == 1  # empty is terminal: called exactly once
    assert result["manifest"].durability["case_retries"] == 0


def test_exhausted_retries_relaunch_backend_then_recover(tmp_path):
    q = "Яка столиця України?"
    items = [gold_item("uk-1", q, "Київ", "Київ")]

    def responder(launcher, _messages):
        # Fails until the backend has been relaunched once (starts == 2).
        if launcher.starts >= 2:
            return ChatResult(text="Київ", completion_tokens=2, latency_s=0.3)
        return ChatResult(text="", error=ERR_BACKEND)

    launcher = FakeLauncher(responder)
    cfg = RunConfig(data_dir=tmp_path, run_name="relaunch", top_k=3, model="fake-uk")
    result = run_eval(
        cfg,
        items=items,
        store=_hit_store(q),
        launcher=launcher,
        runner_fn=_runner_fn(_hit_store(q), launcher, cfg),
        mirror=lambda *a: None,
        emit=False,
        max_case_retries=0,  # first failure immediately triggers a relaunch
        max_backend_relaunches=1,
        sleep=lambda _s: None,
    )
    assert result["manifest"].durability["backend_relaunches"] == 1
    assert result["metrics"]["objective_score"] == 1.0


# --- journal + resume -----------------------------------------------------------------------


def _uninterrupted(tmp_path, items, store, responder):
    launcher = FakeLauncher(responder)
    cfg = RunConfig(data_dir=tmp_path / "clean", run_name="dur", top_k=3, model="fake-uk")
    result = run_eval(
        cfg,
        items=items,
        store=store,
        launcher=launcher,
        runner_fn=_runner_fn(store, launcher, cfg),
        mirror=lambda *a: None,
        emit=False,
        sleep=lambda _s: None,
    )
    return _read_scores(cfg.run_dir(result["run_timestamp"]))


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
        "llb.executor.runner._run_timestamp", lambda run_id: "20260101T000000.0Z-abc"
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
        "llb.executor.runner._run_timestamp", lambda run_id: "20260101T000000.0Z-xyz"
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


# --- journal unit behavior ------------------------------------------------------------------


def test_journal_record_is_idempotent_and_trims_state(tmp_path):
    j = durability.CaseJournal(tmp_path / durability.JOURNAL_NAME)
    state = {
        "retrieved": [{"doc_id": "d", "char_start": 0, "char_end": 3, "text": "abc"}],
        "answer": "Київ",
        "status": "ok",
        "error": None,
        "usage": {"completion_tokens": 2},
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

    # A fresh journal reloads the record and skips a trailing malformed (killed-run) line.
    with (tmp_path / durability.JOURNAL_NAME).open("a", encoding="utf-8") as fh:
        fh.write('{"item_id": "uk-2", "sta')  # truncated
    reloaded = durability.CaseJournal(tmp_path / durability.JOURNAL_NAME)
    assert reloaded.load() == 1
    assert reloaded.get("uk-1")["answer"] == "Київ"
