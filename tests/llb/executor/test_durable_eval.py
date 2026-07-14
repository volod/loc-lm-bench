"""Durability: per-case retry, backend relaunch, and journal-backed resume (durable-eval-runner).

Every fault is driven by fakes -- a flaky/failing responder, a relaunch-counting launcher, and an
injected no-op sleep -- so the whole recovery vertical runs without an endpoint, GPU, or real clock.
"""

import json


from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT, BackendLauncher, ChatResult
from llb.core.config import RunConfig
from llb.eval import graph
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


# --- journal unit behavior ------------------------------------------------------------------
