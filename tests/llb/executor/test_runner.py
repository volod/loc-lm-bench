"""Walking-skeleton end-to-end (RAG core acceptance), driven by fakes.

Exercises the full vertical -- retrieve -> generate -> classify -> score -> aggregate ->
persist -- without FAISS, langgraph, Ollama, or a GPU, by injecting a fake store, a fake
launcher, and a runner_fn that composes the real eval-graph node closures sequentially.
"""

import json
from pathlib import Path

import pytest

from llb.backends.base import BackendLauncher, ChatResult
from llb.core.config import RunConfig
from llb.eval import common
from llb.eval import graph
from llb.executor import runner_setup
from llb.executor.runner import run_eval
from llb.goldset.schema import GoldItem
from llb.prompt_system.template import PromptPackage, TemplateFields

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
    """Returns preset chunks per question (doc_id + offsets drive retrieval scoring)."""

    def __init__(self, by_question):
        self._by_question = by_question

    def retrieve(self, question, k):
        return self._by_question.get(question, [])[:k]


class FakeLauncher(BackendLauncher):
    def __init__(self, responder):
        super().__init__(model="fake-uk", meta={"backend": "fake"})
        self._responder = responder

    def chat(self, messages, max_tokens, temperature, timeout):
        return self._responder(messages)


def _runner_fn(store, launcher, cfg):
    retrieve = graph.make_retrieve_node(store, cfg.top_k)
    generate = graph.make_generate_node(
        launcher, cfg.max_tokens, cfg.temperature, cfg.request_timeout_s
    )

    def run(item):
        state = {
            "question": item.question,
            "gold_spans": [s.model_dump() for s in item.source_spans],
        }
        state.update(retrieve(state))
        state.update(generate(state))
        return state

    return run


def test_walking_skeleton_end_to_end(tmp_path):
    hit_q = "Яка столиця України?"
    miss_q = "Що тече через місто?"
    items = [
        gold_item("uk-1", hit_q, "Київ", "Київ"),
        gold_item("uk-2", miss_q, "Дніпро", "Дніпро"),
    ]
    store = FakeStore(
        {
            # uk-1: chunk overlaps the gold span (doc kyiv.txt, 0..4) -> hit + correct answer
            hit_q: [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}],
            # uk-2: chunk is a different doc -> retrieval miss
            miss_q: [{"doc_id": "other.txt", "char_start": 0, "char_end": 30, "text": "noise"}],
        }
    )

    def responder(messages):
        content = messages[-1]["content"]
        if "столиц" in content:  # only uk-1's context/question mentions the capital
            return ChatResult(text="Київ", completion_tokens=2, latency_s=0.4)
        return ChatResult(text="", completion_tokens=0, latency_s=0.2)  # -> empty

    launcher = FakeLauncher(responder)
    cfg = RunConfig(data_dir=tmp_path, run_name="skeleton-test", top_k=3, model="fake-uk")

    result = run_eval(
        cfg,
        items=items,
        store=store,
        launcher=launcher,
        runner_fn=_runner_fn(store, launcher, cfg),
        mirror=lambda *a: None,
        emit=False,
    )

    # One ranked row, ranked #1, both cases counted.
    rows = result["rows"]
    assert len(rows) == 1 and rows[0]["rank"] == 1 and rows[0]["n_cases"] == 2
    assert rows[0]["model"] == "fake-uk"

    # objective = mean(f1=1.0 for uk-1, f1=0.0 for uk-2) = 0.5; reliability 0.5 (one empty)
    assert result["metrics"]["objective_score"] == 0.5
    assert result["metrics"]["reliability"] == 0.5
    assert result["manifest"].split == "final"

    # retrieval: uk-1 hits, uk-2 misses -> recall 0.5
    assert result["retrieval"]["recall_at_k"] == 0.5

    # canonical record on disk: scores.jsonl (single format, independent of installed extras)
    run_dir = cfg.run_dir(result["run_timestamp"])
    assert run_dir == Path(result["paths"]["manifest"]).parent
    assert (run_dir / "manifest.json").exists()
    assert any(run_dir.glob("scores.*"))


def test_build_messages_applies_prompt_system_package():
    pkg = PromptPackage(
        system_prompt="SYS PROMPT",
        additional_prompt="AUGMENTED KNOWLEDGE",
        fields=TemplateFields(),
        dropped_context={"budget_tokens": 10, "used_tokens": 1, "sections": []},
    )
    messages = graph.build_messages("Питання?", "BASE CONTEXT", pkg)

    assert messages[0]["content"].startswith("SYS PROMPT")
    assert graph.SYSTEM_PROMPT in messages[0]["content"]
    assert "AUGMENTED KNOWLEDGE" in messages[1]["content"]
    assert "BASE CONTEXT" in messages[1]["content"]


def test_run_eval_persists_prompt_system_provenance(tmp_path):
    q = "Яка столиця України?"
    items = [gold_item("uk-1", q, "Київ", "Київ")]
    cfg = RunConfig(data_dir=tmp_path, run_name="prompt-system", model="fake-uk")
    provenance = {
        "prompt_system_id": "ps-test",
        "corpus_digest": "corpus",
        "mapping_digest": "mapping",
        "template_revision": "template",
        "tokenizer": "char-ratio",
        "context_window": 4096,
        "prompt_budget_tokens": 3000,
    }
    result = run_eval(
        cfg,
        items=items,
        launcher=FakeLauncher(lambda messages: ChatResult(text="Київ")),
        runner_fn=lambda item: {
            "answer": "Київ",
            "status": common.OK,
            "retrieved": [],
            "usage": {},
        },
        prompt_system_provenance=provenance,
        mirror=lambda *a: None,
        emit=False,
    )

    manifest = result["manifest"]
    assert manifest.config["prompt_system"] == "ps-test"
    assert manifest.prompt_system_provenance == provenance
    persisted = json.loads(Path(result["paths"]["manifest"]).read_text(encoding="utf-8"))
    assert persisted["config"]["prompt_system"] == "ps-test"
    assert persisted["prompt_system_provenance"] == provenance


def test_run_eval_wires_trusted_judge_and_persists_scores(tmp_path):
    q = "Яка столиця України?"
    items = [gold_item("uk-1", q, "Київ", "Київ")]
    store = FakeStore(
        {q: [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}]}
    )
    launcher = FakeLauncher(lambda _m: ChatResult(text="Київ", completion_tokens=2, latency_s=0.4))
    cfg = RunConfig(
        data_dir=tmp_path,
        run_name="judge-test",
        top_k=3,
        model="fake-uk",
        judge_model="judge-x",
        judge_base_url="http://localhost:9000/v1",
    )
    calls = {}

    def scorer(records, judge_model):
        calls["model"] = judge_model
        assert records[0]["question"] == q and records[0]["answer"] == "Київ"
        assert records[0]["contexts"]  # retrieved chunk text passed through
        return [{"faithfulness": 1.0, "answer_relevancy": 0.8} for _ in records]

    result = run_eval(
        cfg,
        items=items,
        store=store,
        launcher=launcher,
        runner_fn=_runner_fn(store, launcher, cfg),
        mirror=lambda *a: None,
        judge_rho=0.7,  # >= 0.6 threshold -> judge trusted -> enters the blend
        judge_scorer=scorer,
        emit=False,
    )

    assert calls["model"] == "judge-x"
    assert result["metrics"]["judge_score"] == 0.9  # mean(1.0, 0.8)
    row = result["rows"][0]
    assert row["judge"] == 0.9
    assert row["quality"] == 0.95  # blend objective 1.0 with judge 0.9 at weight 0.5
    assert result["manifest"].judge["provider"] == "deepeval-geval"
    assert result["manifest"].judge["base_url"] == "http://localhost:9000/v1"


def test_run_eval_demotes_uncalibrated_judge(tmp_path):
    q = "Яка столиця України?"
    items = [gold_item("uk-1", q, "Київ", "Київ")]
    store = FakeStore(
        {q: [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}]}
    )
    launcher = FakeLauncher(lambda _m: ChatResult(text="Київ", completion_tokens=2, latency_s=0.4))
    cfg = RunConfig(data_dir=tmp_path, run_name="demote", model="fake-uk", judge_model="judge-x")

    def scorer(records, judge_model):  # pragma: no cover - must NOT run when demoted
        raise AssertionError("judge must not run without calibration")

    result = run_eval(
        cfg,
        items=items,
        store=store,
        launcher=launcher,
        runner_fn=_runner_fn(store, launcher, cfg),
        mirror=lambda *a: None,
        judge_rho=None,  # uncalibrated -> demoted -> objective ranks alone
        judge_scorer=scorer,
        emit=False,
    )
    assert "judge_score" not in result["metrics"]
    assert result["rows"][0]["judge"] is None


def test_run_eval_errors_when_split_empty(tmp_path):
    cfg = RunConfig(data_dir=tmp_path, run_name="empty")
    try:
        run_eval(
            cfg,
            items=[],
            launcher=FakeLauncher(lambda m: ChatResult(text="")),
            runner_fn=lambda it: {},
            emit=False,
        )
        raise AssertionError("expected SystemExit on empty eval set")
    except SystemExit:
        pass


@pytest.mark.slow
def test_run_eval_emits_prefilled_worksheet(tmp_path):
    items = [gold_item("cal-1", "Яка столиця України?", "Київ", "Київ", split="calibration")]
    store = FakeStore(
        {
            "Яка столиця України?": [
                {"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}
            ]
        }
    )
    launcher = FakeLauncher(
        lambda messages: ChatResult(text="Київ", completion_tokens=2, latency_s=0.3)
    )
    cfg = RunConfig(data_dir=tmp_path, run_name="cal", model="fake-uk")
    ws = tmp_path / "worksheet.csv"

    run_eval(
        cfg,
        items=items,
        store=store,
        launcher=launcher,
        runner_fn=_runner_fn(store, launcher, cfg),
        mirror=lambda *a: None,
        split="calibration",
        worksheet=ws,
        emit=False,
    )

    text = ws.read_text(encoding="utf-8")
    assert "cal-1" in text and "Київ" in text and "human_rating" in text


def test_worksheet_prefills_judge_rating_ungated(tmp_path):
    import csv

    q = "Яка столиця України?"
    items = [gold_item("cal-1", q, "Київ", "Київ", split="calibration")]
    store = FakeStore(
        {q: [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}]}
    )
    launcher = FakeLauncher(
        lambda messages: ChatResult(text="Київ", completion_tokens=2, latency_s=0.3)
    )
    cfg = RunConfig(data_dir=tmp_path, run_name="cal-judge", model="fake-uk", judge_model="judge-x")
    ws = tmp_path / "worksheet.csv"

    def scorer(records, judge_model):
        return [{"faithfulness": 1.0, "answer_relevancy": 0.6} for _ in records]

    run_eval(
        cfg,
        items=items,
        store=store,
        launcher=launcher,
        runner_fn=_runner_fn(store, launcher, cfg),
        mirror=lambda *a: None,
        split="calibration",
        worksheet=ws,
        judge_rho=None,  # ungated for the worksheet: judge runs even when not (yet) calibrated
        judge_scorer=scorer,
        emit=False,
    )

    rows = list(csv.DictReader(ws.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["judge_rating"] == "0.8"  # mean(1.0, 0.6), pre-filled by the ungated judge
    assert rows[0]["human_rating"] == ""  # human still fills this column


def test_score_case_records_semantic_with_embedder():
    from llb.executor.cases import score_case

    class Emb:
        def encode_queries(self, texts):
            return [[1.0, 0.0] for _ in texts]

    item = gold_item("x", "q", "Київ", "Київ")
    state = {"answer": "Київ", "status": common.OK, "retrieved": [], "usage": {}}
    row = score_case(item, state, embedder=Emb())
    assert row["semantic"] == 1.0


def test_make_launcher_resolves_vllm():
    from llb.backends.vllm import VllmLauncher
    from llb.executor.runner_backend import _make_launcher

    cfg = RunConfig(
        backend="vllm",
        model="org/Model",
        gpu_memory_utilization=0.9,
        cpu_offload_gb=16,
        kv_offloading_size_gb=32,
    )
    launcher = _make_launcher(cfg)
    assert isinstance(launcher, VllmLauncher)
    assert launcher.gpu_memory_utilization == 0.9 and "serve" in launcher.command()
    assert launcher.cpu_offload_gb == 16
    assert launcher.kv_offloading_size_gb == 32


def test_run_eval_records_telemetry(tmp_path):
    q = "Яка столиця України?"
    items = [gold_item("t-1", q, "Київ", "Київ")]
    store = FakeStore(
        {q: [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}]}
    )
    launcher = FakeLauncher(
        lambda messages: ChatResult(text="Київ", completion_tokens=4, latency_s=0.5)
    )
    cfg = RunConfig(data_dir=tmp_path, run_name="telem", model="fake-uk", measure_telemetry=True)

    result = run_eval(
        cfg,
        items=items,
        store=store,
        launcher=launcher,
        runner_fn=_runner_fn(store, launcher, cfg),
        mirror=lambda *a: None,
        emit=False,
    )

    telemetry = result["telemetry"]
    assert telemetry["steady_tokens_per_s"] == 8.0  # 4 tokens / 0.5 s, fixed prompt set
    assert telemetry["backend"] == "fake"
    assert telemetry["load_time_s"] is None
    assert result["manifest"].telemetry == telemetry
    assert result["rows"][0]["tokens_per_s"] == 8.0


def test_run_eval_scores_only_verified_items(tmp_path):
    verified = gold_item("verified", "q1", "Київ", "Київ")
    unverified = gold_item("draft", "q2", "Київ", "Київ").model_copy(update={"verified": False})
    launcher = FakeLauncher(lambda messages: ChatResult(text="Київ"))
    cfg = RunConfig(data_dir=tmp_path, run_name="verified-only", model="fake-uk")

    result = run_eval(
        cfg,
        items=[unverified, verified],
        launcher=launcher,
        runner_fn=lambda item: {"answer": "Київ", "status": common.OK},
        mirror=lambda *args: None,
        emit=False,
    )

    assert result["manifest"].n_cases == 1


def test_failed_eval_removes_unpublished_staging_directory(tmp_path, monkeypatch):
    item = gold_item("failure", "q", "Київ", "Київ")
    cfg = RunConfig(data_dir=tmp_path, run_name="failed", model="fake-uk")
    monkeypatch.setattr("llb.executor.runner_target._run_timestamp", lambda run_id: "fixed-run")
    staging_dir = cfg.run_staging_dir("fixed-run")
    staging_dir.mkdir(parents=True)
    (staging_dir / "backend.log").write_text("partial", encoding="utf-8")

    def fail_case(item):
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        run_eval(
            cfg,
            items=[item],
            launcher=FakeLauncher(lambda messages: ChatResult(text="")),
            runner_fn=fail_case,
            mirror=lambda *args: None,
            emit=False,
        )

    assert not staging_dir.exists()
    assert not cfg.run_dir("fixed-run").exists()


def test_load_store_refuses_embedder_mismatch(tmp_path, monkeypatch):
    """A store built with a different embedder than config.embedding_model aborts with a clear msg."""

    class _FakeStore:
        meta = {"embedding_model": "BAAI/bge-m3"}

    monkeypatch.setattr("llb.rag.store.RagStore.load", classmethod(lambda cls, d: _FakeStore()))
    cfg = RunConfig(data_dir=tmp_path, embedding_model="intfloat/multilingual-e5-base")
    with pytest.raises(SystemExit, match="embedder mismatch"):
        runner_setup._load_store(cfg)


def test_load_store_accepts_matching_embedder(tmp_path, monkeypatch):
    class _FakeStore:
        meta = {"embedding_model": "intfloat/multilingual-e5-base"}

    monkeypatch.setattr("llb.rag.store.RagStore.load", classmethod(lambda cls, d: _FakeStore()))
    cfg = RunConfig(data_dir=tmp_path, embedding_model="intfloat/multilingual-e5-base")
    assert isinstance(runner_setup._load_store(cfg), _FakeStore)
