"""Walking-skeleton end-to-end (Milestone 1 acceptance), driven by fakes.

Exercises the full vertical -- retrieve -> generate -> classify -> score -> aggregate ->
persist -- without FAISS, langgraph, Ollama, or a GPU, by injecting a fake store, a fake
launcher, and a runner_fn that composes the real eval-graph node closures sequentially.
"""

from pathlib import Path

from llb.backends.base import BackendLauncher, ChatResult
from llb.config import RunConfig
from llb.eval import graph
from llb.executor.runner import run_eval
from llb.goldset.schema import GoldItem

DOC = "Київ є столицею України. Дніпро тече через місто."


def gold_item(item_id, question, reference, answer_text, split="final"):
    start = DOC.find(answer_text)
    return GoldItem(
        id=item_id, lang="uk", question=question, reference_answer=reference,
        source_doc_id="kyiv.txt",
        source_spans=[{"doc_id": "kyiv.txt", "char_start": start,
                       "char_end": start + len(answer_text), "text": answer_text}],
        provenance="public-reused", verified=True, split=split,
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
        state = {"question": item.question,
                 "gold_spans": [s.model_dump() for s in item.source_spans]}
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
    store = FakeStore({
        # uk-1: chunk overlaps the gold span (doc kyiv.txt, 0..4) -> hit + correct answer
        hit_q: [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}],
        # uk-2: chunk is a different doc -> retrieval miss
        miss_q: [{"doc_id": "other.txt", "char_start": 0, "char_end": 30, "text": "noise"}],
    })

    def responder(messages):
        content = messages[-1]["content"]
        if "столиц" in content:  # only uk-1's context/question mentions the capital
            return ChatResult(text="Київ", completion_tokens=2, latency_s=0.4)
        return ChatResult(text="", completion_tokens=0, latency_s=0.2)  # -> empty

    launcher = FakeLauncher(responder)
    cfg = RunConfig(data_dir=tmp_path, run_name="skeleton-test", top_k=3, model="fake-uk")

    result = run_eval(
        cfg, items=items, store=store, launcher=launcher,
        runner_fn=_runner_fn(store, launcher, cfg),
        mirror=lambda *a: None, emit=False,
    )

    # One ranked row, ranked #1, both cases counted.
    rows = result["rows"]
    assert len(rows) == 1 and rows[0]["rank"] == 1 and rows[0]["n_cases"] == 2
    assert rows[0]["model"] == "fake-uk"

    # objective = mean(f1=1.0 for uk-1, f1=0.0 for uk-2) = 0.5; reliability 0.5 (one empty)
    assert result["metrics"]["objective_score"] == 0.5
    assert result["metrics"]["reliability"] == 0.5

    # retrieval: uk-1 hits, uk-2 misses -> recall 0.5
    assert result["retrieval"]["recall_at_k"] == 0.5

    # canonical record on disk (scores.parquet with pyarrow, else scores.jsonl)
    run_dir = cfg.run_dir(result["run_timestamp"])
    assert run_dir == Path(result["paths"]["manifest"]).parent
    assert (run_dir / "manifest.json").exists()
    assert any(run_dir.glob("scores.*"))


def test_run_eval_errors_when_split_empty(tmp_path):
    cfg = RunConfig(data_dir=tmp_path, run_name="empty")
    try:
        run_eval(cfg, items=[], launcher=FakeLauncher(lambda m: ChatResult(text="")),
                 runner_fn=lambda it: {}, emit=False)
        raise AssertionError("expected SystemExit on empty eval set")
    except SystemExit:
        pass


def test_run_eval_emits_prefilled_worksheet(tmp_path):
    items = [gold_item("cal-1", "Яка столиця України?", "Київ", "Київ", split="calibration")]
    store = FakeStore({"Яка столиця України?":
                       [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}]})
    launcher = FakeLauncher(lambda messages: ChatResult(text="Київ", completion_tokens=2,
                                                        latency_s=0.3))
    cfg = RunConfig(data_dir=tmp_path, run_name="cal", model="fake-uk")
    ws = tmp_path / "worksheet.csv"

    run_eval(cfg, items=items, store=store, launcher=launcher,
             runner_fn=_runner_fn(store, launcher, cfg), mirror=lambda *a: None,
             split="calibration", worksheet=ws, emit=False)

    text = ws.read_text(encoding="utf-8")
    assert "cal-1" in text and "Київ" in text and "human_rating" in text


def test_score_case_records_semantic_with_embedder():
    from llb.executor.cases import score_case

    class Emb:
        def encode_queries(self, texts):
            return [[1.0, 0.0] for _ in texts]

    item = gold_item("x", "q", "Київ", "Київ")
    state = {"answer": "Київ", "status": graph.OK, "retrieved": [], "usage": {}}
    row = score_case(item, state, embedder=Emb())
    assert row["semantic"] == 1.0


def test_make_launcher_resolves_vllm():
    from llb.backends.vllm import VllmLauncher
    from llb.executor.runner import _make_launcher

    cfg = RunConfig(backend="vllm", model="org/Model", gpu_memory_utilization=0.9)
    launcher = _make_launcher(cfg)
    assert isinstance(launcher, VllmLauncher)
    assert launcher.gpu_memory_utilization == 0.9 and "serve" in launcher.command()


def test_run_eval_records_telemetry(tmp_path):
    q = "Яка столиця України?"
    items = [gold_item("t-1", q, "Київ", "Київ")]
    store = FakeStore({q: [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24,
                            "text": DOC[:24]}]})
    launcher = FakeLauncher(lambda messages: ChatResult(text="Київ", completion_tokens=4,
                                                        latency_s=0.5))
    cfg = RunConfig(data_dir=tmp_path, run_name="telem", model="fake-uk", measure_telemetry=True)

    result = run_eval(cfg, items=items, store=store, launcher=launcher,
                      runner_fn=_runner_fn(store, launcher, cfg), mirror=lambda *a: None,
                      emit=False)

    telemetry = result["telemetry"]
    assert telemetry["steady_tokens_per_s"] == 8.0   # 4 tokens / 0.5 s, fixed prompt set
    assert telemetry["backend"] == "fake"
    assert result["manifest"].telemetry == telemetry
    assert result["rows"][0]["tokens_per_s"] == 8.0


def test_run_eval_scores_only_verified_items(tmp_path):
    verified = gold_item("verified", "q1", "Київ", "Київ")
    unverified = gold_item("draft", "q2", "Київ", "Київ").model_copy(
        update={"verified": False}
    )
    launcher = FakeLauncher(lambda messages: ChatResult(text="Київ"))
    cfg = RunConfig(data_dir=tmp_path, run_name="verified-only", model="fake-uk")

    result = run_eval(
        cfg,
        items=[unverified, verified],
        launcher=launcher,
        runner_fn=lambda item: {"answer": "Київ", "status": graph.OK},
        mirror=lambda *args: None,
        emit=False,
    )

    assert result["manifest"].n_cases == 1
