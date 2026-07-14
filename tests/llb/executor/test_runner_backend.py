"""Tests for runner backend."""

from llb.backends.base import ChatResult
from llb.core.config import RunConfig
from llb.eval import common
from llb.executor.runner import run_eval
from test_runner import DOC, FakeLauncher, FakeStore, _runner_fn, gold_item


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
