"""Tests for runner backend."""

from pathlib import Path

import pytest

from llb.backends.base import BackendLauncher, ChatResult
from llb.backends.launch_log import ServerLog
from llb.core.config import RunConfig
from llb.eval import common
from llb.executor.runner import run_eval
from tests.llb.executor.test_runner import DOC, FakeLauncher, FakeStore, _runner_fn, gold_item


def test_score_case_records_semantic_with_embedder():
    from llb.executor.cases import score_case

    class Emb:
        def encode_queries(self, texts):
            return [[1.0, 0.0] for _ in texts]

    item = gold_item("x", "q", "Київ", "Київ")
    state = {"answer": "Київ", "status": common.OK, "retrieved": [], "usage": {}}
    row = score_case(item, state, embedder=Emb())
    assert row["semantic"] == 1.0


def test_score_case_records_answer_side_span_coverage():
    """answer-side-span-coverage-metric: every case row says whether the ANSWER carried the
    item's gold spans, so the answer-quality lane can read it beside the objective."""
    from llb.executor.cases import score_case

    item = gold_item("x", "Яка столиця України?", "Київ", "Київ")
    state = {"answer": "Столиця - Київ.", "status": common.OK, "retrieved": [], "usage": {}}

    row = score_case(item, state)

    assert row["answer_span_coverage"] == 1.0
    assert row["answer_all_spans"] == 1.0
    assert row["answer_spans_measured"] == 1

    missed = score_case(item, {**state, "answer": "Невідомо."})

    assert missed["answer_span_coverage"] == 0.0
    assert missed["answer_all_spans"] == 0.0


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


class LoggingLauncher(BackendLauncher, ServerLog):
    """A subprocess-style launcher that writes a server log, optionally dying during startup like
    a vLLM engine that cannot allocate the card. `log_dir` is where `_make_launcher` puts it:
    inside the staging dir a failing run is about to delete."""

    def __init__(self, log_dir, *, says, dies=False):
        super().__init__(model="fake-uk", meta={"backend": "fake"})
        self.log_dir = log_dir
        self._says = says
        self._dies = dies

    def start(self):
        self.open_log("fake-8000.log").write(self._says)
        if self._dies:
            self.stop()
            raise self.annotate_launch_failure(RuntimeError("fake backend exited during startup"))

    def stop(self):
        self.close_log()

    def chat(self, messages, max_tokens, temperature, timeout):
        raise AssertionError("this launcher never serves a case")


def test_failed_launch_log_outlives_the_run_that_deleted_its_staging_dir(tmp_path, monkeypatch):
    """search-cell-loses-a-failed-launch-log: a cell's launcher logs into the temp run dir, which
    the failure path removes -- so the traceback must name a path that is still readable."""
    stamp = "20260101T000000.0Z-abc"
    monkeypatch.setattr("llb.executor.runner_target._run_timestamp", lambda run_id: stamp)
    cfg = RunConfig(data_dir=tmp_path, run_name="cell", model="fake-uk")
    staging = cfg.run_staging_dir(stamp)
    says = "engine died: no kernel image for device\n"
    launcher = LoggingLauncher(staging / "vllm", says=says, dies=True)
    q = "Яка столиця України?"

    with pytest.raises(RuntimeError, match="exited during startup") as excinfo:
        run_eval(
            cfg,
            items=[gold_item("t-1", q, "Київ", "Київ")],
            store=FakeStore({q: []}),
            launcher=launcher,
            runner_fn=lambda item: {},
            mirror=lambda *a: None,
            emit=False,
        )

    assert not staging.exists()  # the temp run dir is gone, as it is for every failed cell
    kept = Path(str(excinfo.value).split("startup log: ", 1)[1].rstrip(") "))
    assert kept.parent == tmp_path / "llb" / "logs"
    assert kept.read_text(encoding="utf-8") == says


def test_a_run_that_fails_after_a_healthy_launch_keeps_the_backend_log_too(tmp_path, monkeypatch):
    """The launcher preserved nothing (it started fine), so the staging teardown is what keeps
    the log -- and it keeps exactly one copy."""
    stamp = "20260101T000000.0Z-def"
    monkeypatch.setattr("llb.executor.runner_target._run_timestamp", lambda run_id: stamp)
    cfg = RunConfig(data_dir=tmp_path, run_name="cell", model="fake-uk")
    staging = cfg.run_staging_dir(stamp)
    launcher = LoggingLauncher(staging / "vllm", says="serving\n")
    q = "Яка столиця України?"

    def boom(item):
        raise RuntimeError("case blew up mid-run")

    with pytest.raises(RuntimeError, match="case blew up"):
        run_eval(
            cfg,
            items=[gold_item("t-1", q, "Київ", "Київ")],
            store=FakeStore({q: []}),
            launcher=launcher,
            runner_fn=boom,
            mirror=lambda *a: None,
            emit=False,
            max_case_retries=0,
        )

    kept = list((tmp_path / "llb" / "logs").iterdir())
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8") == "serving\n"
