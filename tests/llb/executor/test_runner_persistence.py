"""Tests for runner persistence."""

import pytest
from llb.backends.base import ChatResult
from llb.core.config import RunConfig
from llb.eval import common
from llb.executor import runner_retrieval
from llb.executor.runner import run_eval
from test_runner import FakeLauncher, gold_item


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
        runner_retrieval._load_store(cfg)


def test_load_store_accepts_matching_embedder(tmp_path, monkeypatch):
    class _FakeStore:
        meta = {"embedding_model": "intfloat/multilingual-e5-base"}

    monkeypatch.setattr("llb.rag.store.RagStore.load", classmethod(lambda cls, d: _FakeStore()))
    cfg = RunConfig(data_dir=tmp_path, embedding_model="intfloat/multilingual-e5-base")
    assert isinstance(runner_retrieval._load_store(cfg), _FakeStore)
