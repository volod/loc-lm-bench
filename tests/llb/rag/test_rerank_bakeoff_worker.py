"""Per-candidate process isolation (`llb.rag.rerank_bakeoff.worker`) over a stub cross-encoder.

No model is loaded: `CrossEncoderReranker` is replaced by a stub and the child is started with
`fork` so it inherits it. A real run uses `spawn` because the parent holds a live CUDA context by
then; CI has no CUDA, so forking here is safe and keeps the protocol -- handshake, scoring calls,
footprint reads, a load that fails, a child that dies mid-pass, release -- fully exercised.
"""

import multiprocessing as mp

import pytest

from llb.rag.rerank_bakeoff.models import ScorerLoadError
from llb.rag.rerank_bakeoff.worker import isolated_loader

# Real registered ids: the loader resolves each candidate's convention to decide trust_remote_code.
WORKING = "BAAI/bge-reranker-v2-m3"
FAILING = "mixedbread-ai/mxbai-rerank-base-v2"
DYING = "Qwen/Qwen3-Reranker-0.6B"


class _StubCrossEncoder:
    """Stands in for `CrossEncoderReranker` inside the child."""

    def __init__(self, model_name: str, device=None, **_kwargs):
        self.model_name = model_name
        self._model = None
        if model_name == FAILING:
            raise RuntimeError("device-side assert triggered")

    def __call__(self, question: str, texts: list[str]) -> list[float]:
        if self.model_name == DYING and question != "warmup":
            raise RuntimeError("CUDA out of memory")
        return [float(len(text)) for text in texts]

    def release(self) -> None:
        return None


@pytest.fixture
def stubbed_child(monkeypatch):
    import llb.rag.rerank as rerank_module

    monkeypatch.setattr(rerank_module, "CrossEncoderReranker", _StubCrossEncoder)
    fork_context = mp.get_context("fork")
    monkeypatch.setattr(mp, "get_context", lambda _method: fork_context)
    return isolated_loader(batch_size=4, dtype="auto")


def test_a_working_candidate_answers_scoring_calls_and_reports_its_load(stubbed_child):
    loaded = stubbed_child(WORKING)
    try:
        assert loaded.load_seconds >= 0.0
        assert loaded.scorer("питання", ["aa", "bbbb"]) == [2.0, 4.0]
        assert loaded.scorer("питання", []) == []
    finally:
        loaded.release()


def test_a_candidate_that_cannot_load_raises_with_the_hosts_own_error(stubbed_child):
    """The whole point of isolation: this failure must not reach the next candidate."""
    with pytest.raises(ScorerLoadError, match="device-side assert"):
        stubbed_child(FAILING)
    # The next candidate still loads in the same parent process.
    loaded = stubbed_child(WORKING)
    try:
        assert loaded.scorer("питання", ["aa"]) == [2.0]
    finally:
        loaded.release()


def test_a_child_that_dies_mid_pass_becomes_a_load_error_not_a_hang(stubbed_child):
    loaded = stubbed_child(DYING)
    try:
        with pytest.raises(ScorerLoadError, match="out of memory"):
            loaded.scorer("питання", ["aa"])
    finally:
        loaded.release()


def test_release_stops_the_child(stubbed_child):
    loaded = stubbed_child(WORKING)
    loaded.release()
    with pytest.raises(ScorerLoadError, match="exited|died"):
        loaded.scorer("питання", ["aa"])
