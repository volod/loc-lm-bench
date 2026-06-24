"""Device resolution for the RAG embedder (`LLB_EMBED_DEVICE` knob).

Pure resolution only -- no SentenceTransformer load, so no `[rag]` extra / GPU is needed.
"""

from llb import env
from llb.rag.embedding import Embedder


def test_resolve_device_defaults_to_none(monkeypatch):
    monkeypatch.delenv(env.LLB_EMBED_DEVICE, raising=False)
    assert Embedder()._resolve_device() is None  # auto-select (CUDA if available)


def test_resolve_device_reads_env(monkeypatch):
    monkeypatch.setenv(env.LLB_EMBED_DEVICE, "cpu")
    assert Embedder()._resolve_device() == "cpu"


def test_resolve_device_constructor_arg_overrides_env(monkeypatch):
    monkeypatch.setenv(env.LLB_EMBED_DEVICE, "cpu")
    assert Embedder(device="cuda:1")._resolve_device() == "cuda:1"
