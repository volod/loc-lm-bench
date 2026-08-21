"""Device / remote-code resolution for the RAG embedder.

Pure resolution only -- no SentenceTransformer load, so no `[rag]` extra / GPU is needed. The
per-family query/passage conventions live in `test_embedding_families.py`.
"""

import pytest

from llb.core import env
from llb.rag.encoders.embedder import Embedder, remote_code_opt_in
from llb.rag.encoders.families import FAMILY_E5, FAMILY_JINA_V3, FAMILY_UNKNOWN


def test_resolve_device_defaults_to_none(monkeypatch):
    monkeypatch.delenv(env.LLB_EMBED_DEVICE, raising=False)
    assert Embedder()._resolve_device() is None  # auto-select (CUDA if available)


def test_resolve_device_reads_env(monkeypatch):
    monkeypatch.setenv(env.LLB_EMBED_DEVICE, "cpu")
    assert Embedder()._resolve_device() == "cpu"


def test_resolve_device_constructor_arg_overrides_env(monkeypatch):
    monkeypatch.setenv(env.LLB_EMBED_DEVICE, "cuda:1")
    assert Embedder(device="cuda:0")._resolve_device() == "cuda:0"


def test_embedder_exposes_family_and_convention():
    embedder = Embedder("intfloat/multilingual-e5-base")
    assert embedder.family == FAMILY_E5
    assert embedder.convention.query_prefix == "query: "


def test_embedder_release_clears_loaded_weights():
    embedder = Embedder("intfloat/multilingual-e5-small")
    embedder._model = object()
    embedder.release()
    assert embedder._model is None


# --- trust_remote_code gate -------------------------------------------------------------------


def test_load_kwargs_empty_for_a_family_that_needs_no_remote_code(monkeypatch):
    monkeypatch.delenv(env.LLB_TRUST_REMOTE_CODE, raising=False)
    assert Embedder("intfloat/multilingual-e5-base")._load_kwargs() == {}


def test_remote_code_family_is_refused_without_the_opt_in(monkeypatch):
    monkeypatch.delenv(env.LLB_TRUST_REMOTE_CODE, raising=False)
    embedder = Embedder("jinaai/jina-embeddings-v3")
    assert embedder.family == FAMILY_JINA_V3
    with pytest.raises(SystemExit) as excinfo:
        embedder._load_kwargs()
    # The refusal must name the knob AND the card, or the operator cannot act on it.
    assert env.LLB_TRUST_REMOTE_CODE in str(excinfo.value)
    assert "huggingface.co/jinaai/jina-embeddings-v3" in str(excinfo.value)


def test_remote_code_family_loads_under_the_env_opt_in(monkeypatch):
    monkeypatch.setenv(env.LLB_TRUST_REMOTE_CODE, "1")
    assert Embedder("Alibaba-NLP/gte-multilingual-base")._load_kwargs() == {
        "trust_remote_code": True
    }


def test_constructor_opt_in_beats_the_env_knob(monkeypatch):
    monkeypatch.delenv(env.LLB_TRUST_REMOTE_CODE, raising=False)
    assert Embedder("jinaai/jina-embeddings-v3", trust_remote_code=True)._load_kwargs() == {
        "trust_remote_code": True
    }
    monkeypatch.setenv(env.LLB_TRUST_REMOTE_CODE, "1")
    with pytest.raises(SystemExit):
        Embedder("jinaai/jina-embeddings-v3", trust_remote_code=False)._load_kwargs()


@pytest.mark.parametrize(
    "value, expected",
    [("1", True), ("true", True), ("YES", True), ("0", False), ("", False), ("no", False)],
)
def test_remote_code_opt_in_reads_truthy_env_values(monkeypatch, value, expected):
    monkeypatch.setenv(env.LLB_TRUST_REMOTE_CODE, value)
    assert remote_code_opt_in() is expected


def test_unknown_family_warns_instead_of_loading_silently(monkeypatch, caplog):
    monkeypatch.delenv(env.LLB_TRUST_REMOTE_CODE, raising=False)
    embedder = Embedder("some-vendor/never-registered-encoder")
    assert embedder.family == FAMILY_UNKNOWN
    with caplog.at_level("WARNING"):
        assert embedder._load_kwargs() == {}
    assert "no registered query/passage convention" in caplog.text
