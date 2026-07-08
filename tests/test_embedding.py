"""Device resolution + per-family query/passage conventions for the RAG embedder.

Pure resolution only -- no SentenceTransformer load, so no `[rag]` extra / GPU is needed.
"""

import pytest

from llb.core import env
from llb.rag.embedding import (
    BGE_QUERY_INSTRUCTION,
    FAMILY_BGE,
    FAMILY_BGE_M3,
    FAMILY_E5,
    FAMILY_PLAIN,
    Embedder,
    apply_passage_convention,
    apply_query_convention,
    embedding_family,
)


def test_resolve_device_defaults_to_none(monkeypatch):
    monkeypatch.delenv(env.LLB_EMBED_DEVICE, raising=False)
    assert Embedder()._resolve_device() is None  # auto-select (CUDA if available)


def test_resolve_device_reads_env(monkeypatch):
    monkeypatch.setenv(env.LLB_EMBED_DEVICE, "cpu")
    assert Embedder()._resolve_device() == "cpu"


def test_resolve_device_constructor_arg_overrides_env(monkeypatch):
    monkeypatch.setenv(env.LLB_EMBED_DEVICE, "cpu")
    assert Embedder(device="cuda:1")._resolve_device() == "cuda:1"


@pytest.mark.parametrize(
    "model, family",
    [
        ("intfloat/multilingual-e5-base", FAMILY_E5),
        ("intfloat/multilingual-e5-large", FAMILY_E5),
        ("BAAI/bge-m3", FAMILY_BGE_M3),
        ("BAAI/bge-large-en-v1.5", FAMILY_BGE),
        ("lang-uk/ukr-paraphrase-multilingual-mpnet-base", FAMILY_PLAIN),
    ],
)
def test_embedding_family_resolves_per_model(model, family):
    assert embedding_family(model) == family
    assert Embedder(model).family == family


def test_e5_prefixes_query_and_passage():
    assert apply_query_convention("intfloat/multilingual-e5-base", ["коли"]) == ["query: коли"]
    assert apply_passage_convention("intfloat/multilingual-e5-base", ["текст"]) == [
        "passage: текст"
    ]


def test_bge_m3_uses_no_prefix_on_either_side():
    # BGE-M3's retrieval default is NO instruction; scoring it with e5 prefixes would cap recall.
    assert apply_query_convention("BAAI/bge-m3", ["коли"]) == ["коли"]
    assert apply_passage_convention("BAAI/bge-m3", ["текст"]) == ["текст"]


def test_bge_v15_instructs_query_only():
    assert apply_query_convention("BAAI/bge-large-en-v1.5", ["q"]) == [BGE_QUERY_INSTRUCTION + "q"]
    assert apply_passage_convention("BAAI/bge-large-en-v1.5", ["p"]) == ["p"]  # passage untouched


def test_plain_paraphrase_model_is_symmetric_no_prefix():
    model = "lang-uk/ukr-paraphrase-multilingual-mpnet-base"
    assert apply_query_convention(model, ["q"]) == ["q"]
    assert apply_passage_convention(model, ["p"]) == ["p"]
