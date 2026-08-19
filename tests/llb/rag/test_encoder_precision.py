"""Declared load precision: making the throughput column a model comparison, not a checkpoint one.

The confound this exists to remove: `multilingual-e5-large-instruct` ships float16 and `e5-large`
ships float32 at identical parameter count and dimension, so the instruct row's throughput lead is
its dtype. `auto` keeps each publisher's upload (which is what reproduces the recorded rows); a
declared dtype loads every candidate the same way.
"""

import pytest

from llb.core import env
from llb.rag.embedding import Embedder
from llb.rag.encoder_precision import (
    CONTROLLED_DTYPE,
    DTYPE_AUTO,
    DTYPE_FLOAT16,
    DTYPE_FLOAT32,
    SUPPORTED_DTYPES,
    UnsupportedDtypeError,
    load_model_kwargs,
    normalize_dtype,
    published_dtype,
)
from llb.rag.embedding_bakeoff_models import DEFAULT_LOCAL_CANDIDATES


def test_an_empty_request_is_auto_and_auto_pins_nothing():
    assert normalize_dtype(None) == DTYPE_AUTO and normalize_dtype("") == DTYPE_AUTO
    assert load_model_kwargs(DTYPE_AUTO) == {}


def test_a_declared_dtype_reaches_the_loader_under_the_key_both_stacks_honor():
    # `torch_dtype`, not the newer `dtype` alias: the legacy transformers 4.x pass has to honor the
    # same string, or the two passes are not comparable.
    assert load_model_kwargs(DTYPE_FLOAT32) == {"torch_dtype": "float32"}


def test_an_unsupported_precision_is_refused_rather_than_silently_ignored():
    with pytest.raises(UnsupportedDtypeError) as excinfo:
        normalize_dtype("fp8")
    assert "fp8" in str(excinfo.value)
    assert all(name in str(excinfo.value) for name in SUPPORTED_DTYPES)


def test_the_controlled_dtype_is_one_every_incumbent_already_ships():
    assert CONTROLLED_DTYPE == DTYPE_FLOAT32
    assert published_dtype("intfloat/multilingual-e5-base") == DTYPE_FLOAT32
    assert published_dtype("intfloat/multilingual-e5-large-instruct") == DTYPE_FLOAT16


def test_every_default_roster_candidate_records_the_precision_its_publisher_uploaded():
    # Without this the report can print a chunks/s column with no way to read it.
    assert all(published_dtype(model) is not None for model in DEFAULT_LOCAL_CANDIDATES)


def test_the_embedder_pins_the_precision_it_was_constructed_with(monkeypatch):
    monkeypatch.delenv(env.LLB_EMBED_DTYPE, raising=False)
    assert Embedder("intfloat/multilingual-e5-base")._load_kwargs() == {}
    kwargs = Embedder("intfloat/multilingual-e5-base", dtype=DTYPE_FLOAT16)._load_kwargs()
    assert kwargs == {"model_kwargs": {"torch_dtype": "float16"}}


def test_the_env_knob_reaches_an_embedder_nobody_passed_a_dtype_to(monkeypatch):
    # The store build, the lazy reload behind retrieve(), the card probe, and the throughput
    # profiler each construct their own Embedder; all of them must agree on the precision.
    monkeypatch.setenv(env.LLB_EMBED_DTYPE, DTYPE_FLOAT32)
    assert Embedder("BAAI/bge-m3")._load_kwargs() == {"model_kwargs": {"torch_dtype": "float32"}}


def test_a_declared_precision_rides_beside_the_remote_code_opt_in(monkeypatch):
    monkeypatch.setenv(env.LLB_EMBED_DTYPE, DTYPE_FLOAT32)
    kwargs = Embedder("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)._load_kwargs()
    assert kwargs == {"model_kwargs": {"torch_dtype": "float32"}, "trust_remote_code": True}


def test_an_unloaded_embedder_reports_no_effective_precision():
    assert Embedder("BAAI/bge-m3").effective_dtype() is None
