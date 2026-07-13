"""Opt-in API embedder (Cohere via litellm) for the embedding bake-off.

Budget arithmetic and the input_type mapping are driven by an injected fake embed callable
(the "fake litellm client"), so no key or network is touched. The numpy-normalized encode
round-trip runs only where numpy is installed (the full test extra).
"""

import pytest

from llb.prep.frontier_telemetry import ProvenanceLog
from llb.rag.api_embedder import (
    COHERE_PASSAGE_INPUT_TYPE,
    COHERE_QUERY_INPUT_TYPE,
    ApiEmbedder,
    BudgetExceeded,
    record_embed_cost,
)


def test_record_embed_cost_accumulates_into_log():
    log = ProvenanceLog()
    record_embed_cost(log, "cohere/embed-multilingual-v3.0", 10, 0.001, max_usd=None)
    record_embed_cost(log, "cohere/embed-multilingual-v3.0", 5, 0.002, max_usd=None)
    summary = log.summary()
    assert summary["calls"] == 2
    assert summary["total_prompt_tokens"] == 15
    assert summary["total_cost_usd"] == pytest.approx(0.003)


def test_record_embed_cost_aborts_when_over_budget():
    log = ProvenanceLog()
    record_embed_cost(log, "m", 1, 0.004, max_usd=0.01)  # under cap: fine
    with pytest.raises(BudgetExceeded, match="budget exceeded"):
        record_embed_cost(log, "m", 1, 0.02, max_usd=0.01)  # crosses 0.01


def test_record_embed_cost_noop_without_log():
    record_embed_cost(None, "m", 1, 99.0, max_usd=0.0)  # no log -> nothing to enforce


def test_api_embedder_maps_input_type_per_direction():
    seen: list[str] = []

    def fake_embed(texts, input_type):
        seen.append(input_type)
        return [[1.0, 0.0] for _ in texts]

    pytest.importorskip("numpy")
    emb = ApiEmbedder("cohere/embed-multilingual-v3.0", fake_embed)
    emb.encode_queries(["коли"])
    emb.encode_passages(["текст"])
    assert seen == [COHERE_QUERY_INPUT_TYPE, COHERE_PASSAGE_INPUT_TYPE]


def test_api_embedder_l2_normalizes_and_batches():
    np = pytest.importorskip("numpy")

    calls: list[int] = []

    def fake_embed(texts, input_type):
        calls.append(len(texts))
        return [[3.0, 4.0] for _ in texts]  # norm 5 -> normalizes to (0.6, 0.8)

    emb = ApiEmbedder("m", fake_embed, batch_size=2)
    vecs = emb.encode_passages(["a", "b", "c"])  # 3 texts, batch 2 -> two calls
    assert calls == [2, 1]
    assert vecs.dtype == np.dtype("float32")
    assert vecs.shape == (3, 2)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert np.allclose(vecs[0], [0.6, 0.8])


def test_api_embedder_exposes_model_name_for_store_meta():
    assert ApiEmbedder("cohere/embed-multilingual-v3.0", lambda t, i: []).model_name == (
        "cohere/embed-multilingual-v3.0"
    )
