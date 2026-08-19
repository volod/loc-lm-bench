"""The encoder card gate over injected encoders (no torch, no download, no network).

What the gate is FOR: it runs each card's own retrieval example through the query/passage
convention this repo registered, so a mismatch means one of two things a bake-off must never
publish -- the weights are wrong, or the format we score them under is not the one the card
documents. Both produce the same verdict, which is the point: either way the row is unreadable.
"""

import pytest

from llb.rag.card_parity import (
    STATUS_ERROR,
    STATUS_MISMATCH,
    STATUS_REPRODUCED,
    STATUS_UNPUBLISHED,
)
from llb.rag.encoder_cards import (
    ENCODER_CARD_REFERENCES,
    card_reference,
    check_encoder_card,
    similarity_matrix,
)

GTE = "Alibaba-NLP/gte-multilingual-base"
JINA = "jinaai/jina-embeddings-v3"


def _unit(index: int, dim: int = 4) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(dim)]


def _encoder(vectors):
    """An encoder that returns `vectors[i]` for the i-th text it is handed."""

    def encode(texts):
        return [vectors[i] for i in range(len(texts))]

    return encode


def test_similarity_matrix_is_row_major_over_normalized_vectors():
    queries = [[3.0, 0.0]]
    passages = [[1.0, 0.0], [0.0, 2.0]]
    assert similarity_matrix(queries, passages) == pytest.approx((1.0, 0.0))


def test_an_id_with_no_declared_reference_is_recorded_as_unchecked_not_as_verified():
    result = check_encoder_card(
        "intfloat/multilingual-e5-base",
        encode_queries=_encoder([_unit(0)]),
        encode_passages=_encoder([_unit(0)]),
    )
    assert result["status"] == STATUS_UNPUBLISHED


def test_an_encoder_reproducing_its_card_clears_the_gate():
    reference = card_reference(GTE)
    assert reference is not None
    expected = reference.expectation.values
    # One query, three passages: build passage vectors whose cosine with the query IS the card.
    queries = [[1.0, 0.0]]
    passages = [[value, (1 - value**2) ** 0.5] for value in expected]
    result = check_encoder_card(
        GTE, encode_queries=_encoder(queries), encode_passages=_encoder(passages)
    )
    assert result["status"] == STATUS_REPRODUCED
    assert result["source"] == reference.source


def test_an_encoder_that_runs_and_misses_its_card_is_refused():
    queries = [[1.0, 0.0]]
    passages = [[0.738, 0.675], [0.676, 0.737], [0.598, 0.801]]
    result = check_encoder_card(
        GTE, encode_queries=_encoder(queries), encode_passages=_encoder(passages)
    )
    assert result["status"] == STATUS_MISMATCH


def test_an_encoder_whose_probe_raises_is_a_verdict_not_an_exception():
    def boom(_texts):
        raise IndexError("index 135484743353454 is out of bounds for dimension 0 with size 11")

    result = check_encoder_card(GTE, encode_queries=boom, encode_passages=boom)
    assert result["status"] == STATUS_ERROR
    assert "IndexError" in result["detail"]


def test_a_card_with_no_numbers_is_checked_against_its_own_reference_call():
    """jina-v3's card publishes a snippet, not values -- so the snippet IS the reference."""
    scored = [_unit(0), _unit(1), _unit(2), _unit(3)]
    seen: list[tuple[str, str]] = []
    reference = card_reference(JINA)
    assert reference is not None

    def reference_encode(texts, task):
        seen.append((task, texts[0]))
        return [scored[i] for i in range(len(texts))] if task == "retrieval.query" else scored[1:]

    result = check_encoder_card(
        JINA,
        encode_queries=_encoder([_unit(0)]),
        encode_passages=_encoder(scored[1:]),
        reference_encode=reference_encode,
    )
    assert result["status"] == STATUS_REPRODUCED
    # The reference is made with the SAME LoRA adapters the scored path selects, per side, and on
    # the RAW texts -- the model applies its own declared prompt, the registry applies its copy.
    assert [task for task, _text in seen] == ["retrieval.query", "retrieval.passage"]
    assert [text for _task, text in seen] == [reference.queries[0], reference.passages[0]]


def test_a_card_with_no_numbers_and_no_reference_implementation_cannot_be_gated_silently():
    result = check_encoder_card(
        JINA, encode_queries=_encoder([_unit(0)]), encode_passages=_encoder([_unit(1)] * 3)
    )
    assert result["status"] == STATUS_ERROR
    assert "reference implementation" in result["detail"]


def test_every_declared_reference_names_the_card_it_was_read_from():
    for model, reference in ENCODER_CARD_REFERENCES.items():
        assert reference.model == model
        assert reference.source.startswith("https://huggingface.co/")
        assert reference.queries and reference.passages
        published = reference.expectation.values
        if published:
            assert len(published) == len(reference.queries) * len(reference.passages)
        else:
            assert reference.reference_implementation
