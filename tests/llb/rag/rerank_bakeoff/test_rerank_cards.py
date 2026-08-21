"""The reranker card gate over an injected scorer (no torch, no download, no network).

Cross-encoder cards print their reference in whichever space their own snippet uses, and the two
registered ones disagree: jina prints sigmoid probabilities, gte prints raw logits. Getting that
wrong would fail a working candidate, so it is what these tests pin.
"""

import math

from llb.rag.encoders.card_parity import (
    STATUS_ERROR,
    STATUS_MISMATCH,
    STATUS_REPRODUCED,
    STATUS_UNPUBLISHED,
    expected_in_model_space,
)
from llb.rag.rerank_bakeoff.cards import (
    RERANK_CARD_REFERENCES,
    card_reference,
    check_rerank_card,
)

JINA = "jinaai/jina-reranker-v2-base-multilingual"
GTE = "Alibaba-NLP/gte-multilingual-reranker-base"


def _replaying(scores):
    """A scorer that hands back the next `len(texts)` values of `scores` per call."""
    remaining = list(scores)

    def scorer(_query, texts):
        taken, remaining[: len(texts)] = remaining[: len(texts)], []
        return taken

    return scorer


def test_an_unregistered_reranker_is_recorded_as_unchecked_not_as_verified():
    result = check_rerank_card("mixedbread-ai/mxbai-rerank-base-v2", _replaying([0.5]))
    assert result["status"] == STATUS_UNPUBLISHED


def test_a_sigmoid_card_is_reproduced_by_the_scorers_own_probabilities():
    reference = card_reference(JINA)
    assert reference is not None
    result = check_rerank_card(JINA, _replaying(reference.expectation.values))
    assert result["status"] == STATUS_REPRODUCED
    assert result["max_abs_diff"] == 0.0


def test_a_logit_card_is_squashed_before_comparison_so_a_working_candidate_passes():
    reference = card_reference(GTE)
    assert reference is not None
    logits = reference.expectation.values
    squashed = [1.0 / (1.0 + math.exp(-value)) for value in logits]
    assert check_rerank_card(GTE, _replaying(squashed))["status"] == STATUS_REPRODUCED
    # The un-squashed logits are what a naive comparison would have expected, and they are wrong.
    assert check_rerank_card(GTE, _replaying(list(logits)))["status"] == STATUS_MISMATCH


def test_each_card_query_is_scored_in_its_own_call_in_card_order():
    seen: list[str] = []

    def scorer(query, texts):
        seen.append(query)
        return [expected_in_model_space(card_reference(GTE).expectation)[len(seen) - 1]] * len(
            texts
        )

    result = check_rerank_card(GTE, scorer)
    assert result["status"] == STATUS_REPRODUCED
    assert seen == [query for query, _passages in card_reference(GTE).groups]


def test_a_scorer_that_raises_mid_probe_is_a_verdict_not_an_exception():
    def scorer(_query, _texts):
        raise ImportError("cannot import name 'create_position_ids_from_input_ids'")

    result = check_rerank_card(JINA, scorer)
    assert result["status"] == STATUS_ERROR
    assert "create_position_ids_from_input_ids" in result["detail"]


def test_every_declared_reference_names_its_card_and_matches_its_own_value_count():
    for model, reference in RERANK_CARD_REFERENCES.items():
        assert reference.model == model
        assert reference.source.startswith("https://huggingface.co/")
        scored = sum(len(passages) for _query, passages in reference.groups)
        assert scored == len(reference.expectation.values)
