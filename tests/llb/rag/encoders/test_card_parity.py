"""The card-parity verdict: what "reproduces its own model card" means, and what it refuses.

`Alibaba-NLP/gte-multilingual-base` is the case these tests encode: a candidate that LOADS, encodes
without raising, and returns numbers its card does not publish. Loading is not evidence a model can
be ranked, so the verdict has to separate three states a report reader must never confuse --
reproduced, mismatched, and never checked.
"""

import math

import pytest

from llb.rag.encoders.card_parity import (
    MODE_PUBLISHED_VALUES,
    MODE_REFERENCE_IMPLEMENTATION,
    SKIP_CARD_PARITY,
    STATUS_ERROR,
    STATUS_MISMATCH,
    STATUS_REPRODUCED,
    STATUS_UNPUBLISHED,
    CardExpectation,
    blocks_scoring,
    compare_to_card,
    expected_in_model_space,
    parity_skip_row,
    probe_error_result,
    unpublished_result,
)

CARD = "https://huggingface.co/acme/encoder"


def test_published_values_within_tolerance_reproduce_the_card():
    expectation = CardExpectation(values=(0.30, 0.75), tolerance=0.01)
    result = compare_to_card("acme/encoder", CARD, expectation, (0.305, 0.748))
    assert result["status"] == STATUS_REPRODUCED
    assert result["mode"] == MODE_PUBLISHED_VALUES
    assert result["max_abs_diff"] == 0.005
    assert not blocks_scoring(result)


def test_a_candidate_that_runs_and_is_wrong_is_refused_with_both_sides_recorded():
    # The gte failure shape: every value plausible on its own, none of them the card's.
    expectation = CardExpectation(values=(0.3017, 0.7504, 0.3203))
    result = compare_to_card("acme/encoder", CARD, expectation, (0.738, 0.676, 0.598))
    assert result["status"] == STATUS_MISMATCH
    assert blocks_scoring(result)
    assert result["expected"] == [0.3017, 0.7504, 0.3203]
    assert result["observed"] == [0.738, 0.676, 0.598]
    assert CARD in result["detail"]


def test_a_card_that_prints_scaled_similarities_is_compared_in_the_models_space():
    # `multilingual-e5-large-instruct` prints `(a @ b.T) * 100`.
    expectation = CardExpectation(values=(91.92854, 67.58030), scale=100.0)
    assert expected_in_model_space(expectation) == pytest.approx((0.9192854, 0.675803))
    assert compare_to_card("m", CARD, expectation, (0.9193, 0.6758))["status"] == STATUS_REPRODUCED


def test_a_card_that_prints_logits_is_compared_after_the_scorers_own_sigmoid():
    # `gte-multilingual-reranker-base` prints raw logits; CrossEncoder returns the sigmoid.
    expectation = CardExpectation(values=(1.2315, 0.5923, 0.3041), transform="sigmoid")
    squashed = tuple(1.0 / (1.0 + math.exp(-v)) for v in (1.2315, 0.5923, 0.3041))
    result = compare_to_card("r", CARD, expectation, squashed)
    assert result["status"] == STATUS_REPRODUCED
    # Comparing the RAW logits against a sigmoid scorer would have failed the candidate wrongly.
    assert compare_to_card("r", CARD, expectation, (1.2315, 0.5923, 0.3041))["status"] == (
        STATUS_MISMATCH
    )


def test_a_probe_that_returns_the_wrong_shape_is_a_mismatch_not_a_crash():
    result = compare_to_card("m", CARD, CardExpectation(values=(0.1, 0.2)), (0.1,))
    assert result["status"] == STATUS_MISMATCH and result["max_abs_diff"] is None
    assert "2 values" in result["detail"]


def test_a_reference_implementation_reference_compares_against_the_computed_side():
    expectation = CardExpectation(tolerance=0.02)
    assert expectation.mode == MODE_REFERENCE_IMPLEMENTATION
    result = compare_to_card("m", CARD, expectation, (0.71, 0.53), expected=(0.7115, 0.5318))
    assert result["status"] == STATUS_REPRODUCED
    assert result["mode"] == MODE_REFERENCE_IMPLEMENTATION


def test_never_checked_is_a_distinct_state_from_reproduced_and_does_not_block():
    result = unpublished_result("acme/unlisted")
    assert result["status"] == STATUS_UNPUBLISHED
    assert not blocks_scoring(result)
    assert result["expected"] == [] and result["observed"] == []


def test_a_probe_that_raised_blocks_scoring_and_carries_the_hosts_own_message():
    result = probe_error_result("acme/encoder", CARD, "card probe failed: IndexError: oob")
    assert result["status"] == STATUS_ERROR and blocks_scoring(result)
    row = parity_skip_row(result, "gte-multilingual")
    assert row == {
        "model": "acme/encoder",
        "family": "gte-multilingual",
        "reason": SKIP_CARD_PARITY,
        "detail": "card probe failed: IndexError: oob",
    }
