"""The transformers-major contract and the roster screening it drives.

Four roster candidates ship repository code written against transformers 4.x while the repo pins
5.x. What must never happen is either of the two silent outcomes: the row disappearing from the
table, or the row being scored anyway on numbers that do not reproduce its card.
"""

from llb.rag.encoders.candidate_screen import SKIP_LEGACY_STACK, SKIP_REMOTE_CODE
from llb.rag.embedding_bakeoff.roster import screen_candidates
from llb.rag.encoders.model_stack import (
    LEGACY_EXTRA,
    PINNED_TRANSFORMERS_MAJOR,
    REQUIRED_TRANSFORMERS_MAJOR_LEGACY,
    installed_transformers_major,
    major_of,
)
from llb.rag.rerank_bakeoff.roster import screen_rerankers

LEGACY_ENCODERS = ("Alibaba-NLP/gte-multilingual-base", "jinaai/jina-embeddings-v3")
LEGACY_RERANKERS = (
    "jinaai/jina-reranker-v2-base-multilingual",
    "Alibaba-NLP/gte-multilingual-reranker-base",
)


def test_major_of_reads_a_version_string_and_shrugs_at_nonsense():
    assert major_of("4.57.6") == 4 and major_of("5.12.1") == 5
    assert major_of(None) is None and major_of("") is None and major_of("main") is None


def test_the_installed_major_is_read_from_metadata_not_by_importing_transformers():
    # Whatever this interpreter holds, it must be a number the screen can compare against.
    assert installed_transformers_major() in (None, *range(1, 100))


def test_a_legacy_encoder_is_routed_off_the_pinned_stack_even_when_opted_in():
    runnable, skipped = screen_candidates(
        ["intfloat/multilingual-e5-base", *LEGACY_ENCODERS],
        allow_remote_code=True,
        transformers_major=PINNED_TRANSFORMERS_MAJOR,
    )
    assert runnable == ["intfloat/multilingual-e5-base"]
    assert [row["reason"] for row in skipped] == [SKIP_LEGACY_STACK] * 2
    for row in skipped:
        assert LEGACY_EXTRA in row["detail"]
        assert "compare-embeddings-legacy" in row["detail"]


def test_the_same_encoders_run_on_the_legacy_stack():
    runnable, skipped = screen_candidates(
        list(LEGACY_ENCODERS),
        allow_remote_code=True,
        transformers_major=REQUIRED_TRANSFORMERS_MAJOR_LEGACY,
    )
    assert runnable == list(LEGACY_ENCODERS) and skipped == []


def test_the_policy_decline_still_comes_first_so_the_operator_sees_the_choice_they_have():
    # Without the opt-in the row is DECLINED, not routed: the operator has not agreed to run
    # downloaded code anywhere yet, so naming a second environment would be premature.
    _runnable, skipped = screen_candidates(
        list(LEGACY_ENCODERS), transformers_major=PINNED_TRANSFORMERS_MAJOR
    )
    assert [row["reason"] for row in skipped] == [SKIP_REMOTE_CODE] * 2


def test_the_reranker_roster_carries_the_same_hole_and_the_same_route():
    runnable, skipped = screen_rerankers(
        ["BAAI/bge-reranker-v2-m3", *LEGACY_RERANKERS],
        allow_remote_code=True,
        transformers_major=PINNED_TRANSFORMERS_MAJOR,
    )
    assert runnable == ["BAAI/bge-reranker-v2-m3"]
    assert [row["reason"] for row in skipped] == [SKIP_LEGACY_STACK] * 2
    assert all("compare-rerankers-legacy" in row["detail"] for row in skipped)


def test_a_caller_that_declares_no_stack_screens_exactly_as_before():
    # The two lanes' other callers (and every existing test) pass no major; the third check is
    # then simply not run, rather than guessing which environment it is in.
    runnable, skipped = screen_candidates(list(LEGACY_ENCODERS), allow_remote_code=True)
    assert runnable == list(LEGACY_ENCODERS) and skipped == []
