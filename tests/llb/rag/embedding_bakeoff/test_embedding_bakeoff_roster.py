"""Roster screening: refuse an unregistered candidate, decline a remote-code one visibly."""

import pytest

from llb.rag.encoders.candidate_screen import SKIP_REMOTE_CODE
from llb.rag.embedding_bakeoff.models import DEFAULT_LOCAL_CANDIDATES
from llb.rag.embedding_bakeoff.roster import UnregisteredCandidateError, screen_candidates

REMOTE_CODE_MODELS = ("Alibaba-NLP/gte-multilingual-base", "jinaai/jina-embeddings-v3")


def test_registered_local_candidates_pass_without_remote_code():
    runnable, skipped = screen_candidates(["intfloat/multilingual-e5-base", "BAAI/bge-m3"])
    assert runnable == ["intfloat/multilingual-e5-base", "BAAI/bge-m3"]
    assert skipped == []


def test_unregistered_candidate_is_refused_not_silently_scored():
    # Building it would encode with no instruction at all, so the row would understate an encoder
    # rather than rank it -- the one failure a bake-off must not commit.
    with pytest.raises(UnregisteredCandidateError) as excinfo:
        screen_candidates(["intfloat/multilingual-e5-base", "acme/mystery-encoder"])
    assert "acme/mystery-encoder" in str(excinfo.value)
    assert "encoders.families" in str(excinfo.value)


def test_remote_code_candidate_is_skipped_with_a_recorded_reason():
    runnable, skipped = screen_candidates(["intfloat/multilingual-e5-base", *REMOTE_CODE_MODELS])
    assert runnable == ["intfloat/multilingual-e5-base"]
    assert [row["model"] for row in skipped] == list(REMOTE_CODE_MODELS)
    for row in skipped:
        assert row["reason"] == SKIP_REMOTE_CODE
        assert "--allow-remote-code" in row["detail"]
        assert "huggingface.co/" in row["detail"]  # the card to review before opting in


def test_remote_code_candidate_runs_once_opted_in():
    runnable, skipped = screen_candidates(REMOTE_CODE_MODELS, allow_remote_code=True)
    assert runnable == list(REMOTE_CODE_MODELS)
    assert skipped == []


def test_default_roster_screens_cleanly_in_both_modes():
    declined, skipped = screen_candidates(DEFAULT_LOCAL_CANDIDATES)
    opted_in, none_skipped = screen_candidates(DEFAULT_LOCAL_CANDIDATES, allow_remote_code=True)
    assert opted_in == DEFAULT_LOCAL_CANDIDATES and none_skipped == []
    # Without the opt-in the roster still ranks; only the remote-code rows drop out.
    assert set(declined) | {row["model"] for row in skipped} == set(DEFAULT_LOCAL_CANDIDATES)
    assert {row["model"] for row in skipped} == set(REMOTE_CODE_MODELS)


def test_screening_preserves_roster_order():
    roster = ["BAAI/bge-m3", "jinaai/jina-embeddings-v3", "intfloat/multilingual-e5-small"]
    runnable, _ = screen_candidates(roster)
    assert runnable == ["BAAI/bge-m3", "intfloat/multilingual-e5-small"]
