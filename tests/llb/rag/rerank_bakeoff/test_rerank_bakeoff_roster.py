"""Reranker convention registry and roster screening (pure; no model, no network)."""

import pytest

from llb.rag.encoders.candidate_screen import SKIP_REMOTE_CODE, UnregisteredCandidateError
from llb.rag.rerank_bakeoff.families import (
    FAMILY_QWEN3_RERANKER,
    FAMILY_UNKNOWN,
    is_registered,
    rerank_family,
    resolve_convention,
)
from llb.rag.rerank_bakeoff.models import DEFAULT_RERANK_CANDIDATES_ROSTER, ROW_NO_RERANK
from llb.rag.rerank_bakeoff.roster import screen_rerankers


def test_every_default_candidate_has_a_declared_convention():
    """The bake-off's own guard rail: a shipped roster entry nobody read the card for is a bug."""
    assert all(is_registered(model) for model in DEFAULT_RERANK_CANDIDATES_ROSTER)


def test_an_id_nobody_registered_resolves_to_unknown_rather_than_a_bare_default():
    assert rerank_family("acme/mystery-reranker") == FAMILY_UNKNOWN
    assert not is_registered("acme/mystery-reranker")


def test_a_reranker_id_never_matches_an_embedding_family():
    """`bge-reranker-v2-m3` carries both `bge` and `m3`; the registry must not read it as an encoder."""
    assert rerank_family("BAAI/bge-reranker-v2-m3") == "bge-reranker"


def test_the_instruct_candidate_records_the_prompt_its_own_config_applies():
    convention = resolve_convention("Qwen/Qwen3-Reranker-0.6B")
    assert convention.family == FAMILY_QWEN3_RERANKER
    assert convention.default_prompt and "query" in convention.default_prompt
    assert convention.source.startswith("https://huggingface.co/")


def test_unregistered_candidates_fail_the_run_with_an_actionable_message():
    with pytest.raises(UnregisteredCandidateError, match="llb.rag.rerank_bakeoff.families"):
        screen_rerankers(["acme/mystery-reranker"])


def test_remote_code_candidates_are_declined_visibly_without_the_opt_in():
    runnable, skipped = screen_rerankers(DEFAULT_RERANK_CANDIDATES_ROSTER)
    declined = {row["model"] for row in skipped}
    assert declined == {
        "jinaai/jina-reranker-v2-base-multilingual",
        "Alibaba-NLP/gte-multilingual-reranker-base",
    }
    assert all(row["reason"] == SKIP_REMOTE_CODE for row in skipped)
    assert all("--allow-remote-code" in row["detail"] for row in skipped)
    assert declined.isdisjoint(runnable)


def test_the_opt_in_runs_the_remote_code_candidates():
    runnable, skipped = screen_rerankers(DEFAULT_RERANK_CANDIDATES_ROSTER, allow_remote_code=True)
    assert skipped == [] and runnable == DEFAULT_RERANK_CANDIDATES_ROSTER


def test_the_reranker_off_row_is_never_screened_as_a_candidate():
    """The lane adds it itself, so spelling it out must not produce the row twice."""
    runnable, skipped = screen_rerankers([ROW_NO_RERANK, "BAAI/bge-reranker-v2-m3"])
    assert runnable == ["BAAI/bge-reranker-v2-m3"] and skipped == []
