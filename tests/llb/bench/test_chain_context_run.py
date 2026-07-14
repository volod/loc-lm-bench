"""Tests for chain context run."""

import pytest
from llb.bench import chain_context as cc
from llb.bench import chain_context_policy as policy
from llb.scoring.aggregate import TIER_CHAIN_CONTEXT
from test_chain_context import FakeRetriever, _chunk, _grounded_complete, _two_step_chain


def test_run_chain_context_ranks_policies_and_records_provenance(tmp_path):
    chains = [_two_step_chain("chain-0"), _two_step_chain("chain-1")]
    retriever = FakeRetriever([_chunk("контекст")])
    run = cc.run_chain_context(
        chains,
        model="cand",
        backend="ollama",
        retriever=retriever,
        complete=_grounded_complete(chains),
        data_dir=tmp_path,
    )
    # one report + one ranked row per policy
    assert {r.policy for r in run.reports} == set(cc.CONTEXT_POLICIES)
    assert len(run.board) == len(cc.CONTEXT_POLICIES)
    # each report scores every step, and final objectives = one per chain
    for report in run.reports:
        assert len(report.step_rows) == 4  # 2 chains x 2 steps
        assert len(report.final_objectives) == 2
        assert report.final_ci is not None  # >= 2 points -> a bootstrap CI exists
    # carryover policies beat fresh on the final step (which needs step-1 context)
    final = {r.policy: sum(r.final_objectives) / len(r.final_objectives) for r in run.reports}
    assert final[policy.POLICY_HISTORY] > final[policy.POLICY_FRESH]
    # the winning row is a carryover policy, and the recommendation names it
    assert run.board[0]["model"] != policy.POLICY_FRESH
    assert run.board[0]["model"] in run.recommendation
    # persisted provenance records policy, prompt-system ids, and the chain digest
    import json
    from pathlib import Path

    for report in run.reports:
        manifest = json.loads(Path(report.paths["manifest"]).read_text(encoding="utf-8"))
        config = manifest["config"]
        assert config["tier"] == TIER_CHAIN_CONTEXT
        assert config["policy"] == report.policy
        assert config["prompt_system_ids"] == policy.prompt_system_ids(report.policy)
        assert config["chain_set_digest"] == run.chain_digest


def test_run_chain_context_rejects_unknown_policy(tmp_path):
    with pytest.raises(SystemExit):
        cc.run_chain_context(
            [_two_step_chain()],
            model="m",
            backend="ollama",
            retriever=FakeRetriever([_chunk("c")]),
            complete=lambda p: "x",
            policies=["fresh", "bogus"],
            persist=False,
        )


def test_run_chain_context_empty_chains_raises():
    with pytest.raises(SystemExit):
        cc.run_chain_context(
            [],
            model="m",
            backend="ollama",
            retriever=FakeRetriever([]),
            complete=lambda p: "x",
            persist=False,
        )


def test_data_verified_requires_verification_ref(tmp_path):
    with pytest.raises(ValueError):
        cc.run_chain_context(
            [_two_step_chain()],
            model="m",
            backend="ollama",
            retriever=FakeRetriever([_chunk("c")]),
            complete=lambda p: "x",
            persist=False,
            data_verified=True,
            verification_ref=None,
        )


def test_chain_set_digest_is_content_sensitive():
    a = [_two_step_chain("chain-0")]
    b = [_two_step_chain("chain-9")]
    assert cc.chain_set_digest(a) != cc.chain_set_digest(b)
    assert cc.chain_set_digest(a) == cc.chain_set_digest([_two_step_chain("chain-0")])


def test_board_and_recommend_load_persisted_policies(tmp_path):
    from llb.board.chain_context import chain_context_comparison, load_chain_context_records
    from llb.board.recommend.sections import format_chain_context_section_md, latest_chain_context

    chains = [_two_step_chain("chain-0"), _two_step_chain("chain-1")]
    cc.run_chain_context(
        chains,
        model="cand",
        backend="ollama",
        retriever=FakeRetriever([_chunk("контекст")]),
        complete=_grounded_complete(chains),
        data_dir=tmp_path,
    )
    records = load_chain_context_records(tmp_path)
    assert {(r.model, r.policy) for r in records} == {("cand", p) for p in cc.CONTEXT_POLICIES}
    rows, table, policies = chain_context_comparison(tmp_path, "cand")
    assert {row["model"] for row in rows} == set(cc.CONTEXT_POLICIES)
    assert sorted(policies) == sorted(cc.CONTEXT_POLICIES)
    assert "policy:" in table  # ranking-policy note rendered
    # fresh cannot be the top row (it loses the dependent final step)
    top = next(row for row in rows if row["rank"] == 1)
    assert top["model"] != policy.POLICY_FRESH

    section = format_chain_context_section_md(latest_chain_context(tmp_path))
    assert "## Context policy" in section
    assert "cand" in section


def test_recommend_section_empty_without_bundles(tmp_path):
    from llb.board.recommend.sections import format_chain_context_section_md, latest_chain_context

    assert latest_chain_context(tmp_path) is None
    assert format_chain_context_section_md(None) == ""
