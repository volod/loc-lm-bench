"""Scorer-policy seam: lane routing, consent, cost ledger, and frontier judge."""

import json

import pytest

from llb.core.config import RunConfig
from llb.scoring.policy import (
    BudgetExceeded,
    ConsentRecord,
    CostLedger,
    HUMAN_LANE_REASON,
    LedgerEntry,
    ScorerPolicyError,
    ScorerPolicyRequest,
    build_frontier_judge_prompt,
    frontier_scorer,
    human_scorer,
    load_consent,
    parse_frontier_judge_response,
    parse_scorer_lane,
    record_consent,
    resolve_scorer,
    wrap_llm_complete,
)


def _record(answer: str = "Київ") -> dict:
    return {
        "question": "Столиця України?",
        "answer": answer,
        "contexts": ["Київ — столиця України."],
    }


def test_parse_scorer_lane_defaults_and_rejects():
    assert parse_scorer_lane(None) == "local"
    assert parse_scorer_lane("frontier") == "frontier"
    with pytest.raises(ScorerPolicyError, match="scorer_policy"):
        parse_scorer_lane("cloud")


def test_policy_matrix_routes_human_local_frontier(tmp_path):
    human = resolve_scorer(
        ScorerPolicyRequest(lane="human", judge_model=None, run_dir=tmp_path)
    )
    assert human.lane == "human"
    assert HUMAN_LANE_REASON in human.reason
    assert human.scorer([_record()], "n/a") == [
        {"faithfulness": 0.0, "answer_relevancy": 0.0}
    ]

    local = resolve_scorer(
        ScorerPolicyRequest(
            lane="local",
            judge_model="local-judge",
            local_scorer=lambda records, model: [
                {"faithfulness": 1.0, "answer_relevancy": 0.5} for _ in records
            ],
        )
    )
    assert local.lane == "local"
    assert local.metadata["model"] == "local-judge"
    assert local.scorer([_record()], "local-judge")[0]["faithfulness"] == 1.0

    frontier = resolve_scorer(
        ScorerPolicyRequest(
            lane="frontier",
            judge_model="openai/gpt-test",
            egress_consent=True,
            max_usd=1.0,
            max_calls=5,
            run_dir=tmp_path / "frontier-run",
            frontier_complete=wrap_llm_complete(
                lambda _prompt: '{"faithfulness": 0.9, "answer_relevancy": 0.8}',
                cost_usd=0.01,
            ),
        )
    )
    assert frontier.lane == "frontier"
    assert frontier.ledger is not None
    scores = frontier.scorer([_record()], "openai/gpt-test")
    assert scores == [{"faithfulness": 0.9, "answer_relevancy": 0.8}]
    assert frontier.ledger.calls == 1
    assert frontier.ledger.cost_usd == pytest.approx(0.01)


def test_frontier_requires_consent_and_budget(tmp_path):
    with pytest.raises(ScorerPolicyError, match="approved consent"):
        resolve_scorer(
            ScorerPolicyRequest(
                lane="frontier",
                judge_model="openai/gpt-test",
                egress_consent=False,
                max_usd=1.0,
                run_dir=tmp_path,
            )
        )
    with pytest.raises(ScorerPolicyError, match="max_usd or max_calls"):
        record_consent(tmp_path, model="m", approved=True, max_usd=None, max_calls=None)


def test_ledger_enforces_call_cap_and_aborts(tmp_path):
    ledger = CostLedger.open(tmp_path, max_usd=None, max_calls=2)
    ledger.reserve_call()
    ledger.record(LedgerEntry(model="m", prompt_tokens=1, completion_tokens=1, cost_usd=0.0))
    ledger.reserve_call()
    ledger.record(LedgerEntry(model="m", prompt_tokens=1, completion_tokens=1, cost_usd=0.0))
    with pytest.raises(BudgetExceeded, match="call budget exhausted"):
        ledger.reserve_call()


def test_ledger_enforces_spend_cap(tmp_path):
    ledger = CostLedger.open(tmp_path, max_usd=0.05, max_calls=10)
    ledger.reserve_call()
    with pytest.raises(BudgetExceeded, match="spend budget exceeded"):
        ledger.record(
            LedgerEntry(model="m", prompt_tokens=10, completion_tokens=5, cost_usd=0.06)
        )


def test_ledger_resume_restores_spend(tmp_path):
    first = CostLedger.open(tmp_path, max_usd=1.0, max_calls=10)
    first.reserve_call()
    first.record(LedgerEntry(model="m", prompt_tokens=3, completion_tokens=2, cost_usd=0.25))
    resumed = CostLedger.open(tmp_path, max_usd=1.0, max_calls=10)
    assert resumed.calls == 1
    assert resumed.cost_usd == pytest.approx(0.25)
    assert resumed.remaining_usd() == pytest.approx(0.75)
    summary = resumed.summary()
    assert summary["resumable"] is True
    assert (tmp_path / "scorer" / "ledger.jsonl").is_file()


def test_frontier_scorer_aborts_cleanly_at_cap(tmp_path):
    ledger = CostLedger.open(tmp_path, max_usd=0.015, max_calls=10)
    scorer = frontier_scorer(
        "openai/gpt-test",
        ledger,
        complete=wrap_llm_complete(
            lambda _prompt: '{"faithfulness": 1.0, "answer_relevancy": 1.0}',
            cost_usd=0.01,
        ),
    )
    scorer([_record()], "openai/gpt-test")
    with pytest.raises(BudgetExceeded):
        scorer([_record()], "openai/gpt-test")
    abort = ledger.abort_payload("cap")
    assert abort["status"] == "aborted"
    assert abort["resumable"] is True
    assert abort["calls"] == 2


def test_consent_round_trip(tmp_path):
    record = record_consent(
        tmp_path, model="openai/gpt-test", approved=True, max_usd=0.5, max_calls=20
    )
    loaded = load_consent(tmp_path)
    assert loaded == record
    assert isinstance(loaded, ConsentRecord)
    assert loaded.approved is True


def test_frontier_prompt_reuses_ua_steps():
    prompt = build_frontier_judge_prompt(_record())
    assert "faithfulness" in prompt
    assert "Київ" in prompt
    assert "вірності" in prompt or "Критерії" in prompt


def test_parse_frontier_judge_response():
    assert parse_frontier_judge_response(
        'note\n{"faithfulness": 0.7, "answer_relevancy": 0.6}\n'
    ) == {"faithfulness": 0.7, "answer_relevancy": 0.6}
    assert parse_frontier_judge_response("not json") == {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
    }


def test_human_scorer_zeros():
    assert human_scorer()([_record(""), _record()], "x") == [
        {"faithfulness": 0.0, "answer_relevancy": 0.0},
        {"faithfulness": 0.0, "answer_relevancy": 0.0},
    ]


def test_run_config_frontier_validation(tmp_path):
    with pytest.raises(ValueError, match="frontier_max_usd or frontier_max_calls"):
        RunConfig(
            data_dir=tmp_path,
            scorer_policy="frontier",
            judge_model="openai/gpt-test",
            scorer_egress_consent=True,
        )
    cfg = RunConfig(
        data_dir=tmp_path,
        scorer_policy="frontier",
        judge_model="openai/gpt-test",
        scorer_egress_consent=True,
        frontier_max_usd=1.0,
    )
    assert cfg.scorer_policy == "frontier"
    with pytest.raises(ValueError, match="scorer_egress_consent"):
        RunConfig(data_dir=tmp_path, scorer_policy="local", scorer_egress_consent=True)


def test_frontier_empty_answer_skips_spend(tmp_path):
    ledger = CostLedger.open(tmp_path, max_usd=1.0, max_calls=5)
    called = []

    def complete(prompt: str):
        called.append(prompt)
        return '{"faithfulness": 1.0, "answer_relevancy": 1.0}', 0.1, 1, 1

    scorer = frontier_scorer("m", ledger, complete=complete)
    scores = scorer([_record(""), _record("Київ")], "m")
    assert scores[0] == {"faithfulness": 0.0, "answer_relevancy": 0.0}
    assert scores[1]["faithfulness"] == 1.0
    assert len(called) == 1
    assert ledger.calls == 1
