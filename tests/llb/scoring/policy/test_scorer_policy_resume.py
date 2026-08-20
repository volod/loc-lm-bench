"""Frontier scorer checkpoint and resume tests."""

import json

import pytest

from llb.scoring.policy import BudgetExceeded, CostLedger, LedgerEntry, frontier_scorer


def _record(answer: str) -> dict:
    return {
        "question": "Столиця України?",
        "answer": answer,
        "contexts": ["Київ - столиця України."],
    }


def test_frontier_resume_skips_already_scored_cases(tmp_path):
    """After a mid-batch abort, resume issues calls only for unfinished cases."""
    n_cases = 5
    k_scored = 2
    records = [_record(f"answer-{index}") for index in range(n_cases)]
    calls: list[str] = []

    def complete(prompt: str):
        calls.append(prompt)
        call_number = len(calls)
        return (
            json.dumps(
                {
                    "faithfulness": 0.1 * call_number,
                    "answer_relevancy": 0.2 * call_number,
                }
            ),
            0.01,
            1,
            1,
        )

    first = CostLedger.open(tmp_path, max_usd=None, max_calls=k_scored)
    with pytest.raises(BudgetExceeded, match="call budget exhausted"):
        frontier_scorer("m", first, complete=complete)(records, "m")
    assert len(calls) == k_scored
    assert first.summary()["scored_cases"] == k_scored
    assert set(first.case_scores) == {0, 1}

    calls.clear()
    resumed = CostLedger.open(tmp_path, max_usd=None, max_calls=n_cases)
    assert resumed.calls == k_scored
    assert resumed.scored_case(0) == {"faithfulness": 0.1, "answer_relevancy": 0.2}
    assert resumed.scored_case(1) == {"faithfulness": 0.2, "answer_relevancy": 0.4}
    scores = frontier_scorer("m", resumed, complete=complete)(records, "m")
    assert len(calls) == n_cases - k_scored
    assert len(scores) == n_cases
    assert scores[0] == {"faithfulness": 0.1, "answer_relevancy": 0.2}
    assert scores[1] == {"faithfulness": 0.2, "answer_relevancy": 0.4}
    assert scores[2]["faithfulness"] == pytest.approx(0.1)
    assert resumed.calls == n_cases
    assert resumed.summary()["scored_cases"] == n_cases


def test_ledger_jsonl_persists_case_scores(tmp_path):
    ledger = CostLedger.open(tmp_path, max_usd=1.0, max_calls=5)
    ledger.reserve_call()
    ledger.record(
        LedgerEntry(
            model="m",
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=0.01,
            case_index=3,
            faithfulness=0.55,
            answer_relevancy=0.66,
        )
    )
    line = (tmp_path / "scorer" / "ledger.jsonl").read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["case_index"] == 3
    assert payload["faithfulness"] == 0.55
    assert payload["answer_relevancy"] == 0.66
    resumed = CostLedger.open(tmp_path, max_usd=1.0, max_calls=5)
    assert resumed.scored_case(3) == {"faithfulness": 0.55, "answer_relevancy": 0.66}
