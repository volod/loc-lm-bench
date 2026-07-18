"""Runner wiring for the scorer-policy seam."""

import json
from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.executor.runner_judge import (
    JudgeCaseResult,
    _build_judge_metadata,
    _judge_cases,
)
from llb.goldset.schema import GoldItem, SourceSpan
from llb.scoring.policy import wrap_llm_complete
from llb.scoring.policy.resolve import ScorerPolicyRequest, resolve_scorer


class _FakeBatch:
    def __init__(self):
        self.answers = []
        self.retrieval_pairs = []
        self.rows = []


def _item() -> GoldItem:
    return GoldItem(
        id="1",
        lang="uk",
        question="Столиця?",
        reference_answer="Київ",
        source_doc_id="d",
        source_spans=[SourceSpan(doc_id="d", char_start=0, char_end=4, text="Київ")],
        provenance="human-authored",
        verified=True,
        split="final",
    )


def test_human_policy_skips_judge(tmp_path):
    cfg = RunConfig(data_dir=tmp_path, scorer_policy="human")
    result = _judge_cases(cfg, _FakeBatch(), judge_rho=0.9, scorer=None, staging_dir=tmp_path)
    assert isinstance(result, JudgeCaseResult)
    assert result.mean_score is None
    assert result.policy_metadata["scorer_policy"] == "human"
    meta = _build_judge_metadata(cfg, 0.9, result.policy_metadata)
    assert meta["scorer_policy"] == "human"
    assert meta["provider"] == "human"


def test_local_policy_routes_injected_scorer(tmp_path):
    item = _item()
    batch = _FakeBatch()
    batch.answers = [(item, "Київ")]
    batch.retrieval_pairs = [([{"text": "Київ — столиця."}], [])]
    batch.rows = [{"id": "1"}]
    cfg = RunConfig(data_dir=tmp_path, scorer_policy="local", judge_model="local-judge")

    def scorer(records, model):
        assert model == "local-judge"
        return [{"faithfulness": 1.0, "answer_relevancy": 1.0} for _ in records]

    result = _judge_cases(cfg, batch, judge_rho=0.9, scorer=scorer, staging_dir=tmp_path)
    assert result.mean_score == pytest.approx(1.0)
    assert batch.rows[0]["judge_score"] == 1.0


def test_frontier_budget_abort_writes_artifact(tmp_path):
    item = _item()
    batch = _FakeBatch()
    batch.answers = [(item, "Київ"), (item, "Київ")]
    batch.retrieval_pairs = [
        ([{"text": "Київ — столиця."}], []),
        ([{"text": "Київ — столиця."}], []),
    ]
    batch.rows = [{"id": "1"}, {"id": "2"}]
    staging = tmp_path / "staging"
    staging.mkdir()
    cfg = RunConfig(
        data_dir=tmp_path,
        scorer_policy="frontier",
        judge_model="openai/gpt-test",
        scorer_egress_consent=True,
        frontier_max_usd=0.015,
    )
    resolved = resolve_scorer(
        ScorerPolicyRequest(
            lane="frontier",
            judge_model="openai/gpt-test",
            egress_consent=True,
            max_usd=0.015,
            run_dir=staging,
            frontier_complete=wrap_llm_complete(
                lambda _prompt: '{"faithfulness": 1.0, "answer_relevancy": 1.0}',
                cost_usd=0.01,
            ),
        )
    )
    with pytest.raises(SystemExit, match="budget exceeded"):
        _judge_cases(
            cfg, batch, judge_rho=0.9, scorer=resolved.scorer, staging_dir=staging
        )
    abort = json.loads((staging / "scorer" / "abort.json").read_text(encoding="utf-8"))
    assert abort["resumable"] is True
    assert abort["status"] == "aborted"
    assert Path(staging / "scorer" / "ledger_state.json").is_file()
