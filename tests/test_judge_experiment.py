import json

import pytest

from llb.judge.experiment import EXPERIMENT_CASES, run_judge_experiment


def test_judge_experiment_records_prompts_scores_and_endpoint(tmp_path):
    def scorer(records, model):
        assert records == EXPERIMENT_CASES
        assert model == "hosted_vllm/test-judge"
        return [
            {"faithfulness": 1.0, "answer_relevancy": 1.0},
            {"faithfulness": 0.0, "answer_relevancy": 1.0},
            {"faithfulness": 1.0, "answer_relevancy": 0.0},
        ]

    report, out_path = run_judge_experiment(
        "hosted_vllm/test-judge",
        base_url="http://localhost:8000",
        data_dir=tmp_path,
        scorer=scorer,
    )

    assert out_path.parent.parent == tmp_path / "judge-experiment"
    assert json.loads(out_path.read_text(encoding="utf-8")) == report
    assert report["judge"]["provider"] == "deepeval-geval"
    assert report["judge"]["base_url"] == "http://localhost:8000/v1"
    assert len(report["cases"]) == 3
    assert report["prompts"]["faithfulness_steps"]
    assert "Ти оцінювач україномовної RAG-системи" in report["prompts"]["result_template"]


def test_judge_experiment_rejects_missing_scores(tmp_path):
    with pytest.raises(ValueError, match="1 scores for 3"):
        run_judge_experiment(
            "judge",
            data_dir=tmp_path,
            scorer=lambda _records, _model: [{"faithfulness": 1.0, "answer_relevancy": 1.0}],
        )
