import json
from types import SimpleNamespace

import pytest

from llb.scoring.judge import (
    UA_ANSWER_RELEVANCY_STEPS,
    UA_FAITHFULNESS_STEPS,
    UkrainianGEvalTemplate,
    deepeval_scorer,
    extract_scores,
    judge_experiment_metadata,
    judge_is_trusted,
    resolve_judge_endpoint,
    run_judge,
)


def test_gate_helper():
    assert judge_is_trusted(0.6)
    assert not judge_is_trusted(0.59)
    assert not judge_is_trusted(None)


def test_no_judge_configured_is_demoted():
    out = run_judge([], judge_model=None, calibration_rho=0.9)
    assert out.trusted is False and "no judge" in out.reason


def test_uncalibrated_is_demoted():
    out = run_judge([], judge_model="gpt-judge", calibration_rho=None)
    assert out.trusted is False and "not calibrated" in out.reason


def test_below_threshold_is_demoted():
    out = run_judge([], judge_model="gpt-judge", calibration_rho=0.4)
    assert out.trusted is False and "threshold" in out.reason


def test_trusted_routes_to_scorer():
    out = run_judge(
        [{"q": 1}],
        judge_model="gpt-judge",
        calibration_rho=0.8,
        scorer=lambda recs, model: [{"faithfulness": 1.0}],
    )
    assert out.trusted is True
    assert out.scores == [{"faithfulness": 1.0}]


# --- M3.8 DeepEval scorer (pure extraction + injected evaluate) ---------------------------


def test_extract_scores_normalizes_signals():
    rows = [
        {"faithfulness": 0.8, "answer_relevancy": 0.7},
        {"faithfulness": 0.5},
    ]
    scores = extract_scores(rows)
    assert scores[0] == {"faithfulness": 0.8, "answer_relevancy": 0.7}
    assert scores[1] == {"faithfulness": 0.5, "answer_relevancy": 0.0}


def test_deepeval_scorer_uses_injected_evaluate():
    records = [{"question": "Столиця?", "answer": "Київ", "contexts": ["Київ - столиця."]}]

    def fake_evaluate(received, judge_model):
        assert received == records and judge_model == "judge-x"
        return [{"faithfulness": 1.0, "answer_relevancy": 0.9}]

    scores = deepeval_scorer(records, "judge-x", evaluate_fn=fake_evaluate)
    assert scores == [{"faithfulness": 1.0, "answer_relevancy": 0.9}]


def test_ua_metric_prompts_are_ukrainian():
    assert "контекст" in " ".join(UA_FAITHFULNESS_STEPS)
    assert "запитання" in " ".join(UA_ANSWER_RELEVANCY_STEPS)
    prompt = UkrainianGEvalTemplate.generate_evaluation_results(
        "1. Перевір.", "Фактична відповідь: Київ", "Фактична відповідь"
    )
    assert "Ти оцінювач україномовної RAG-системи" in prompt
    assert '"score"' in prompt and '"reason"' in prompt


def test_resolve_local_judge_endpoint(monkeypatch):
    monkeypatch.setenv("VLLM_HOST", "http://localhost:8000")
    assert resolve_judge_endpoint("hosted_vllm/org/model") == (
        "org/model",
        "http://localhost:8000/v1",
    )
    assert resolve_judge_endpoint("plain-model", "http://judge:9000/v1") == (
        "plain-model",
        "http://judge:9000/v1",
    )


def test_judge_experiment_metadata_has_no_secret():
    metadata = judge_experiment_metadata("ollama_chat/gemma", "http://localhost:11434")
    assert metadata == {
        "provider": "deepeval-geval",
        "model": "gemma",
        "base_url": "http://localhost:11434/v1",
        "prompt_language": "uk",
        "metrics": ["faithfulness", "answer_relevancy"],
    }

    with pytest.raises(ValueError, match="must not contain credentials"):
        judge_experiment_metadata("gemma", "http://user:secret@localhost:11434")


def test_deepeval_scorer_requires_explicit_local_endpoint(monkeypatch):
    monkeypatch.setattr("llb.scoring.judge.load_project_env", lambda: None)
    monkeypatch.delenv("DEEPEVAL_JUDGE_BASE_URL", raising=False)
    with pytest.raises(SystemExit, match="local judge endpoint is required"):
        deepeval_scorer([], "plain-model")


def test_real_deepeval_metric_path_uses_ua_prompts(monkeypatch):
    from deepeval.models import LocalModel

    prompts = []

    def create(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        message = SimpleNamespace(
            content=json.dumps({"score": 10, "reason": "Відповідність повна."}, ensure_ascii=False)
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(LocalModel, "load_model", lambda self, async_mode=False: client)
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    scores = deepeval_scorer(
        [
            {
                "question": "Яка столиця України?",
                "answer": "Київ.",
                "contexts": ["Київ є столицею України."],
            }
        ],
        "test-judge",
        base_url="http://127.0.0.1:8000/v1",
    )

    assert scores == [{"faithfulness": 1.0, "answer_relevancy": 1.0}]
    assert len(prompts) == 2
    assert all("Ти оцінювач україномовної RAG-системи" in prompt for prompt in prompts)
    assert all("Actual Output" not in prompt for prompt in prompts)
    assert all("Фактична відповідь" in prompt for prompt in prompts)
    assert any("контекст" in prompt for prompt in prompts)
    assert any("запитання" in prompt for prompt in prompts)
