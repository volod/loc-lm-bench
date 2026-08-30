import json

import pytest

from llb.robotics.benchmark.parser import parse_model_decision
from llb.robotics.benchmark.profile import load_measured_profile


def test_parser_accepts_one_typed_fenced_object_and_rejects_trailing_text():
    payload = '{"decision":"refuse","reason":"unsafe","proposal":null}'
    assert parse_model_decision(f"```json\n{payload}\n```").decision == "refuse"

    with pytest.raises(ValueError, match="trailing text"):
        parse_model_decision(payload + " do it")


def test_profile_consumes_only_measured_fields_and_refuses_measured_model_mixing(tmp_path):
    path = tmp_path / "agent_profile.json"
    fields = {
        "model": {"state": "measured", "value": "model-a"},
        "backend": {"state": "measured", "value": "ollama"},
        "adapter": {"state": "measured", "value": "none"},
        "top_k": {"state": "demoted", "value": 99},
        "context_budget": {"state": "measured", "value": 8192},
    }
    path.write_text(json.dumps({"generated_at": "now", "fields": fields}), encoding="utf-8")

    loaded = load_measured_profile(path, model="model-a", backend="ollama")
    assert loaded["measured_fields"] == {
        "model": "model-a",
        "backend": "ollama",
        "adapter": "none",
        "context_budget": 8192,
    }
    assert loaded["excluded_field_states"] == {"top_k": "demoted"}
    with pytest.raises(ValueError, match="conflicts"):
        load_measured_profile(path, model="model-b", backend="ollama")
