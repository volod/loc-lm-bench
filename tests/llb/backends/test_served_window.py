"""Unit tests for live served-window probing and min(declared, served) budget binding."""

import json

from llb.backends.served_window import (
    BUDGET_SOURCE_DECLARED,
    BUDGET_SOURCE_SERVED,
    BUDGET_SOURCE_UNBOUNDED,
    bind_window,
    parse_ollama_served_context,
    probe_served_max_model_len,
)
from llb.bench.agentic.context_budget import resolve_context_budget
from llb.core.config import RunConfig
from llb.optimize.tuning_space import CHARS_PER_TOKEN, PROMPT_HEADROOM_TOKENS


def test_parse_ollama_served_context_reads_loaded_model_context():
    body = json.dumps(
        {
            "models": [
                {"name": "other:latest", "context": 2048},
                {"name": "mamaylm:latest", "context_length": 4096},
            ]
        }
    )
    assert parse_ollama_served_context(body, "mamaylm:latest") == 4096
    assert parse_ollama_served_context(body, "mamaylm") == 4096
    assert parse_ollama_served_context(body, "missing") is None
    assert parse_ollama_served_context("not-json", "mamaylm") is None


def test_bind_window_names_which_side_bound():
    assert bind_window(32768, 4096) == (4096, BUDGET_SOURCE_SERVED)
    assert bind_window(4096, 32768) == (4096, BUDGET_SOURCE_DECLARED)
    assert bind_window(8192, None) == (8192, BUDGET_SOURCE_DECLARED)
    assert bind_window(0, 4096) == (4096, BUDGET_SOURCE_SERVED)
    assert bind_window(0, None) == (0, BUDGET_SOURCE_UNBOUNDED)


def test_probe_served_max_model_len_dispatches_per_backend():
    def fake_get(url: str):
        if url.endswith("/api/ps"):
            return 200, json.dumps({"models": [{"name": "m:latest", "context": 4096}]})
        if url.endswith("/v1/models"):
            return 200, json.dumps({"data": [{"max_model_len": 8192}]})
        if url.endswith("/props"):
            return 200, json.dumps({"n_ctx": 2048})
        return None

    assert (
        probe_served_max_model_len("ollama", model="m", host="http://h", http_get=fake_get) == 4096
    )
    assert probe_served_max_model_len("vllm", model="m", host="http://h", http_get=fake_get) == 8192
    assert (
        probe_served_max_model_len("llamacpp", model="m", host="http://h", http_get=fake_get)
        == 2048
    )
    assert (
        probe_served_max_model_len("ollama", model="m", host="http://h", http_get=lambda _u: None)
        is None
    )


def test_resolve_context_budget_is_bound_by_a_smaller_served_window():
    config = RunConfig().with_overrides(model="unlisted-model", context_budget=32768)
    budget = resolve_context_budget(
        config, model_spec=None, vram_mib=0, ram_mib=0, served_max_model_len=4096
    )
    usable = 4096 - PROMPT_HEADROOM_TOKENS - config.max_tokens
    assert budget.budget_source == BUDGET_SOURCE_SERVED
    assert budget.served_max_model_len == 4096
    assert budget.declared_max_model_len == 32768
    assert budget.max_prompt_chars == int(usable * CHARS_PER_TOKEN)
    assert budget.fits(budget.max_prompt_chars) is True
    assert budget.fits(budget.max_prompt_chars + int(2 * CHARS_PER_TOKEN)) is False


def test_resolve_context_budget_falls_back_to_declared_when_probe_misses():
    config = RunConfig().with_overrides(model="unlisted-model", context_budget=8192)

    def miss(_url: str):
        return None

    budget = resolve_context_budget(
        config, model_spec=None, vram_mib=0, ram_mib=0, probe=True, http_get=miss
    )
    usable = 8192 - PROMPT_HEADROOM_TOKENS - config.max_tokens
    assert budget.budget_source == BUDGET_SOURCE_DECLARED
    assert budget.served_max_model_len is None
    assert budget.declared_max_model_len == 8192
    assert budget.max_prompt_chars == int(usable * CHARS_PER_TOKEN)
    assert budget.provenance()["budget_source"] == BUDGET_SOURCE_DECLARED


def test_ollama_launcher_sends_num_ctx_in_options(monkeypatch):
    from llb.backends.ollama import OllamaLauncher

    captured: dict[str, object] = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def read(self):
            return json.dumps(
                {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}
            ).encode()

    def fake_urlopen(request, timeout=0):  # noqa: ARG001
        captured["url"] = getattr(request, "full_url", None) or request.get_full_url()
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr("llb.backends.ollama.urllib.request.urlopen", fake_urlopen)
    launcher = OllamaLauncher("m", num_ctx=8192)
    result = launcher.chat(
        [{"role": "user", "content": "hi"}], max_tokens=16, temperature=0.0, timeout=5
    )
    assert result.text == "ok"
    assert captured["payload"]["options"]["num_ctx"] == 8192
    assert captured["payload"]["options"]["num_predict"] == 16
