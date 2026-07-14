"""Tests for ontology endpoint."""

import pytest
from llb.backends.base import ChatResult
from llb.prep.frontier_telemetry import ProvenanceLog
import llb.prep.ontology.endpoint as ep
from llb.prep.ontology.endpoint import build_complete
from llb.prep.ontology.endpoint_config import EndpointConfig


def test_endpoint_config_validates_kind_model_and_egress():
    with pytest.raises(ValueError, match="endpoint kind"):
        EndpointConfig(kind="cloud", model="m")
    with pytest.raises(ValueError, match="model must be set"):
        EndpointConfig(kind="local", model="")
    with pytest.raises(ValueError, match="local backend"):
        EndpointConfig(kind="local", model="m", backend="bad")
    with pytest.raises(ValueError, match="local backend can only"):
        EndpointConfig(
            kind="frontier", model="gpt", backend="vllm", egress_consent=True, max_calls=10
        )
    with pytest.raises(ValueError, match="explicit egress consent"):
        EndpointConfig(kind="frontier", model="gpt")
    with pytest.raises(ValueError, match="egress consent can only"):
        EndpointConfig(kind="local", model="m", egress_consent=True)
    assert EndpointConfig(kind="local", model="m").egress is False
    assert EndpointConfig(kind="local", model="m").provenance()["backend"] == "ollama"
    frontier = EndpointConfig(kind="frontier", model="gpt", egress_consent=True, max_calls=10)
    assert frontier.egress is True and frontier.provenance()["egress"] is True


def test_build_complete_local_records_tokens_and_raises_on_error(monkeypatch):
    monkeypatch.setattr(ep, "make_client", lambda base_url, api_key="x": object())
    monkeypatch.setattr(
        ep, "chat_once", lambda *a, **k: ChatResult(text="OK", prompt_tokens=5, completion_tokens=2)
    )
    log = ProvenanceLog()
    cfg = EndpointConfig(kind="local", model="m", backend="openai")
    complete = build_complete(cfg, log)
    assert complete("hi") == "OK"
    summary = log.summary()
    assert summary["calls"] == 1 and summary["total_prompt_tokens"] == 5

    monkeypatch.setattr(ep, "chat_once", lambda *a, **k: ChatResult(text="", error="timeout"))
    with pytest.raises(RuntimeError, match="local endpoint error"):
        build_complete(cfg, ProvenanceLog())("hi")


def test_native_chat_url_maps_v1_to_api_chat():
    assert ep._native_chat_url("http://localhost:11434/v1") == "http://localhost:11434/api/chat"
    assert ep._native_chat_url("http://h:8000/v1/") == "http://h:8000/api/chat"
    assert ep._native_chat_url("http://h:11434") == "http://h:11434/api/chat"


def test_think_disabled_routes_through_native_endpoint(monkeypatch):
    # think is honored only by Ollama's native /api/chat, so a think-set config must NOT use /v1
    monkeypatch.setattr(
        ep, "make_client", lambda *a, **k: pytest.fail("must not use the /v1 client when think set")
    )
    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, object]:
            return {"message": {"content": "OK"}, "prompt_eval_count": 7, "eval_count": 3}

    def fake_post(url, json, timeout):  # noqa: A002 - mirror httpx.post signature
        captured["url"] = url
        captured["think"] = json["think"]
        captured["format"] = json["format"]
        captured["num_predict"] = json["options"]["num_predict"]
        return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    log = ProvenanceLog()
    cfg = EndpointConfig(kind="local", model="gemma4:26b", think=False, max_tokens=4096)
    assert cfg.provenance()["think"] is False
    assert build_complete(cfg, log)("hi") == "OK"
    assert captured["url"].endswith("/api/chat")
    assert captured["think"] is False and captured["num_predict"] == 4096
    assert captured["format"] == "json"
    assert log.summary()["total_prompt_tokens"] == 7


def test_vllm_think_disabled_uses_openai_extra_body(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ep, "make_client", lambda *a, **k: sentinel)
    captured: dict[str, object] = {}

    def fake_chat_once(client, model, messages, **kwargs):
        captured["client"] = client
        captured["extra_body"] = kwargs.get("extra_body")

        class _Result:
            text = "OK"
            prompt_tokens = 11
            completion_tokens = 4
            error = None

        return _Result()

    monkeypatch.setattr(ep, "chat_once", fake_chat_once)
    cfg = EndpointConfig(
        kind="local",
        backend="vllm",
        model="hf/reasoning-model",
        base_url="http://localhost:8000/v1",
        think=False,
    )
    assert cfg.provenance()["backend"] == "vllm"
    assert build_complete(cfg, ProvenanceLog())("hi") == "OK"
    extra_body = captured["extra_body"]
    assert isinstance(extra_body, dict)
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra_body["include_reasoning"] is False
    assert extra_body["reasoning_effort"] == "none"
    assert captured["client"] is sentinel


def test_vllm_host_for_port_rewrites_default_host():
    from llb.cli.prep.draft_endpoints import _vllm_host_for_port

    assert _vllm_host_for_port("http://localhost:8000", 8010) == "http://localhost:8010"


def test_num_ctx_routes_through_native_endpoint_and_bounds_context(monkeypatch):
    # num_ctx (like think) exists only on Ollama's native /api/chat; a num_ctx-set config must
    # right-size the loaded context instead of inheriting the modelfile default (CPU offload).
    monkeypatch.setattr(
        ep,
        "make_client",
        lambda *a, **k: pytest.fail("must not use the /v1 client when num_ctx set"),
    )
    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, object]:
            return {"message": {"content": "OK"}, "prompt_eval_count": 5, "eval_count": 2}

    def fake_post(url, json, timeout):  # noqa: A002 - mirror httpx.post signature
        captured["url"] = url
        captured["options"] = json["options"]
        return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    cfg = EndpointConfig(kind="local", model="qwen3.6:35b", num_ctx=16384)
    assert cfg.provenance()["num_ctx"] == 16384
    assert build_complete(cfg, ProvenanceLog())("hi") == "OK"
    assert captured["url"].endswith("/api/chat")
    options = captured["options"]
    assert isinstance(options, dict) and options["num_ctx"] == 16384


def test_default_ollama_config_uses_native_json_mode(monkeypatch):
    monkeypatch.setattr(
        ep, "make_client", lambda *a, **k: pytest.fail("Ollama must use native JSON mode")
    )
    seen: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, object]:
            return {"message": {"content": "{}"}, "prompt_eval_count": 1, "eval_count": 1}

    def fake_post(url, json, timeout):  # noqa: A002 - mirror httpx.post signature
        seen["url"] = url
        seen["payload"] = json
        return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    cfg = EndpointConfig(kind="local", model="any:model")
    assert "num_ctx" not in cfg.provenance()
    assert build_complete(cfg, ProvenanceLog())("hi") == "{}"
    assert str(seen["url"]).endswith("/api/chat")
    payload = seen["payload"]
    assert isinstance(payload, dict) and payload["format"] == "json"
