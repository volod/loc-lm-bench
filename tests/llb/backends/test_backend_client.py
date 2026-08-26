import json
import types

import openai
import pytest

from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT
from llb.backends.ollama import OllamaLauncher
from llb.backends.openai_client import chat_once


def client_with(create_fn):
    completions = types.SimpleNamespace(create=create_fn)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))


def test_success_parses_text_and_usage():
    def create(**kwargs):
        msg = types.SimpleNamespace(content="Київ")
        usage = types.SimpleNamespace(prompt_tokens=12, completion_tokens=4)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)

    result = chat_once(client_with(create), "m", [{"role": "user", "content": "hi"}])
    assert result.error is None
    assert result.text == "Київ"
    assert result.prompt_tokens == 12 and result.completion_tokens == 4


def test_timeout_maps_to_timeout_token():
    def create(**kwargs):
        raise openai.APITimeoutError(request=object())

    result = chat_once(client_with(create), "m", [])
    assert result.error == ERR_TIMEOUT and result.text == ""


def test_connection_error_maps_to_backend_error():
    def create(**kwargs):
        raise openai.APIConnectionError(request=object())

    assert chat_once(client_with(create), "m", []).error == ERR_BACKEND


def test_unexpected_exception_is_not_hidden_as_transport_failure():
    def create(**kwargs):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        chat_once(client_with(create), "m", [])


def test_ollama_launcher_disables_thinking_for_bounded_scoring(monkeypatch):
    seen: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {
                    "message": {"content": "answer"},
                    "prompt_eval_count": 7,
                    "eval_count": 3,
                }
            ).encode()

    def fake_urlopen(request, timeout):
        seen["payload"] = json.loads(request.data)
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("llb.backends.ollama.urllib.request.urlopen", fake_urlopen)
    launcher = OllamaLauncher("qwen3.6:27b")

    result = launcher.chat(
        [{"role": "user", "content": "question"}],
        max_tokens=32,
        temperature=0.0,
        timeout=5.0,
    )

    assert result.text == "answer"
    assert result.prompt_tokens == 7 and result.completion_tokens == 3
    assert seen["payload"]["think"] is False
    assert seen["payload"]["options"] == {"num_predict": 32, "temperature": 0.0}


def test_ollama_launcher_sends_sampling_seed(monkeypatch):
    seen: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"message": {"content": "answer"}}).encode()

    def fake_urlopen(request, timeout):  # noqa: ARG001
        seen["payload"] = json.loads(request.data)
        return FakeResponse()

    monkeypatch.setattr("llb.backends.ollama.urllib.request.urlopen", fake_urlopen)
    launcher = OllamaLauncher("model", seed=29)
    launcher.chat([], max_tokens=16, temperature=0.2, timeout=5.0)

    assert seen["payload"]["options"] == {
        "num_predict": 16,
        "temperature": 0.2,
        "seed": 29,
    }


def test_ollama_names_an_architecture_the_daemon_cannot_serve(monkeypatch):
    """An `unknown model architecture` refusal is a host fact, not an anonymous backend error."""
    import io
    import urllib.error

    from llb.backends.base import ERR_ARCH_UNSUPPORTED

    def fake_urlopen(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            url="http://localhost:11434/api/chat",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"unknown model architecture: \'gemma4\'"}'),
        )

    monkeypatch.setattr("llb.backends.ollama.urllib.request.urlopen", fake_urlopen)
    result = OllamaLauncher("gemma4:12b").chat([], max_tokens=8, temperature=0.0, timeout=5.0)
    assert result.error == ERR_ARCH_UNSUPPORTED


def test_ollama_keeps_an_unrelated_http_error_generic(monkeypatch):
    import io
    import urllib.error

    def fake_urlopen(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            url="http://localhost:11434/api/chat",
            code=500,
            msg="Server Error",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"llama runner process has terminated"}'),
        )

    monkeypatch.setattr("llb.backends.ollama.urllib.request.urlopen", fake_urlopen)
    result = OllamaLauncher("qwen3.6:27b").chat([], max_tokens=8, temperature=0.0, timeout=5.0)
    assert result.error == ERR_BACKEND


def test_ollama_start_refuses_a_model_the_daemon_is_too_old_for(monkeypatch):
    """The floor is checked once at launch, so a doomed run fails with the version it needs."""
    from llb.backends.runtime_floor import RuntimeRequirement

    launcher = OllamaLauncher("gemma4:12b")
    monkeypatch.setattr(launcher, "_reachable", lambda: True)
    monkeypatch.setattr(
        "llb.backends.ollama.artifact_requirement",
        lambda *_a, **_k: RuntimeRequirement("ollama", "gemma4:12b", "gemma4", "0.30.5"),
    )
    monkeypatch.setattr("llb.backends.ollama.runtime_version", lambda *_a, **_k: "0.20.6")

    with pytest.raises(RuntimeError, match=r"ollama >= 0\.30\.5"):
        launcher.start()
