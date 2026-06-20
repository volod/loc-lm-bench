import types

import openai

from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT
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


def test_unexpected_exception_maps_to_backend_error():
    def create(**kwargs):
        raise ValueError("boom")

    assert chat_once(client_with(create), "m", []).error == ERR_BACKEND
