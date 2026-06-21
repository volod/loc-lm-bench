"""OpenAI-compatible chat call with normalized error mapping.

Wraps the `openai` SDK (a base dependency) so transport failures become `ChatResult`s
with a normalized `error` token instead of leaking SDK exceptions into the eval graph.
The `client` is injectable, so error mapping is unit-testable against a fake that raises
the SDK exception types without any server.
"""

import time
from typing import Any

import openai

from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT, ChatResult
from llb.contracts import ChatMessage


def make_client(base_url: str, api_key: str = "not-needed") -> openai.OpenAI:
    """An OpenAI SDK client pointed at a local OpenAI-compatible endpoint."""
    return openai.OpenAI(base_url=base_url, api_key=api_key)


def chat_once(
    client: Any,
    model: str,
    messages: list[ChatMessage],
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> ChatResult:
    """One chat completion. Maps timeouts/transport errors to ChatResult.error."""
    start = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
    except openai.APITimeoutError:
        return ChatResult(text="", latency_s=time.monotonic() - start, error=ERR_TIMEOUT)
    except openai.APIError:
        return ChatResult(text="", latency_s=time.monotonic() - start, error=ERR_BACKEND)

    latency = time.monotonic() - start
    text = (resp.choices[0].message.content or "") if resp.choices else ""
    usage = getattr(resp, "usage", None)
    return ChatResult(
        text=text,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        latency_s=latency,
    )
